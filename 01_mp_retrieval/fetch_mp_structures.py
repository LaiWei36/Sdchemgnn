"""
fetch_mp_structures.py
==========================
Retrieves crystal structures together with TWO graph-level properties —
the formation energy per atom and the total magnetization — from the
Materials Project database and saves them in JSON Lines (.jsonl) format
for downstream graph machine learning workflows.

Graph-level properties retrieved per structure:
  - formation_energy_per_atom : formation energy in eV/atom (intensive)
  - total_magnetization_normalized_formula_units :
                                total magnetization per formula unit in μB/f.u.
                                (intensive — independent of cell size; this is
                                the recommended magnetization regression target)
  - total_magnetization       : total magnetization of the whole cell in μB
                                (Bohr magnetons). This is EXTENSIVE (scales with
                                cell size), so it is kept for reference rather
                                than as the default target.

Any magnetization field may be None/null for the small number of materials
where it was not computed; downstream code should treat a missing value as
"label unavailable". Either property (or both) can later be used as a
regression target when training the ChemGNN model.

Parallelization design
----------------------
The bottleneck is network I/O (waiting for the MP API), not CPU computation.
The approach used is a producer/consumer pipeline with threads:

    ┌─────────────────────────────────────────────────────────────────┐
    │  Main thread                                                    │
    │   └─ MPRester.search()  ──► raw doc chunks ──► work_queue      │
    └─────────────────────────────┬───────────────────────────────────┘
                                  │  (Queue of raw doc batches)
              ┌───────────────────▼───────────────────┐
              │  ThreadPoolExecutor  (N worker threads) │
              │   Each worker:                          │
              │    1. Pops a batch from work_queue      │
              │    2. Generates POSCAR strings (CPU)    │
              │    3. Assembles JSON records            │
              │    4. Pushes results to write_queue     │
              └───────────────────┬───────────────────┘
                                  │  (Queue of serialised JSON lines)
              ┌───────────────────▼───────────────────┐
              │  Single writer thread                   │
              │   Drains write_queue → disk (.jsonl)    │
              └────────────────────────────────────────┘

Why threads (not multiprocessing)?
  - The GIL is released during network I/O, so threads overlap API waits.
  - POSCAR generation is lightweight CPU work — no need for separate processes.
  - A single writer thread avoids file-corruption races with zero locking cost.

Recommended num_workers for the AMD EPYC 7763 node (128 cores, NPS=4):
  - 16–32 threads saturates MP API rate limits without thrashing.
  - The node has 4 NUMA domains; keeping workers ≤ 32 avoids cross-NUMA
    memory traffic for this I/O-bound workload.

Requirements:
    pip install mp-api pymatgen tqdm

Usage:
    from fetch_mp_structures import fetch_mp_structures_dataset

    fetch_mp_structures_dataset(
        api_key="your_MP_api_key_here",
        output_path="./dataset_mp/mp_dataset.jsonl",
        num_workers=16,               # parallel threads (16–32 recommended)
        batch_size=200,               # docs per worker batch
        # optional filters:
        elements=["Li", "Fe", "O"],
        num_elements=(1, 4),
        energy_above_hull_max=0.1,
        chunk_size=1000,
        max_entries=None,
    )
"""

import json
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from mp_api.client import MPRester
from pymatgen.io.vasp import Poscar
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Sentinel object that tells the writer thread to stop
_DONE = object()


# ---------------------------------------------------------------------------
# Worker: process a batch of raw API docs → list of JSON strings
# ---------------------------------------------------------------------------

def _process_batch(batch: list) -> list[str]:
    """
    Convert a batch of MPRester document objects into serialised JSON lines.
    Runs in a worker thread.

    Parameters
    ----------
    batch : list
        A list of MPDataDoc objects returned by MPRester.

    Returns
    -------
    list[str]
        Ready-to-write JSON lines (newline NOT included).
    """
    lines = []
    for doc in batch:
        # -- POSCAR conversion (light CPU work) --------------------------
        try:
            poscar_str = Poscar(doc.structure).get_str()
        except Exception as exc:
            logger.warning(
                "Could not generate POSCAR for %s: %s — skipping.",
                doc.material_id, exc,
            )
            continue

        # -- Symmetry info (graceful fallback) ---------------------------
        crystal_system = None
        space_group    = None
        sg_number      = None
        if doc.symmetry is not None:
            crystal_system = getattr(doc.symmetry, "crystal_system", None)
            space_group    = getattr(doc.symmetry, "symbol", None)
            sg_number      = getattr(doc.symmetry, "number", None)
            if hasattr(crystal_system, "value"):
                crystal_system = crystal_system.value

        # -- Graph-level properties (regression targets) -----------------
        # formation_energy_per_atom : eV/atom (intensive).
        # total_magnetization_normalized_formula_units : μB per formula unit
        #   (intensive — the recommended magnetization target; independent of
        #   cell size, so it stays consistent between MP primitive cells and
        #   generated conventional cells).
        # total_magnetization : raw whole-cell magnetization in μB (EXTENSIVE,
        #   scales with cell size). Kept for reference only.
        #   All are stored verbatim; any may be None where MP did not compute a
        #   moment (serialised as JSON null). They are NOT used to filter out
        #   structures, so formation-energy-only training keeps full coverage.
        record = {
            "material_id":               str(doc.material_id),
            "formula_pretty":            doc.formula_pretty,
            "poscar":                    poscar_str,
            "formation_energy_per_atom": doc.formation_energy_per_atom,
            "total_magnetization_normalized_formula_units":
                                         doc.total_magnetization_normalized_formula_units,
            "total_magnetization":       doc.total_magnetization,
            "energy_above_hull":         doc.energy_above_hull,
            "n_sites":                   doc.nsites,
            "crystal_system":            crystal_system,
            "space_group":               space_group,
            "space_group_number":        sg_number,
        }
        lines.append(json.dumps(record, ensure_ascii=False))
    return lines


# ---------------------------------------------------------------------------
# Writer thread: drains write_queue → disk
# ---------------------------------------------------------------------------

def _writer_thread(write_queue: queue.Queue, output_path: Path,
                   counter: list, pbar: tqdm) -> None:
    """
    Single dedicated thread that writes serialised JSON lines to disk.
    Receives items from write_queue; stops when it sees the _DONE sentinel.

    Parameters
    ----------
    write_queue : queue.Queue
        Source of (list[str] | _DONE) items.
    output_path : Path
        Destination .jsonl file (opened in write mode).
    counter : list
        Single-element list used as a shared mutable integer for the total
        written count (avoids threading.Lock overhead for a simple counter).
    pbar : tqdm
        Progress bar to update.
    """
    with open(output_path, "w", encoding="utf-8") as fout:
        while True:
            item = write_queue.get()
            if item is _DONE:
                break
            for line in item:
                fout.write(line + "\n")
                counter[0] += 1
                pbar.update(1)
            write_queue.task_done()


# ---------------------------------------------------------------------------
# Core retrieval function (parallelised)
# ---------------------------------------------------------------------------

def fetch_mp_structures_dataset(
    api_key: str,
    output_path: str = "mp_dataset.jsonl",
    *,
    elements: Optional[list[str]] = None,
    num_elements: Optional[tuple[int, int]] = None,
    energy_above_hull_max: Optional[float] = None,
    chunk_size: int = 1000,
    batch_size: int = 200,
    num_workers: int = 16,
    max_entries: Optional[int] = None,
    retry_attempts: int = 3,
    retry_delay: float = 5.0,
) -> Path:
    """
    Fetch crystal structures and formation energies from the Materials Project
    in parallel and save as a JSON Lines (.jsonl) file.

    Architecture
    ------------
    A ThreadPoolExecutor processes docs in batches while a dedicated writer
    thread flushes results to disk. This overlaps API network latency with
    POSCAR generation and disk I/O.

    Each line in the output file is a self-contained JSON object:
    {
        "material_id":               "mp-149",
        "formula_pretty":            "Si",
        "poscar":                    "Si2 ...<POSCAR string>...",
        "formation_energy_per_atom": -0.012,
        "total_magnetization_normalized_formula_units": 0.0,
        "total_magnetization":       0.0,
        "energy_above_hull":         0.0,
        "n_sites":                   2,
        "crystal_system":            "cubic",
        "space_group":               "Fd-3m",
        "space_group_number":        227
    }

    (formation_energy_per_atom and total_magnetization_normalized_formula_units
    are the two intensive graph-level targets; total_magnetization is the raw
    per-cell value, kept for reference. Any magnetization field may be null when
    the Materials Project did not compute a magnetic moment for the material.)

    Parameters
    ----------
    api_key : str
        Your Materials Project API key (https://materialsproject.org/api).
    output_path : str
        Path to the output .jsonl file.
    elements : list[str], optional
        Restrict results to structures containing only these elements.
        E.g. ["Li", "Fe", "O"].
    num_elements : tuple(int, int), optional
        (min, max) number of distinct elements per structure.
        E.g. (1, 4) for unary through quaternary compounds.
    energy_above_hull_max : float, optional
        Maximum energy above convex hull in eV/atom.
        0.0 = only thermodynamically stable phases.
        0.1 = include near-stable phases (common for ML datasets).
    chunk_size : int
        Number of records per API page request (default 1000).
    batch_size : int
        Number of docs dispatched to each worker thread at once (default 200).
        Larger batches reduce scheduling overhead; smaller batches improve
        load balancing. 100–500 is a good range.
    num_workers : int
        Number of parallel worker threads (default 16).
        Recommended range for the AMD EPYC 7763 node: 16–32.
        The MP API will rate-limit above ~32 concurrent connections.
    max_entries : int, optional
        Cap the total number of entries downloaded (useful for testing).
    retry_attempts : int
        Number of retries on transient API errors.
    retry_delay : float
        Seconds to wait between retries.

    Returns
    -------
    Path
        Absolute path to the saved .jsonl file.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Build query filters
    # ------------------------------------------------------------------ #
    query_kwargs: dict = {
        "fields": [
            "material_id",
            "formula_pretty",
            "structure",
            "formation_energy_per_atom",
            "total_magnetization_normalized_formula_units",
            "total_magnetization",
            "energy_above_hull",
            "nsites",
            "symmetry",
        ],
        "chunk_size": chunk_size,
    }
    if elements is not None:
        query_kwargs["elements"] = elements
    if num_elements is not None:
        query_kwargs["num_elements"] = num_elements
    if energy_above_hull_max is not None:
        query_kwargs["energy_above_hull"] = (0.0, energy_above_hull_max)

    # ------------------------------------------------------------------ #
    # Connect and retrieve the result iterator
    # ------------------------------------------------------------------ #
    logger.info("Connecting to the Materials Project API …")

    mpr = MPRester(api_key)

    for attempt in range(1, retry_attempts + 1):
        try:
            results = mpr.materials.summary.search(**query_kwargs)
            break
        except Exception as exc:
            if attempt == retry_attempts:
                mpr.__exit__(None, None, None)
                raise RuntimeError(
                    f"API query failed after {retry_attempts} attempts."
                ) from exc
            logger.warning(
                "API error on attempt %d/%d: %s — retrying in %.1fs …",
                attempt, retry_attempts, exc, retry_delay,
            )
            time.sleep(retry_delay)

    logger.info(
        "Streaming with %d worker threads (batch_size=%d) → %s",
        num_workers, batch_size, output_path,
    )

    # ------------------------------------------------------------------ #
    # Shared state
    # ------------------------------------------------------------------ #
    write_queue: queue.Queue = queue.Queue(maxsize=num_workers * 4)
    total_written = [0]   # mutable list so writer thread can update it

    # ------------------------------------------------------------------ #
    # Progress bar (total unknown until fully fetched)
    # ------------------------------------------------------------------ #
    pbar = tqdm(desc="Entries written", unit="entry", dynamic_ncols=True)

    # ------------------------------------------------------------------ #
    # Start the dedicated writer thread
    # ------------------------------------------------------------------ #
    writer = threading.Thread(
        target=_writer_thread,
        args=(write_queue, output_path, total_written, pbar),
        daemon=True,
        name="jsonl-writer",
    )
    writer.start()

    # ------------------------------------------------------------------ #
    # Producer: iterate API results, accumulate batches, dispatch to pool
    # ------------------------------------------------------------------ #
    try:
        with ThreadPoolExecutor(max_workers=num_workers,
                                thread_name_prefix="mp-worker") as pool:
            futures = []
            current_batch: list = []
            total_fetched = 0

            for doc in results:
                if max_entries is not None and total_fetched >= max_entries:
                    break

                current_batch.append(doc)
                total_fetched += 1

                if len(current_batch) >= batch_size:
                    batch_to_submit = current_batch
                    current_batch = []
                    futures.append(pool.submit(_process_batch, batch_to_submit))

            # Submit the final partial batch
            if current_batch:
                futures.append(pool.submit(_process_batch, current_batch))

            # Collect results in completion order and feed the writer
            for future in as_completed(futures):
                lines = future.result()
                if lines:
                    write_queue.put(lines)

    finally:
        mpr.__exit__(None, None, None)

    # Signal the writer thread to finish and wait for it
    write_queue.put(_DONE)
    writer.join()
    pbar.close()

    logger.info("Done. %d entries saved to %s", total_written[0], output_path)
    return output_path


# ---------------------------------------------------------------------------
# Convenience loader for graph ML pipelines
# ---------------------------------------------------------------------------

def load_dataset(jsonl_path: str) -> list[dict]:
    """
    Load the saved .jsonl dataset back into a list of dicts.

    Tip: feed each entry into pymatgen's `Poscar.from_str(entry["poscar"])`
    to reconstruct the Structure object, then build your graph from it.

    Parameters
    ----------
    jsonl_path : str
        Path to the .jsonl file produced by fetch_mp_structures_dataset().

    Returns
    -------
    list[dict]
        All dataset entries as a list of dicts.
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed line %d: %s", line_no, exc)

    logger.info("Loaded %d entries from %s", len(records), path)
    return records


# ---------------------------------------------------------------------------
# Graph ML usage example (PyTorch Geometric)
# ---------------------------------------------------------------------------

GRAPH_ML_EXAMPLE = '''
# ── Graph construction sketch (PyTorch Geometric) ──────────────────────────
#
# from fetch_mp_structures import load_dataset
# from pymatgen.io.vasp import Poscar
# from torch_geometric.data import Data
# import torch
#
# dataset = load_dataset("mp_dataset.jsonl")
#
# graph_list = []
# for entry in dataset:
#     structure = Poscar.from_str(entry["poscar"]).structure
#
#     atomic_numbers = [site.specie.Z for site in structure]
#     x = torch.tensor(atomic_numbers, dtype=torch.float).unsqueeze(1)
#
#     # Periodic neighbour list (enumerates all images within the cutoff, so
#     # the graph is the same for primitive / conventional / supercell inputs):
#     neighbors = structure.get_all_neighbors(r=5.0)
#     edge_index, edge_attr = [], []
#     for i, nbrs in enumerate(neighbors):
#         for nbr in nbrs:
#             edge_index.append([i, nbr.index])
#             edge_attr.append(nbr.nn_distance)
#
#     edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
#     edge_attr  = torch.tensor(edge_attr,  dtype=torch.float).unsqueeze(1)
#
#     # Two INTENSIVE graph-level targets stacked into a single [1, 2] tensor:
#     #   column 0 → formation energy per atom (eV/atom)
#     #   column 1 → total magnetization per formula unit (μB/f.u.); use NaN
#     #              when it is null so the loss can mask materials without a
#     #              magnetic label. Both targets are cell-size-independent.
#     fe  = entry["formation_energy_per_atom"]
#     mag = entry["total_magnetization_normalized_formula_units"]
#     mag = float("nan") if mag is None else mag
#     y = torch.tensor([[fe, mag]], dtype=torch.float)        # shape [1, 2]
#
#     graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
#                  material_id=entry["material_id"])
#     graph_list.append(graph)
#
# # In the ChemGNN training loop, select the target(s) you want:
# #   y_formation = batch.y[:, 0]      # formation energy per atom only
# #   y_magnet    = batch.y[:, 1]      # magnetization per formula unit only
# #   y_both      = batch.y            # both properties (multi-task)
'''


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download Materials Project structures + formation energies "
                    "+ total magnetization (parallel)."
    )
    parser.add_argument("--api-key",     required=True,       help="MP API key")
    parser.add_argument("--output",      default="./dataset_mp/mp_dataset.jsonl")
    parser.add_argument("--elements",    nargs="+", default=None, metavar="EL")
    parser.add_argument("--min-nelems",  type=int,  default=None)
    parser.add_argument("--max-nelems",  type=int,  default=None)
    parser.add_argument("--hull-max",    type=float, default=None,
                        help="Max energy above hull in eV/atom (e.g. 0.1)")
    parser.add_argument("--max-entries", type=int,  default=None)
    parser.add_argument("--chunk-size",  type=int,  default=1000)
    parser.add_argument("--batch-size",  type=int,  default=200,
                        help="Docs per worker batch (default 200)")
    parser.add_argument("--num-workers", type=int,  default=16,
                        help="Parallel worker threads (default 16, max ~32)")
    args = parser.parse_args()

    num_elements = None
    if args.min_nelems is not None or args.max_nelems is not None:
        lo = args.min_nelems or 1
        hi = args.max_nelems or 10
        num_elements = (lo, hi)

    fetch_mp_structures_dataset(
        api_key=args.api_key,
        output_path=args.output,
        elements=args.elements,
        num_elements=num_elements,
        energy_above_hull_max=args.hull_max,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_entries=args.max_entries,
    )


