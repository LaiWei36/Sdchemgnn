"""
build_crystal_graph_dataset.py
======================
Builds a PyG InMemoryDataset from the Materials Project .jsonl file, IDENTICAL to
dataset_mp.py in every respect EXCEPT the node features `x`, which are enriched for
magnetism prediction.

Why: the baseline dataset uses a plain one-hot atomic-type node feature, which tells
the model *which* element but nothing about *why* it is (or isn't) magnetic. This
builder replaces/augments that with physically meaningful, element-derived features
that are computable from the atomic symbol alone (so they are perfectly consistent
between training and inference on generated structures, and never fail):

  - CGCNN atom embeddings: the standard 92-dim binary vector per element from the
    published CGCNN `atom_init.json` (binned group, period, electronegativity,
    covalent radius, valence electrons, ionization energy, electron affinity, block,
    atomic volume). Keyed by atomic number.
  - Magnetism-extras (3 dims): free-atom, Hund's-rule descriptors that CGCNN's
    generic vector lacks but that most directly drive atomic moments:
        [ unpaired_electrons / 8,  valence_d_electrons / 10,  valence_f_electrons / 14 ]
    e.g. Fe -> [0.50, 0.60, 0.00], Gd -> [1.00, 0.10, 0.50], O -> [0.25, 0, 0].
    Computed via pymatgen's electron configuration (exception-correct: Cr, Cu, Gd...).

node_feature_mode selects the composition (for clean A/B experiments):
  'onehot' : one-hot over the dataset's elements            (== baseline dataset_mp)
  'cgcnn'  : CGCNN(92) (+) magnetism-extras(3)   = 95        [default; no memorization]
  'both'   : one-hot (+) CGCNN(92) (+) magnetism-extras(3)   [maximal info]

EVERYTHING ELSE is preserved so this is a drop-in for the trainer: the periodic
neighbour-list edges, the lazy 50-D Gaussian edge expansion + its meta/transform
wiring, the two intensive targets and their named attributes (y,
formation_energy_per_atom, total_magnetization_normalized_formula_units,
total_magnetization, material_id, formula), and the parallel two-pass build. The
trainer reads in_channels from the data, so it adapts automatically to the new width.

To USE it, point the trainer/config at this module and root:
    sdchemgnn_config.py:  dataset_root = './dataset_mp_magnetic'
    trainer import:      from build_crystal_graph_dataset import MPDataset
NOTE: the tail-audit's composition-decoding assumes one-hot x and won't decode
compositions on 'cgcnn'/'both' datasets — it uses material_id anyway.

Requirements:
    pip install torch torch_geometric pymatgen ase tqdm scikit-learn
    plus atom_init.json in the same folder as this script (bundled).

Command line (build to the magnetic root; keep cutoff matched to the baseline = 7.0):
    python build_crystal_graph_dataset.py \
        --jsonl ./dataset_mp/mp_dataset.jsonl \
        --root  ./dataset_mp_magnetic \
        --cutoff 7.0 \
        --node-features cgcnn \
        --num-workers 64 \
        --batch-size 25
"""

import os
import os.path as osp
import json
import io
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from tqdm import tqdm

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import Compose

from ase.io.vasp import read_vasp
from ase.neighborlist import neighbor_list
from sklearn import preprocessing


# atom_init.json is bundled alongside this script by default.
_DEFAULT_ATOM_INIT = osp.join(osp.dirname(osp.abspath(__file__)), 'atom_init.json')


# ============================================================
# Edge features (unchanged from the original ChemGNN dataset builder, dataset_mp.py,
# which is not part of this release)
# ============================================================

def gaussian_expansion(distances, dmin=0.0, dmax=7.0, num=50):
    """Expand scalar interatomic distances into a Gaussian radial basis.
    distances: (E,) -> (E, num)."""
    centers = np.linspace(dmin, dmax, num)
    sigma = (dmax - dmin) / num
    d = np.asarray(distances).reshape(-1, 1)
    return np.exp(-((d - centers) ** 2) / (2.0 * sigma ** 2))


class _GaussianEdgeExpansion:
    """Load-time transform: expand a stored raw bond distance per edge (edge_attr of
    shape [E, 1]) into the [E, num] Gaussian radial basis the model consumes. Keeps
    the stored InMemoryDataset ~50x smaller on edge features (identical to dataset_mp)."""

    def __init__(self, dmax, num=50, dmin=0.0):
        self.dmin, self.dmax, self.num = dmin, dmax, num

    def __call__(self, data):
        d = data.edge_attr.view(-1).cpu().numpy()
        data.edge_attr = torch.as_tensor(
            gaussian_expansion(d, dmin=self.dmin, dmax=self.dmax, num=self.num),
            dtype=torch.float,
        )
        return data


# ============================================================
# One-hot vocabulary (unchanged from dataset_mp.py; used by 'onehot'/'both')
# ============================================================

def gen_one_hot_dict(symbols: list) -> dict:
    """One-hot dict covering ALL unique elements in `symbols` (duplicates allowed),
    so the encoding dimension is consistent across the dataset. Identical to
    dataset_mp.gen_one_hot_dict."""
    import copy
    le = preprocessing.LabelEncoder()
    symbols_num_list = le.fit_transform(symbols)
    symbols_tensor = torch.unique(torch.as_tensor(symbols_num_list))
    numS = symbols_tensor.size(0)
    code = [0.0] * numS
    atom_type_dict = {}
    for pos, atom in enumerate(symbols_tensor):
        a_code = copy.deepcopy(code)
        a_code[pos] = 1.0
        atom_type_dict[le.inverse_transform([atom]).item()] = a_code
    return atom_type_dict


def collect_all_elements(jsonl_path: str) -> list:
    """Single pass over the .jsonl to collect every element symbol across ALL
    structures (with duplicates). Identical to dataset_mp.collect_all_elements."""
    all_symbols = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            try:
                atoms = read_vasp(io.StringIO(record['poscar']))
                all_symbols.extend(atoms.get_chemical_symbols())
            except Exception:
                continue
    return all_symbols


# ============================================================
# NEW: CGCNN embeddings + magnetism-extras node features
# ============================================================

def load_atom_init(path: str) -> dict:
    """Load the CGCNN atom_init.json: {atomic_number_str: [92 floats]} (binary)."""
    with open(path, 'r') as f:
        raw = json.load(f)
    return {str(k): [float(x) for x in v] for k, v in raw.items()}


_ORBITAL_CAP = {'s': 1, 'p': 3, 'd': 5, 'f': 7}   # orbitals per subshell


def magnetism_extras(symbol: str) -> list:
    """Free-atom, Hund's-rule magnetism descriptors from the element's electron
    configuration (via pymatgen, which is exception-correct for Cr/Cu/Gd/...):
        [ unpaired_electrons / 8, valence_d_electrons / 10, valence_f_electrons / 14 ]
    Returns [0, 0, 0] (with a warning) if the configuration cannot be read, so the
    build degrades gracefully rather than crashing on an unexpected element."""
    try:
        from pymatgen.core.periodic_table import Element
        fes = Element(symbol).full_electronic_structure   # [(n, l, occ), ...]
        unpaired = 0.0
        d_val = f_val = 0.0
        d_n = f_n = -1
        for entry in fes:
            n, l, occ = entry[0], entry[1], entry[2]
            k = _ORBITAL_CAP.get(l, 0)
            if k:
                unpaired += occ if occ <= k else 2 * k - occ   # Hund unpaired in subshell
            if l == 'd' and n > d_n:
                d_n, d_val = n, occ                            # valence (highest-n) d
            if l == 'f' and n > f_n:
                f_n, f_val = n, occ                            # valence (highest-n) f
        return [unpaired / 8.0, d_val / 10.0, f_val / 14.0]
    except Exception as exc:
        print(f'  [warn] magnetism-extras failed for {symbol}: {exc}; zero-filling')
        return [0.0, 0.0, 0.0]


def build_node_feature_dict(symbols: list, mode: str, atom_init_path: str) -> dict:
    """Build {element_symbol: node_feature_vector (list[float])} for every unique
    element in `symbols`, according to `mode`:
        'onehot' -> one-hot                                     (len = #elements)
        'cgcnn'  -> CGCNN(92) + magnetism-extras(3)             (len = 95)
        'both'   -> one-hot + CGCNN(92) + magnetism-extras(3)   (len = #elements + 95)
    Runs once in the main process (Pass 1); workers only look up the finished dict."""
    if mode not in ('onehot', 'cgcnn', 'both'):
        raise ValueError(f"node_feature_mode must be 'onehot'|'cgcnn'|'both', got {mode!r}")

    onehot = gen_one_hot_dict(symbols)
    unique = sorted(onehot.keys())

    cgcnn = {}
    mag = {}
    if mode in ('cgcnn', 'both'):
        from ase.data import atomic_numbers
        atom_init = load_atom_init(atom_init_path)
        for sym in unique:
            z = atomic_numbers.get(sym)
            vec = atom_init.get(str(z))
            if vec is None:
                print(f'  [warn] {sym} (Z={z}) not in atom_init.json; zero-filling CGCNN block')
                vec = [0.0] * 92
            cgcnn[sym] = vec
            mag[sym] = magnetism_extras(sym)

    feat = {}
    for sym in unique:
        v = []
        if mode in ('onehot', 'both'):
            v = v + list(onehot[sym])
        if mode in ('cgcnn', 'both'):
            v = v + cgcnn[sym] + mag[sym]
        feat[sym] = v
    return feat


# ============================================================
# Per-entry graph builder (main process; unchanged except x comes from the
# combined node_feature_dict, and the dict param is renamed for clarity)
# ============================================================

def build_graph_from_entry(record: dict, node_feature_dict: dict, cutoff: float):
    """Convert one worker-produced raw dict into a PyG Data object (tensors built in
    the MAIN process to avoid the shared-memory fd explosion). Preserves every named
    attribute the trainer/audit rely on. `node_feature_dict` is accepted for signature
    parity but not used here (x is already in the raw dict)."""
    if record is None or record.get('skipped', False):
        return None
    try:
        x          = torch.tensor(record['x'],          dtype=torch.float)
        edge_index = torch.tensor(record['edge_index'], dtype=torch.long)
        edge_attr  = torch.tensor(record['edge_attr'],  dtype=torch.float)   # [E,1] raw dist
        y          = torch.tensor(record['y'],          dtype=torch.float)   # [1,2]

        formation_energy_per_atom = torch.tensor(
            [[record['formation_energy_per_atom']]], dtype=torch.float)
        total_magnetization_normalized_formula_units = torch.tensor(
            [[record['total_magnetization_normalized_formula_units']]], dtype=torch.float)
        total_magnetization = torch.tensor(
            [[record['total_magnetization']]], dtype=torch.float)

        return Data(
            x                                            = x,
            edge_index                                   = edge_index,
            edge_attr                                    = edge_attr,
            y                                            = y,
            formation_energy_per_atom                    = formation_energy_per_atom,
            total_magnetization_normalized_formula_units = total_magnetization_normalized_formula_units,
            total_magnetization                          = total_magnetization,
            material_id                                  = record['material_id'],
            formula                                      = record['formula'],
        )
    except Exception as exc:
        print(f"  [skip tensor conversion] {record.get('material_id', '?')} — {exc}")
        return None


# ============================================================
# Worker (builds x via node_feature_dict lookup; otherwise identical to dataset_mp)
# ============================================================

def _build_raw_entry(record: dict, node_feature_dict: dict, cutoff: float):
    """Build a graph as plain-Python lists/floats (NO tensors) in a worker process.
    Steps mirror dataset_mp._build_raw_entry; only the node-feature lookup differs
    (a richer per-element vector instead of the one-hot)."""
    required = ('poscar', 'formation_energy_per_atom', 'material_id', 'formula_pretty')
    for key in required:
        if record.get(key) is None:
            return {'skipped': True, 'reason': f'missing_field:{key}',
                    'material_id': record.get('material_id', '?')}
    try:
        compound = read_vasp(io.StringIO(record['poscar']))

        formation_energy = float(record['formation_energy_per_atom'])
        mag_fu_raw   = record.get('total_magnetization_normalized_formula_units', None)
        mag_fu       = float('nan') if mag_fu_raw is None else float(mag_fu_raw)
        mag_cell_raw = record.get('total_magnetization', None)
        mag_cell     = float('nan') if mag_cell_raw is None else float(mag_cell_raw)
        y = [[formation_energy, mag_fu]]

        compound.pbc = True
        src, dst, edge_distances = neighbor_list('ijd', compound, cutoff)
        if len(src) == 0:
            return {'skipped': True, 'reason': f'no_edges_at_cutoff_{cutoff}A',
                    'material_id': record['material_id']}

        edge_index = [src.tolist(), dst.tolist()]
        atomic_symbols = compound.get_chemical_symbols()

        # Node features: enriched per-element vector (CGCNN (+) magnetism-extras, and/or
        # one-hot), looked up from the dict built once in the main process.
        x = [node_feature_dict[atom] for atom in atomic_symbols]

        edge_attr = np.asarray(edge_distances, dtype=np.float32).reshape(-1, 1)  # [E,1]

        return {
            'skipped': False,
            'x': x,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'y': y,
            'formation_energy_per_atom': formation_energy,
            'total_magnetization_normalized_formula_units': mag_fu,
            'total_magnetization': mag_cell,
            'material_id': record['material_id'],
            'formula': record['formula_pretty'],
        }
    except Exception as exc:
        return {'skipped': True, 'reason': f'exception:{type(exc).__name__}:{exc}',
                'material_id': record.get('material_id', '?')}


def _process_batch(args: tuple) -> list:
    """Process a batch of raw .jsonl records into plain-Python dicts (worker process).
    Returns plain Python (no tensors) to avoid the 'Too many open files' fd limit."""
    batch, node_feature_dict, cutoff = args
    return [_build_raw_entry(record, node_feature_dict, cutoff) for record in batch]


# ============================================================
# InMemoryDataset subclass (adds node_feature_mode / atom_init_path)
# ============================================================

class MPDataset(InMemoryDataset):
    """PyG InMemoryDataset for MP crystal structures with magnetism-enriched node
    features. Class name kept as `MPDataset` so the trainer change is just the import
    module and dataset_root. Layout: <root>/processed/data.pt (auto-generated).

    Parameters (new vs dataset_mp)
    ------------------------------
    node_feature_mode : 'onehot' | 'cgcnn' | 'both'   (default 'cgcnn')
        See module docstring. Baked into data.pt at build time; rebuild to change.
    atom_init_path : str | None
        Path to the CGCNN atom_init.json (default: bundled alongside this file).
    """

    def __init__(self,
                 root: str,
                 jsonl_path: str,
                 cutoff: float = 7.0,
                 num_workers: int = 32,
                 batch_size: int = 500,
                 edge_gaussians: int = 50,
                 node_feature_mode: str = 'cgcnn',
                 atom_init_path: str = None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):

        self.jsonl_path        = jsonl_path
        self.cutoff            = cutoff
        self.num_workers       = num_workers
        self.batch_size        = batch_size
        self.edge_gaussians    = edge_gaussians
        self.node_feature_mode = node_feature_mode
        self.atom_init_path    = atom_init_path or _DEFAULT_ATOM_INIT

        super().__init__(root, None, pre_transform, pre_filter)

        path   = osp.join(self.my_processed_dir, 'data.pt')
        loaded = torch.load(path, weights_only=False)
        self.data, self.slices = loaded[0], loaded[1]
        meta = loaded[2] if len(loaded) > 2 else {}

        # Expand edges exactly as they were built (cutoff/basis persisted in data.pt).
        self.cutoff            = meta.get('cutoff',            self.cutoff)
        self.edge_gaussians    = meta.get('edge_gaussians',    self.edge_gaussians)
        self.node_feature_mode = meta.get('node_feature_mode', self.node_feature_mode)
        _expand = _GaussianEdgeExpansion(dmax=self.cutoff, num=self.edge_gaussians)
        self.transform = _expand if transform is None else Compose([_expand, transform])

    @property
    def raw_file_names(self):
        return []

    @property
    def my_processed_dir(self):
        return osp.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        """Two-pass build: Pass 1 (single process) builds the combined node-feature
        dict for the whole dataset; Pass 2 (parallel) builds graphs. Mirrors
        dataset_mp.process(); only Pass 1's featurization differs."""
        if not os.path.exists(self.my_processed_dir):
            os.makedirs(self.my_processed_dir)

        # ── Pass 1: elements + node-feature dict ─────────────────────────
        print(f'Pass 1: scanning elements and building node features '
              f'(mode={self.node_feature_mode}) …')
        all_symbols = collect_all_elements(self.jsonl_path)
        if not all_symbols:
            raise RuntimeError(f'No valid entries found in {self.jsonl_path}.')
        node_feature_dict = build_node_feature_dict(
            all_symbols, self.node_feature_mode, self.atom_init_path)
        node_dim = len(next(iter(node_feature_dict.values())))
        print(f'  {len(node_feature_dict)} unique elements; node feature dim = {node_dim} '
              f'(mode={self.node_feature_mode})')
        print(f'  elements: {sorted(node_feature_dict.keys())}')

        # ── Pass 2: build graphs in parallel ─────────────────────────────
        print(f'Pass 2: building graphs '
              f'(num_workers={self.num_workers}, batch_size={self.batch_size}) …')
        print('  Loading records from .jsonl …')
        all_records = []
        with open(self.jsonl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    all_records.append(json.loads(line))
        total = len(all_records)
        print(f'  {total} records loaded.')

        batches = [all_records[i:i + self.batch_size]
                   for i in range(0, total, self.batch_size)]
        print(f'  {len(batches)} batches of up to {self.batch_size} records each.')

        worker_args = [(batch, node_feature_dict, self.cutoff) for batch in batches]

        data_list, dataset_len, missed_count = [], 0, 0
        skip_reasons: dict = {}
        pbar = tqdm(total=total, desc='Processing MP dataset', unit='entry')

        with ProcessPoolExecutor(max_workers=self.num_workers) as pool:
            futures = {pool.submit(_process_batch, args): args for args in worker_args}
            for future in as_completed(futures):
                for raw in future.result():
                    if raw is None or raw.get('skipped', False):
                        reason = raw.get('reason', 'unknown') if raw else 'none_returned'
                        reason_key = reason.split(':')[0] if ':' in reason else reason
                        skip_reasons[reason_key] = skip_reasons.get(reason_key, 0) + 1
                        missed_count += 1
                        pbar.update(1)
                        continue
                    data = build_graph_from_entry(raw, node_feature_dict, self.cutoff)
                    if self.pre_filter is not None and data is not None:
                        if not self.pre_filter(data):
                            data = None
                    if self.pre_transform is not None and data is not None:
                        data = self.pre_transform(data)
                    if data is not None:
                        data_list.append(data)
                        dataset_len += 1
                    else:
                        missed_count += 1
                    pbar.update(1)
        pbar.close()

        print(f'dataset_len  = {dataset_len}')
        print(f'missed_count = {missed_count}')
        if skip_reasons:
            print('Skip reason summary:')
            for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
                print(f'  {count:>6}  {reason}')

        # ── Collate and save (meta records cutoff, basis, feature mode + dim) ──
        data, slices = self.collate(data_list)
        meta = {'cutoff': self.cutoff, 'edge_gaussians': self.edge_gaussians,
                'node_feature_mode': self.node_feature_mode, 'node_feature_dim': node_dim}
        torch.save((data, slices, meta), osp.join(self.my_processed_dir, 'data.pt'))
        print(f'Saved → {osp.join(self.my_processed_dir, "data.pt")}'
              f'  (cutoff={self.cutoff}, edge_gaussians={self.edge_gaussians}, '
              f'node_feature_mode={self.node_feature_mode}, node_feature_dim={node_dim})')


# ============================================================
# Convenience entry point
# ============================================================

def build_mp_dataset(jsonl_path: str,
                     root: str = 'dataset/mp_magnetic',
                     cutoff: float = 7.0,
                     num_workers: int = 32,
                     batch_size: int = 500,
                     node_feature_mode: str = 'cgcnn',
                     atom_init_path: str = None) -> MPDataset:
    print('Building MPDataset (magnetic node features)')
    print(f'  jsonl_path        : {jsonl_path}')
    print(f'  root              : {root}')
    print(f'  cutoff            : {cutoff} Å')
    print(f'  node_feature_mode : {node_feature_mode}')
    print(f'  atom_init_path    : {atom_init_path or _DEFAULT_ATOM_INIT}')
    print(f'  num_workers       : {num_workers}')
    print(f'  batch_size        : {batch_size}')

    dataset = MPDataset(
        root=root, jsonl_path=jsonl_path, cutoff=cutoff,
        num_workers=num_workers, batch_size=batch_size,
        node_feature_mode=node_feature_mode, atom_init_path=atom_init_path,
    )

    print('MPDataset ready.')
    print(f'  Number of graphs : {len(dataset)}')
    print(f'  Node feature dim : {dataset[0].x.shape[1]}')
    print(f'  Edge feature dim : {dataset[0].edge_attr.shape[1]}')
    print(f'  Target (y) shape : {tuple(dataset[0].y.shape)}  '
          f'[col 0 = formation energy/atom, col 1 = magnetization per f.u.]')
    print(f'  Example entry    : {dataset[0]}')
    return dataset


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Build a PyG InMemoryDataset from an MP .jsonl file with '
                    'magnetism-enriched (CGCNN + magnetism-extras) node features.')
    parser.add_argument('--jsonl',        default='mp_dataset.jsonl')
    parser.add_argument('--root',         default='dataset/mp_magnetic')
    parser.add_argument('--cutoff',       type=float, default=7.0,
                        help='Interatomic distance cutoff in Å (default 7.0; MUST match '
                             'the baseline dataset_mp build for a fair comparison)')
    parser.add_argument('--node-features', choices=['onehot', 'cgcnn', 'both'],
                        default='cgcnn',
                        help="Node feature composition (default 'cgcnn' = CGCNN + "
                             "magnetism-extras, no one-hot)")
    parser.add_argument('--atom-init', default=None,
                        help='Path to CGCNN atom_init.json (default: bundled alongside '
                             'this script)')
    parser.add_argument('--num-workers', type=int, default=32,
                        help='Worker processes (default 32; use 64-128 on the EPYC node)')
    parser.add_argument('--batch-size',  type=int, default=500,
                        help='Records per worker batch (default 500)')
    args = parser.parse_args()

    build_mp_dataset(
        jsonl_path=args.jsonl,
        root=args.root,
        cutoff=args.cutoff,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        node_feature_mode=args.node_features,
        atom_init_path=args.atom_init,
    )
