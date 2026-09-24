#!/usr/bin/env python
"""MagSieve scoring pass: runs the trained SD-ChemGNN model, its weights fixed,
on every proposed spin configuration, and dumps the router features.

EVALUATION ONLY. Loads a finished run's best weights and scores the requested
splits several ways. No optimizer is built, no gradient is taken, no weight is
written, no existing run directory is modified. It READS `best_model.pt` and
WRITES new files under --out.

It imports the trainer as a MODULE to reuse its tested helpers, which is why
the trainer's `if __name__ == '__main__'` guard matters: importing defines the
helpers and runs nothing.

Parts, selected with --parts:

  0a  SHUFFLED-DECORATION SELF-CONSISTENCY. The trained model reproduces
      whatever arrangement it is handed (self-consistency near 1 under the
      label-derived decoration). This part dumps the per-graph
      self-consistency under the true, a shuffled and an all-parallel
      decoration side by side: if it is just as high under a WRONG
      arrangement, a self-consistency filter discards nothing.

  0b  UNDECORATED-MODE ORDERING-HEAD DUMP. The auxiliary ordering-diagnostic
      head's output in the undecorated mode, on the population that would
      actually need routing.

  0c  SPREAD ACROSS PROPOSED CONFIGURATIONS. For every test structure,
      enumerate up to K candidate arrangements with
      propose_spin_configurations and score EVERY one of them. This decides
      the selector's design: whether the spread of predictions across
      candidate arrangements separates compensated structures from real
      magnets. Scored offline afterwards by diagnostics_score.py.

  0d  ROUTER FEATURES (the part MagSieve uses). The same K-arrangement
      scoring for the train, val and test splits, dumped per split as the
      pooled embedding (phase0d_<split>_h.npy), the per-structure table
      (phase0d_<split>_meta.csv: truth, undecorated prediction, ordering
      annotation, descriptors) and the (structure, arrangement) candidate
      table (phase0d_<split>_cand.txt). ordering_router.py fits on them and
      screen_shortlist.py scores the shortlist from them.

The proposed-configuration injection deliberately reuses the SHIPPED
decoration path: it overwrites data.seed_sign / data.seed_elig on a per-batch
CLONE and then calls the trainer's own decorate_batch in 'oracle' mode. No new
decoration mode is added, so what the model sees is byte-for-byte what a
label-derived decoration would produce.

WHY THE SETUP IS REPLICATED HERE RATHER THAN IMPORTED
-----------------------------------------------------
The trainer builds the split inside main(), which also trains. There is no
reusable entry point. So build_test_split() below performs the SAME sequence
in the SAME order with the trainer's OWN functions, and prints the same census
lines the trainer prints. Those lines are the check: diff them against the
training log. If any of them differs, the split or the labels have drifted and
every number from this job is void.

USAGE (from a working directory holding the dataset and the label files)
    python score_configurations_with_sdchemgnn.py --run ../03_sdchemgnn/trained_model --out ./magsieve_features --parts 0d --splits train,val,test --k 8 --cache-workers 0
    python score_configurations_with_sdchemgnn.py --run ../03_sdchemgnn/trained_model --out ./magsieve_diagnostics --parts 0a,0b,0c --k 8
"""

from __future__ import annotations

# ── repository path bootstrap (added for the GitHub release) ──────────────
# The stage folders import each other's modules; make all of them importable
# whichever folder this file is run from.
import os as _os
import sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _sub in ('02_data_curation', '03_sdchemgnn', '04_magsieve'):
    _p = _os.path.join(_REPO_ROOT, _sub)
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
# ─────────────────────────────────────────────────────────────────────────
import argparse
import csv
import gc
import json
import os
import os.path as osp
import sys
import time

import numpy as np
import torch

import propose_spin_configurations as PROP

# Importing the trainer defines its helpers and runs nothing (main() is behind
# the __main__ guard). Everything the trainer itself imported — build_cache,
# build_magmom_labels, build_seed_table, collect_modes, _dec_eval, cfg-derived
# device — is reachable through this one handle.
#
# The import is guarded so the pure batch-geometry helpers below stay unit
# testable in an environment that has torch but not the training package (the
# offline suite exercises _node_ptr and _split_edges that way). main() refuses
# to start if it failed, so nothing can silently run half-imported.
try:
    import train_sdchemgnn as T
    import spin_decoration as S
    from sdchemgnn_config import config_magnetic as cfg
    from sdchemgnn_model import make_model_bs, SEED_CHANNELS
    _IMPORT_ERROR = None
except Exception as _exc:                                   # noqa: BLE001
    T = S = cfg = make_model_bs = None
    SEED_CHANNELS = 3
    _IMPORT_ERROR = _exc


def log(msg):
    print(msg, flush=True)


# ══════════════════════════════════════════════════════════════════════════════
#  preflight — the deployed config must describe the run being diagnosed
# ══════════════════════════════════════════════════════════════════════════════
def check_run_config(run_dir, strict=True):
    """Compare the deployed config against what the run was ACTUALLY trained
    with, read from its own `*_hyper_parameters.json`.

    `--run` supplies the weights; the config builds the model. They are
    independent and neither can be inferred from the other, so pointing --run
    at an SD-ChemGNN run while the config still says bs_arm='xu_baseline'
    constructs the wrong network. That already aborts at load_state_dict, but
    five minutes into the job, after the dataset and every label table have
    been built. This catches it in the first second instead.

    Returns the recorded hyper-parameter dict (or None if the run predates it).
    """
    import glob
    hits = sorted(glob.glob(osp.join(run_dir, '*hyper_parameters.json')))
    if not hits:
        log(f'  NOTE: no *hyper_parameters.json in {run_dir} — cannot verify '
            f'that the deployed config describes this run. The census check '
            f'later is then the only guard; read it carefully.')
        return None
    with open(hits[0]) as fh:
        hp = json.load(fh)

    want_split = int(getattr(cfg, 'split_seed', None) or cfg.seed)
    checks = [('bs_arm', str(getattr(cfg, 'bs_arm', 'seed_input')).lower(),
               lambda v: str(v).lower()),
              ('split_seed', want_split, int),
              ('init_seed', getattr(cfg, 'init_seed', None),
               lambda v: None if v is None else int(v)),
              ('seed_epsilon', float(cfg.seed_epsilon), float),
              ('seed_fmag_scale', float(cfg.seed_fmag_scale), float),
              ('target_cap', float(cfg.target_cap), float)]
    bad = []
    for key, have, cast in checks:
        if key not in hp:
            continue
        was = cast(hp[key])
        flag = '   <-- MISMATCH' if was != have else ''
        log(f'    {key:<18} run {str(was):<12} config {str(have):<12}{flag}')
        if was != have:
            bad.append((key, was, have))
    if bad and strict:
        lines = '\n'.join(f'    {k}: the run used {w!r}, the deployed config '
                          f'says {h!r}' for k, w, h in bad)
        raise SystemExit(
            f'\nFATAL: the deployed config does not describe this run.\n{lines}\n\n'
            f'--run supplies the WEIGHTS; sdchemgnn_config.py builds the '
            f'MODEL and\nthe split. Nothing can reconcile them automatically.\n\n'
            f'If bs_arm is the mismatch, set it to the variant the run was '
            f'trained with\n(seed_input for the released weights) and rerun. '
            f'Editing that line is a\nconfig change, so this job will not make '
            f'it.')
    return hp


# ══════════════════════════════════════════════════════════════════════════════
#  setup — the trainer's sequence, its own functions, in its own order
# ══════════════════════════════════════════════════════════════════════════════
def build_test_split(verbose=True, expect_n_elig=None, return_all=False):
    """Rebuild the split exactly as the trainer does, and return
    (test_ds, man, magmom_labels, seeds, modes_by_id, base_in, edge_dim, g).

    With `return_all=True` the first element becomes the dict
    {'train': train_ds, 'val': val_ds, 'test': test_ds} instead — needed by
    part 0d, which has to dump router features for every split so the router
    can be FITTED off the evaluation set. Nothing else about the sequence
    changes, so the test split is identical either way.

    Order is load-bearing and mirrors train_sdchemgnn.main() line for line:
        dataset -> split draw (split_seed) -> manifest relabel + drops ->
        ordering modes -> magmom node labels -> seed channels
    Each attach is followed by _fresh_item_view(), which is the trainer's item-cache
    fix; skipping one silently serves stale items.
    """
    v = log if verbose else (lambda *_a, **_k: None)

    # ── dataset + split draw ────────────────────────────────────────────────
    dataset = T.MPDataset(root=cfg.dataset_root, jsonl_path='')
    split_seed = int(getattr(cfg, 'split_seed', None) or cfg.seed)
    torch.manual_seed(split_seed)                 # pins membership — never move
    perm = torch.randperm(len(dataset))
    dataset = dataset[perm]
    n_total = len(dataset)
    n_train = int(n_total * cfg.train_p)
    n_val = int(n_total * cfg.val_p)
    train_ds = dataset[:n_train]
    val_ds = dataset[n_train:n_train + n_val]
    test_ds = dataset[n_train + n_val:]
    v(f'Dataset sizes  —  train: {len(train_ds)}, val: {len(val_ds)}, '
      f'test: {len(test_ds)}')

    _init_seed = getattr(cfg, 'init_seed', None)
    if _init_seed is not None:
        v(f'SEED-2 MODE: split_seed={split_seed} (membership pinned), '
          f'init_seed={int(_init_seed)} (fresh init/dropout/shuffle streams)')
    g = torch.Generator()
    g.manual_seed(int(_init_seed) if _init_seed is not None else cfg.seed)

    if cfg.scope_max is not None:
        raise SystemExit('scope_max is set in the config. This pass is a '
                         'full-split diagnostic and a scope cut would silently '
                         'change the population — refusing to run.')

    # ── curated labels: manifest AFTER the split ────────────────────────────
    if not getattr(cfg, 'label_manifest', None):
        raise SystemExit('label_manifest is unset — the run being diagnosed '
                         'was curated, so the config has drifted.')
    man, manifest_md5 = T.load_manifest(cfg.label_manifest)
    v(f'LABEL MANIFEST: {cfg.label_manifest}')
    v(f'    manifest rows: {len(man)}; md5 {manifest_md5}')
    ids, n_rel = T.relabel_master(dataset, man)
    drop_mids = {m for m, (a, _) in man.items() if a == 'drop'}

    def _drop(view, orig_idx, tag):
        keep = [ids[j] not in drop_mids for j in orig_idx.tolist()]
        n_drop = len(keep) - sum(keep)
        v(f'    manifest[{tag}]: {n_drop} dropped, {sum(keep)} kept')
        if n_drop == 0:
            return view
        return view[torch.tensor([i for i, k in enumerate(keep) if k])]

    train_ds = _drop(train_ds, perm[:n_train], 'train')
    val_ds = _drop(val_ds, perm[n_train:n_train + n_val], 'val')
    test_ds = _drop(test_ds, perm[n_train + n_val:], 'test')
    v(f'    {n_rel} labels rewritten in the master store')

    # ── ordering-mode aux labels ────────────────────────────────────────────
    if not getattr(cfg, 'ordering_sidetable', None):
        raise SystemExit('ordering_sidetable is unset — the site-decomposition model requires it.')
    from ordering_mode_labels import build_modes
    modes_by_id, ordering_meta = build_modes(
        cfg.ordering_sidetable, man,
        collapse_path=getattr(cfg, 'collapse_suspects', None),
        cap=float(cfg.target_cap),
        near_zero=float(getattr(cfg, 'mode_near_zero', 0.5)))
    v(f'ORDERING MODES: {cfg.ordering_sidetable}')
    v(f'    side-table md5 {ordering_meta["sidetable_md5"]}; census '
      f'{ordering_meta["census"]}')
    T.attach_ordering_mode_labels(dataset, modes_by_id)
    T._fresh_item_view(dataset)

    # ── per-site node labels ────────────────────────────────────────────────
    if not getattr(cfg, 'magmom_sidetable', None):
        raise SystemExit('magmom_sidetable is unset — the site-decomposition model has no meaning '
                         'without node labels.')
    _sl = getattr(dataset, 'slices', None) or getattr(dataset, '_slices', None)
    _store = getattr(dataset, '_data', None)
    if _store is None:
        _store = dataset.data
    _xb = _sl['x'].tolist()
    _mids_master = [str(x[0] if isinstance(x, (list, tuple)) else x)
                    for x in _store.material_id]
    node_counts = {m: int(_xb[j + 1] - _xb[j]) for j, m in enumerate(_mids_master)}
    magmom_labels, magmom_meta = T.build_magmom_labels(
        cfg.magmom_sidetable, cfg.ordering_sidetable, man, node_counts,
        collapse_path=getattr(cfg, 'collapse_suspects', None),
        cap=float(cfg.target_cap),
        shift_tol=float(cfg.nodesup_shift_tol),
        zero_tol=float(cfg.no_magmom_zero_tol),
        resid_clip=float(cfg.resid_target_clip))
    v(f'MAGMOM NODE LABELS: {cfg.magmom_sidetable}')
    v(f'    magmom md5 {magmom_meta["magmom_md5"]}')
    v(f'    census {magmom_meta["census"]}')
    v(f'    supervised: {magmom_meta["n_sup"]}/{magmom_meta["n_rows"]} graphs')
    T.attach_magmom_labels(dataset, magmom_labels)
    T._fresh_item_view(dataset)

    # ── seed channels ───────────────────────────────────────────────────────
    seeds, seed_meta = T.build_seed_table(
        cfg.magmom_sidetable, cfg.ordering_sidetable, man, magmom_labels,
        node_counts, eps=float(cfg.seed_epsilon),
        gauge_min_y=float(cfg.seed_gauge_min_y),
        gauge_min_summ=float(cfg.seed_gauge_min_summ),
        cap=float(cfg.target_cap))
    _n_trip = T.verify_seed_vs_sidetable(seeds, cfg.magmom_sidetable,
                                         float(cfg.seed_epsilon))
    _ss = seed_meta['site_signs']
    v(f'SEED DECORATION: eps={cfg.seed_epsilon} uB')
    v(f"    eligible {seed_meta['n_elig']}/{seed_meta['n_rows']} graphs "
      f"({seed_meta['n_nm_seed']} NM seeds, {seed_meta['n_mixed_sign']} "
      f"mixed-sign); {seed_meta['n_sites_decorated']} decorated sites "
      f"[+{_ss['plus']} -{_ss['minus']} 0:{_ss['zero']}]")
    v(f"    gauge conflicts seed-masked: {seed_meta['n_gauge_conflicts']}; "
      f"census {seed_meta['census']}")
    v(f'    tripwire: {_n_trip} rows re-derived from the raw side-table — '
      f'consistent')
    # The single strongest check that the side-tables and the manifest have not
    # drifted since the run: the trained run recorded how many graphs its seed
    # build found eligible, and that number is a function of all of them.
    if expect_n_elig is not None and int(seed_meta['n_elig']) != int(expect_n_elig):
        raise SystemExit(
            f'\nFATAL: the seed build found {seed_meta["n_elig"]} eligible graphs; '
            f'the run recorded {expect_n_elig}.\nA side-table, the manifest or '
            f'seed_epsilon has changed since this model was trained, so the '
            f'weights\nno longer match the data this job would score them on. '
            f'Every number would be void.\nReconcile the inputs before '
            f'rerunning — see the README.')
    if expect_n_elig is not None:
        v(f'    seed-eligible count matches the trained run ({expect_n_elig})')
    T.attach_seed_channels(dataset, seeds)
    T._fresh_item_view(dataset)

    base_in = int(test_ds[0].x.size(-1))
    edge_dim = int(test_ds[0].edge_attr.shape[1])
    v(f'in_channels: {base_in} base + {SEED_CHANNELS} seed = '
      f'{base_in + SEED_CHANNELS}, edge_dim: {edge_dim}')
    first = ({'train': train_ds, 'val': val_ds, 'test': test_ds}
             if return_all else test_ds)
    return first, man, magmom_labels, seeds, modes_by_id, base_in, edge_dim, g


def _ids_of(batch):
    mid = batch.material_id
    if not isinstance(mid, (list, tuple)):
        mid = [mid]
    return [str(m[0] if isinstance(m, (list, tuple)) else m) for m in mid]


def _valid_mask(batch):
    """The NaN mask every trainer dump applies, so rows line up with the
    existing regression_*.txt / sitedecomp_*.txt artifacts."""
    return ~torch.isnan(T.get_magnetization(batch))


def _node_ptr(batch):
    """Node offsets [G+1] for a collated batch, on CPU as plain ints.

    PyG collation lays each graph's nodes down contiguously, so a graph is a
    SLICE, not a gather. Using the slice bounds keeps the per-graph work in
    python integers; the obvious `(batch_vec == gi).nonzero()` costs one GPU
    kernel and one sync per graph, which at 1024 graphs a batch and 16 batches
    a pass is ~16k syncs for work that is arithmetic.
    """
    ptr = getattr(batch, 'ptr', None)
    if ptr is not None:
        return [int(v) for v in ptr.cpu().tolist()]
    counts = torch.bincount(batch.batch.cpu())
    return [0] + torch.cumsum(counts, 0).tolist()



# ══════════════════════════════════════════════════════════════════════════════
#  side-table readers
# ══════════════════════════════════════════════════════════════════════════════
def species_by_id(path):
    """Per-site species from the magmom side-table — the same file and key the
    training-time seed builder reads, so 'magnetic' means the same thing here
    as everywhere else in SD-ChemGNN."""
    out = {}
    try:
        with open(str(path)) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                mid, sp = rec.get('material_id'), rec.get('species')
                if mid and sp:
                    out[str(mid)] = list(sp)
    except OSError as exc:
        log(f'    magmom side-table unreadable ({exc})')
    return out


def ordering_raw_by_id(path):
    """The RAW Materials Project `ordering` string per material_id.

    The router routes to one of NM / FM / AFM / FiM, which map one-to-one onto
    the proposer's candidate families. The trainer's `ordering_mode_labels.build_modes`
    collapses those four into three OUTCOME families and is the label the
    auxiliary ordering-diagnostic head was trained on — deliberately not what is wanted here. Reading
    the side-table directly avoids depending on `ordering_mode_labels` and keeps the
    4-class label available even though that module is not importable off the
    cluster. Defensive about the column name; reports what it found.
    """
    out = {}
    try:
        with open(str(path), newline='') as fh:
            rd = csv.DictReader(fh)
            cols = [c.strip() for c in (rd.fieldnames or [])]
            idc = next((c for c in cols if c.lower() in
                        ('material_id', 'mid', 'id')), None)
            ordc = next((c for c in cols if c.lower() in
                         ('ordering', 'magnetic_ordering', 'order')), None)
            if idc is None or ordc is None:
                log(f'    ordering side-table columns {cols} — no usable '
                    f'(material_id, ordering) pair; 4-class labels UNAVAILABLE')
                return out
            for r in rd:
                mid = (r.get(idc) or '').strip()
                val = (r.get(ordc) or '').strip()
                if mid and val:
                    out[mid] = val
            log(f'    ordering side-table: column {ordc!r}, {len(out):,} rows')
    except OSError as exc:
        log(f'    ordering side-table unreadable ({exc}); 4-class labels '
            f'UNAVAILABLE')
    return out


class _CaptureH:
    """Grab the pooled graph embedding `h` from the forward pass already being
    run, via a pre-hook on `mode_head` — which forward_b() calls as
    `self.mode_head(h)`. A hook rather than a second forward pass, and a hook
    rather than reaching into private methods, so it works unchanged for both
    model variants."""

    def __init__(self, model):
        self.h = None
        self._handle = model.mode_head.register_forward_pre_hook(self._grab)

    def _grab(self, _module, args):
        self.h = args[0].detach()

    def close(self):
        self._handle.remove()


@torch.no_grad()
def collect_undecorated(model, loader, dcfg):
    """ONE undecorated pass returning everything the router reads that does not
    require running the candidate arrangements:

        stats [N, 9]  pred, sum_abs_m, sum_m, resid_fu, z_fu,
                      p_net0, p_partial, p_aligned, mode_label
        H     [N, D]  the pooled graph embedding

    Undecorated deliberately: this is what a deployed router would see.
    """
    model.eval()
    cap = _CaptureH(model)
    try:
        S_, H_ = [], []
        for batch in loader:
            data = batch.to(T.device, non_blocking=True)
            data = S.decorate_batch(data, 'undecorated', dcfg)
            v = _valid_mask(data)
            out = model.forward_b(data)
            m_hat = out['m_hat'].reshape(-1)
            n = int(v.numel())
            sum_abs = torch.zeros(n, device=T.device, dtype=m_hat.dtype)
            sum_abs = sum_abs.index_add(0, data.batch, m_hat.abs())
            probs = torch.softmax(out['mode_logits'], dim=1)
            if cap.h is None or cap.h.size(0) != n:
                raise RuntimeError('pooled embedding not captured — the model '
                                   'did not call mode_head as expected')
            cols = torch.stack([out['y'].reshape(-1), sum_abs,
                                out['m_sum'].reshape(-1),
                                out['r_hat'].reshape(-1),
                                T.get_z_fu(data).reshape(-1).to(m_hat.dtype),
                                probs[:, 0], probs[:, 1], probs[:, 2],
                                T.get_ordering_mode(data).to(m_hat.dtype)],
                               dim=1)
            S_.append(cols[v].cpu())
            H_.append(cap.h[v].cpu())
    finally:
        cap.close()
    return torch.cat(S_), torch.cat(H_)


def enumerate_candidates(cache, sp_by_id, k_max):
    """Propose up to k_max arrangements for every valid graph in a cache."""
    cand, n_no_sp, n_mismatch = {}, 0, 0
    for batch in cache:                       # read-only sweep, no clone
        ptr, edges = _split_edges(batch)
        keep = _valid_mask(batch).cpu().numpy()
        for gi, mid in enumerate(_ids_of(batch)):
            if not keep[gi]:
                continue
            sp = sp_by_id.get(mid)
            if sp is None:
                n_no_sp += 1
                cand[mid] = []
            elif len(sp) != ptr[gi + 1] - ptr[gi]:
                n_mismatch += 1
                cand[mid] = []
            else:
                cand[mid] = PROP.propose(sp, edges[gi], k_max=k_max)
    return cand, n_no_sp, n_mismatch


def score_candidate_slots(model, loader, dcfg, cand_by_id, order, tiny,
                          out_path, t0):
    """Score every candidate slot and write the (structure, arrangement) table.
    Returns (n_slots, mean_candidates)."""
    n_c = np.asarray([len(cand_by_id.get(m, [])) for m in order] or [0])
    kmax = int(n_c.max())
    if kmax == 0:
        log('    no candidates to score.')
        return 0, 0.0
    with open(out_path, 'w') as fp:
        fp.write('# row material_id cand_idx cand_name exact net_formal pred '
                 'sum_abs_m sum_m resid_fu z_fu sc_frac\n')
        for k in range(kmax):
            signs = {m: (cand_by_id[m][k]['sign']
                         if k < len(cand_by_id.get(m, [])) else None)
                     for m in order}
            got = sum(1 for s in signs.values() if s is not None)
            res = score_with_signs(model, loader, dcfg, signs, tiny).numpy()
            if res.shape[0] != len(order):
                sys.exit(f'FATAL: slot {k} produced {res.shape[0]} rows for '
                         f'{len(order)} ids — row alignment broken.')
            for i, mid in enumerate(order):
                cl = cand_by_id.get(mid, [])
                if k >= len(cl):
                    continue
                c = cl[k]
                fp.write('%d %s %d %s %d %.4f %.6f %.6f %.6f %.6f %.1f %.6f\n'
                         % (i, mid, k, c['name'], int(c['exact']),
                            c['net_formal'], res[i, 0], res[i, 1], res[i, 2],
                            res[i, 3], res[i, 4], res[i, 5]))
            log(f'    slot {k}: {got:6d} structures scored   '
                f'[{time.time() - t0:6.1f}s]')
    return kmax, float(n_c.mean())


# ══════════════════════════════════════════════════════════════════════════════
#  scoring
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def score_with_signs(model, loader, dcfg, signs_by_id, tiny):
    """Score the split once, injecting a supplied sign vector per graph.

    `signs_by_id` maps material_id -> int8 array over that graph's sites, or
    None / missing to leave the graph unseeded (three zero channels — the
    undecorated fallback). Returns a [N, 6] float tensor

        pred, sum_abs_m, sum_m, resid_pred_fu, z_fu, sc_frac

    in loader order under the NaN mask: the row order of the trainer's dumps.
    `pred` is forward_b()['y'] — the shipped reconstruction, not a
    reimplementation of it.
    """
    model.eval()
    rows = []
    for batch in loader:                 # CachedLoader yields a fresh clone
        ids = _ids_of(batch)
        ptr = _node_ptr(batch)
        # Assemble both vectors on the CPU in one pass and transfer once,
        # instead of a per-graph scatter with a GPU sync each.
        sign_np = np.zeros(ptr[-1], dtype=np.float32)
        elig_np = np.zeros(len(ids), dtype=np.int8)
        for gi, mid in enumerate(ids):
            s = signs_by_id.get(mid)
            if s is None:
                continue
            lo, hi = ptr[gi], ptr[gi + 1]
            if hi - lo != len(s):
                raise RuntimeError(
                    f'{mid}: proposer returned {len(s)} signs for a graph with '
                    f'{hi - lo} sites — species/graph order mismatch')
            sign_np[lo:hi] = np.asarray(s, dtype=np.float32)
            elig_np[gi] = 1
        data = batch.to(T.device, non_blocking=True)
        batch_vec = data.batch
        sign = torch.from_numpy(sign_np).to(device=T.device, dtype=data.x.dtype)
        elig = torch.from_numpy(elig_np).to(T.device)
        # Inject, then hand over to the SHIPPED decoration path unchanged.
        # seed_fmag is deliberately NOT overwritten: it is the spin-only formal
        # magnitude looked up per species (build_seed_table line ~178), so it
        # is available at inference and carries no label information. elig = 0
        # leaves a graph with three zero channels, i.e. exactly the undecorated
        # fallback — which is what a structure with no candidate should get.
        data.seed_sign = sign
        data.seed_elig = elig
        data = S.decorate_batch(data, 'oracle', dcfg)

        v = _valid_mask(data)
        out = model.forward_b(data)
        m_hat = out['m_hat'].reshape(-1)
        pred = out['y'].reshape(-1)
        r_hat = out['r_hat'].reshape(-1)
        z_fu = T.get_z_fu(data).reshape(-1).to(pred.dtype)
        sum_abs = torch.zeros(len(ids), device=T.device,
                              dtype=m_hat.dtype).index_add_(0, batch_vec,
                                                            m_hat.abs())
        sum_m = out['m_sum'].reshape(-1)
        sc, _ = S.self_consistency(m_hat, data, tiny=tiny)
        rows.append(torch.stack([pred[v], sum_abs[v], sum_m[v], r_hat[v],
                                 z_fu[v], sc[v].to(pred.dtype)], dim=1).cpu())
    return torch.cat(rows, dim=0)


@torch.no_grad()
def collect_sc(model, loader, dec, tiny):
    """Per-graph self-consistency under an arbitrary SHIPPED decoration mode."""
    model.eval()
    out = []
    for batch in loader:                 # CachedLoader yields a fresh clone
        data = batch.to(T.device, non_blocking=True)
        data = T._dec_eval(data, dec)
        v = _valid_mask(data)
        m_hat = model.forward_b(data)['m_hat'].reshape(-1)
        sc, _ = S.self_consistency(m_hat, data, tiny=tiny)
        out.append(sc[v].cpu())
    return torch.cat(out, dim=0)


def _split_edges(batch):
    """Per-graph edge lists for a whole batch, each reindexed to that graph's
    local site numbering. One CPU pass over the batch rather than a per-graph
    mask, and the same contiguity fact `_node_ptr` relies on.
    """
    ei = batch.edge_index.cpu().numpy()
    ptr = _node_ptr(batch)
    if ei.shape[1] == 0:
        return ptr, [None] * (len(ptr) - 1)
    owner = np.searchsorted(np.asarray(ptr[1:]), ei[0], side='right')
    order = np.argsort(owner, kind='stable')
    ei, owner = ei[:, order], owner[order]
    bounds = np.searchsorted(owner, np.arange(len(ptr)))
    out = []
    for gi in range(len(ptr) - 1):
        lo, hi = bounds[gi], bounds[gi + 1]
        out.append(None if hi <= lo else ei[:, lo:hi] - ptr[gi])
    return ptr, out


# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True,
                    help='finished run directory (reads best_model.pt)')
    ap.add_argument('--out', required=True, help='output directory (created)')
    ap.add_argument('--parts', default='0a,0b,0c')
    ap.add_argument('--k', type=int, default=PROP.DEFAULT_K,
                    help='candidate arrangements per structure')
    ap.add_argument('--limit-batches', type=int, default=0,
                    help='smoke test: use only the first N test batches')
    ap.add_argument('--splits', default='train,val,test',
                    help='part 0d: which splits to dump features for')
    ap.add_argument('--cache-batch', type=int, default=0,
                    help='0d: collation batch size; 0 = mirror the trainer '
                         '(train at cfg.batch_size, val/test at the eval batch)')
    ap.add_argument('--cache-workers', type=int, default=-1,
                    help='0d: collation workers; -1 = cfg.cache_workers, '
                         '0 = the fork/shm-free fallback if memory is tight')
    ap.add_argument('--max-rows', type=int, default=0,
                    help='0d: subsample a split to at most N rows (train only, '
                         'in practice) — 0 = all')
    ap.add_argument('--check-only', action='store_true',
                    help='run the preflight and exit; no dataset, no GPU')
    a = ap.parse_args()
    if _IMPORT_ERROR is not None:
        raise SystemExit(
            f'FATAL: the training package did not import '
            f'({type(_IMPORT_ERROR).__name__}: {_IMPORT_ERROR}).\n'
            f'This script must run from the directory holding the deployed '
            f'trainer, config, model and decoration modules — see the README.')
    parts = {p.strip() for p in a.parts.split(',') if p.strip()}
    unknown = parts - {'0a', '0b', '0c', '0d'}
    if unknown:
        sys.exit(f'unknown parts {sorted(unknown)}')
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    log(f'MagSieve scoring pass — parts {sorted(parts)}  K={a.k}')
    log(f'  run  {a.run}')
    log(f'  out  {a.out}')
    log(f'  device {T.device} | torch {torch.__version__}')
    log(f'  variant  {getattr(cfg, "bs_arm", "?")}   eps {cfg.seed_epsilon}')

    ckpt = osp.join(a.run, 'best_model.pt')
    if not osp.exists(ckpt):
        sys.exit(f'FATAL: {ckpt} not found — nothing to diagnose.')

    # ── preflight: does the deployed config describe THIS run? ───────────────
    log('\n[preflight] deployed config vs the run\'s own hyper_parameters.json')
    hp = check_run_config(a.run)
    if a.check_only:
        log('\n[preflight] OK — --check-only, stopping here.')
        return
    log('  OK')

    # ── setup ────────────────────────────────────────────────────────────────
    log('\n[setup] rebuilding the split with the trainer\'s own functions.')
    log('        DIFF THE CENSUS LINES BELOW AGAINST THE TRAINING .out —')
    log('        any difference voids every number this job produces.')
    need_all = '0d' in parts
    (_first, man, magmom_labels, seeds, modes_by_id,
     base_in, edge_dim, g) = build_test_split(
        expect_n_elig=(hp or {}).get('n_seed_eligible'), return_all=need_all)
    splits = _first if need_all else {'test': _first}
    test_ds = splits['test']

    dcfg = {'base_channels': int(base_in),
            'p_undecorate': float(cfg.seed_p_undecorate),
            'p_site_mask': float(cfg.seed_p_site_mask),
            'p_promote': float(cfg.seed_p_promote),
            'fmag_scale': float(cfg.seed_fmag_scale),
            'coflip': False}          # eval: the gauge pair is training-only
    tiny = float(getattr(cfg, 'sc_tiny', 0.05))

    log('\n[setup] collating the test cache')
    test_cache = T.build_cache(test_ds, cfg.batch_size * cfg.eval_batch_factor,
                               cfg.cache_workers, g, tag='test')
    # the trainer's own cache tripwires, on the tensors this job will consume
    T.verify_manifest_in_cache(test_cache, man, 'test')
    T.verify_magmom_in_cache(test_cache, magmom_labels, 'test')
    S.verify_seed_in_cache(test_cache, seeds, 'test')
    if a.limit_batches:
        test_cache = test_cache[:a.limit_batches]
        log(f'  SMOKE TEST: first {len(test_cache)} batches only — the '
            f'numbers are NOT the split numbers.')
    # Every model pass goes through CachedLoader, which yields a CLONE. Without
    # it, `.to(device)` and the decoration's `data.x = cat(...)` would mutate
    # the cache in place and the second pass would abort on a 98-channel x.
    test_loader = T.CachedLoader(test_cache, shuffle=False, generator=g)

    # ── row index + truth: the join keys every offline scorer needs ──────────
    # The trainer dumps neither, which is why earlier artifacts had to be
    # aligned by hand against test_ids.csv (and were off by one row).
    order, truth = [], []
    for batch in test_cache:                     # read-only sweep, no clone
        v = _valid_mask(batch)
        order += [m for m, keep in zip(_ids_of(batch), v.tolist()) if keep]
        truth.append(T.cap_target(T.get_magnetization(batch)[v]).cpu())
    truth = torch.cat(truth).numpy()
    if len(set(order)) != len(order):
        sys.exit(f'FATAL: {len(order) - len(set(order))} duplicate material_ids '
                 f'in the test split — the id keying below would collide.')
    with open(osp.join(a.out, 'phase0_test_ids.csv'), 'w') as fh:
        fh.write('row,material_id,truth_capped\n')
        for i, (m, t) in enumerate(zip(order, truth)):
            fh.write(f'{i},{m},{t:.6f}\n')
    log(f'  rows scored: {len(order)}  (|truth| median '
        f'{float(np.median(np.abs(truth))):.3f}, '
        f'{int((np.abs(truth) >= 1.0).sum())} at |M| >= 1)  '
        f'-> phase0_test_ids.csv')

    # ── model ────────────────────────────────────────────────────────────────
    model = make_model_bs(base_in + SEED_CHANNELS, edge_dim, cfg).to(T.device)
    state = torch.load(ckpt, map_location=T.device)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        sys.exit(f'FATAL: {len(missing)} weights missing from the checkpoint '
                 f'(e.g. {list(missing)[:4]}). The config says '
                 f'bs_arm={getattr(cfg, "bs_arm", "?")!r} — wrong variant for this '
                 f'run directory?')
    if unexpected:
        log(f'  note: {len(unexpected)} unexpected keys ignored '
            f'(e.g. {list(unexpected)[:4]})')
    model.eval()
    log(f'  loaded {ckpt}')

    manifest = {'run': a.run, 'k': a.k, 'parts': sorted(parts),
                'bs_arm': str(getattr(cfg, 'bs_arm', '?')),
                'seed_epsilon': float(cfg.seed_epsilon),
                'sc_tiny': tiny, 'n_rows': len(order),
                'limit_batches': int(a.limit_batches)}

    # ── 0a ───────────────────────────────────────────────────────────────────
    if '0a' in parts:
        log('\n[0a] self-consistency under true / shuffled / FM-guess seeds')
        gsh = torch.Generator()
        gsh.manual_seed(1729)          # the trainer's probe seed, reproducible
        cols, names = [], []
        for tag, dec in (('oracle', {'mode': 'oracle', 'dcfg': dcfg}),
                         ('shuffled', {'mode': 'shuffled', 'dcfg': dcfg,
                                       'gen': gsh}),
                         ('fm_formal', {'mode': 'fm_formal', 'dcfg': dcfg})):
            sc = collect_sc(model, test_loader, dec, tiny)
            fin = sc[~torch.isnan(sc)]
            log(f'    {tag:<10} committal on {int(fin.numel()):6d} graphs   '
                f'mean sc_frac {float(fin.mean()):.4f}   '
                f'fully self-consistent '
                f'{100.0 * float((fin >= 0.9999).float().mean()):.2f}%')
            cols.append(sc.numpy())
            names.append(tag)
        np.savetxt(osp.join(a.out, 'phase0a_selfconsistency.txt'),
                   np.column_stack(cols), header=' '.join(names), fmt='%.6f')
        log('    -> phase0a_selfconsistency.txt')
        log('    READ: if shuffled sits close to oracle, selector v0\'s '
            'self-consistency filter discards nothing.')

    # ── 0b ───────────────────────────────────────────────────────────────────
    if '0b' in parts:
        log('\n[0b] mode head in UNDECORATED mode (never dumped before)')
        dec = {'mode': 'undecorated', 'dcfg': dcfg}
        P, MU, K = T.collect_modes(model, test_loader, dec)
        lab = K.reshape(-1).numpy()
        pred = P.numpy().argmax(1)
        m = lab >= 0
        if m.sum():
            acc = float((pred[m] == lab[m]).mean())
            maj = float(np.bincount(lab[m].astype(int), minlength=3).max() / m.sum())
            log(f'    labelled {int(m.sum())} graphs   accuracy {acc:.4f}   '
                f'majority-class baseline {maj:.4f}   '
                f'{"BEATS" if acc > maj else "LOSES TO"} a constant predictor')
            manifest['mode_undec_acc'] = acc
            manifest['mode_undec_majority'] = maj
        np.savetxt(osp.join(a.out, 'phase0b_modes_undec.txt'),
                   np.column_stack([P.numpy(), MU.numpy(),
                                    lab.reshape(-1, 1)]),
                   header='p_net0 p_partial p_aligned mu_net0 mu_partial '
                          'mu_aligned mode_label', fmt='%.6f')
        log('    -> phase0b_modes_undec.txt')

    # ── 0c ───────────────────────────────────────────────────────────────────
    if '0c' in parts:
        log(f'\n[0c] aggregation feasibility — up to {a.k} arrangements '
            f'per structure')
        sp_by_id = species_by_id(cfg.magmom_sidetable)
        log(f'    side-table species rows: {len(sp_by_id):,}')
        t_enum = time.time()
        cand_by_id, n_no_sp, n_mismatch = enumerate_candidates(
            test_cache, sp_by_id, a.k)
        log(f'    enumeration took {time.time() - t_enum:.1f}s on CPU')
        n_c = np.asarray([len(cand_by_id.get(m, [])) for m in order] or [0])
        log(f'    enumerated: mean {n_c.mean():.2f} candidates/structure, '
            f'max {int(n_c.max())}, zero-candidate {int((n_c == 0).sum())}'
            f'  ({n_no_sp} without species, {n_mismatch} site-count mismatches)')
        kmax, meanc = score_candidate_slots(
            model, test_loader, dcfg, cand_by_id, order, tiny,
            osp.join(a.out, 'phase0c_proposed.txt'), t0)
        if kmax:
            manifest['n_candidate_slots'] = kmax
            manifest['mean_candidates'] = meanc
            log('    -> phase0c_proposed.txt   (score with diagnostics_score.py)')

    # ── 0d ───────────────────────────────────────────────────────────────────
    # Router features for EVERY split. The router has to be fitted off the
    # evaluation set, so train and val are not optional extras here — without
    # them any test number would be fitted on the rows it is scored on.
    if '0d' in parts:
        want = [x.strip() for x in a.splits.split(',') if x.strip()]
        log(f'\n[0d] router features for splits {want}')
        sp_by_id = species_by_id(cfg.magmom_sidetable)
        ord_raw = ordering_raw_by_id(cfg.ordering_sidetable)
        manifest['ordering_raw_rows'] = len(ord_raw)
        if ord_raw:
            from collections import Counter
            log(f'    ordering census: '
                f'{dict(Counter(ord_raw.values()).most_common(8))}')
        else:
            log('    WARNING: no 4-class ordering labels. The router falls '
                'back to the 3-class outcome family, which is the label the '
                'auxiliary head already used — read the README before '
                'treating a null result as informative.')
        # MIRROR THE TRAINER'S OWN BATCH SIZES PER SPLIT. Its log reads
        # "cached train=241 / val=16 / test=16", i.e. train at cfg.batch_size
        # (512) and val/test at the eval batch (1024) — and that combination is
        # known to fit the 240G request while ALSO holding all three caches at
        # once. Building train at the eval batch instead doubles the in-flight
        # batch in each of cache_workers collation workers, and that is what
        # OOM-killed a first attempt at part 0d. Overridable, but the default should
        # be whatever the trainer already proved fits.
        for name in want:
            ds = splits.get(name)
            if ds is None:
                log(f'    unknown split {name!r} — skipping')
                continue
            log(f'\n  [{name}] {len(ds)} graphs')
            tsp = time.time()
            if a.max_rows and len(ds) > a.max_rows:
                # A subsample is legitimate for TRAIN — the router is a 4-class
                # model over ~128k labelled rows and does not need all of them.
                # It is NOT legitimate for val or test, so say which was cut.
                idx = torch.randperm(len(ds), generator=torch.Generator()
                                     .manual_seed(23647))[:a.max_rows]
                ds = ds[idx]
                log(f'    SUBSAMPLED to {len(ds)} rows (--max-rows)'
                    + ('  <-- NOTE: this is an evaluation split'
                       if name in ('val', 'test') else ''))
            cb = a.cache_batch or (cfg.batch_size if name == 'train'
                                   else cfg.batch_size * cfg.eval_batch_factor)
            cw = cfg.cache_workers if a.cache_workers < 0 else a.cache_workers
            log(f'    cache: batch {cb}, workers {cw}')
            cache = T.build_cache(ds, cb, cw, g, tag=name)
            loader = T.CachedLoader(cache, shuffle=False, generator=g)
            ids, truth = [], []
            for b in cache:
                vv = _valid_mask(b)
                ids += [m for m, kp in zip(_ids_of(b), vv.tolist()) if kp]
                truth.append(T.cap_target(T.get_magnetization(b)[vv]).cpu())
            truth = torch.cat(truth).numpy()
            if len(set(ids)) != len(ids):
                sys.exit(f'FATAL: duplicate material_ids in {name}.')
            stats, H = collect_undecorated(model, loader, dcfg)
            if stats.shape[0] != len(ids):
                sys.exit(f'FATAL: {name} stats has {stats.shape[0]} rows for '
                         f'{len(ids)} ids — row alignment broken.')
            np.save(osp.join(a.out, f'phase0d_{name}_h.npy'),
                    H.numpy().astype(np.float32))
            sarr = stats.numpy()
            with open(osp.join(a.out, f'phase0d_{name}_meta.csv'), 'w') as fh:
                fh.write('row,material_id,truth_capped,pred_undec,sum_abs_m,'
                         'sum_m,resid_fu,z_fu,p_net0,p_partial,p_aligned,'
                         'mode_label,ordering_raw\n')
                for r, (mid, ty) in enumerate(zip(ids, truth)):
                    fh.write('%d,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.1f,%.6f,%.6f,'
                             '%.6f,%d,%s\n'
                             % (r, mid, ty, sarr[r, 0], sarr[r, 1], sarr[r, 2],
                                sarr[r, 3], sarr[r, 4], sarr[r, 5], sarr[r, 6],
                                sarr[r, 7], int(sarr[r, 8]),
                                ord_raw.get(mid, '')))
            n_lab = sum(1 for m in ids if ord_raw.get(m))
            log(f'    embedding {tuple(H.shape)}; 4-class label on '
                f'{n_lab}/{len(ids)} rows')
            cb, nns, nmm = enumerate_candidates(cache, sp_by_id, a.k)
            nc = np.asarray([len(cb.get(m, [])) for m in ids] or [0])
            log(f'    candidates: mean {nc.mean():.2f}, zero {int((nc==0).sum())}'
                f' ({nns} without species, {nmm} site mismatches)')
            score_candidate_slots(model, loader, dcfg, cb, ids, tiny,
                                  osp.join(a.out, f'phase0d_{name}_cand.txt'),
                                  t0)
            log(f'    [{name}] done in {time.time() - tsp:.1f}s')
            del cache, loader
            gc.collect()
        log('\n    -> phase0d_*_meta.csv / _h.npy / _cand.txt   '
            '(fit the router with ordering_router.py)')

    with open(osp.join(a.out, 'phase0_manifest.json'), 'w') as fh:
        json.dump(manifest, fh, indent=1)
    log(f'\nDONE in {time.time() - t0:.1f}s -> {a.out}')


if __name__ == '__main__':
    main()
