"""Spin decoration for SD-ChemGNN (spin-decorated node features).

Design decisions implemented here:
    channels  three channels per node appended to x (95 -> 98):
            seed sign      s_i in {-1, 0, +1}
            formal magn.   mu_i / seed_fmag_scale  (spin-only, per-species;
                           the SAME table the formula-ceiling clamp of the
                           shortlist scorer uses, so training and deployment
                           share one semantics; 0 for non-magnetic species)
            seed-present   f_i in {0, 1}
        Label magnitudes NEVER enter the input (copy-shortcut guard): the
        magnitude channel is the formal per-species value, not the label.
    dead zone  eps = 0.1 uB for sign assignment (|m| <= eps -> sign 0).
    gauge by augmentation: per decorated graph, random global co-flip of
        the seed signs AND every signed supervision target (site labels,
        residual target, the per-f.u. total the umbrella/aux/mode-value losses
        read). Node loss is gauge-PINNED on decorated graphs; min-over-s stays
        for undecorated graphs only (combined_node_loss).
    corruption schedule (train mode): p_undecorate whole-graph dropout of
        the decoration; p_site_mask per-site channel zeroing; p_promote
        promotion of zero-sign magnetic-species sites to a random +/-1 at
        formal magnitude (closes the "this site converged magnetic" leak).
    eligibility: rows without usable per-site labels (node_sup == 0) are NEVER
        decorated; 'zero' rows (no_magmom, both totals ~ 0) get the NM seed
        (all signs 0, flags 1) — the proposer's NM hypothesis.

The sign-convention corner: rows whose stored site-moment gauge CONTRADICTS
the curated total (|y_cur| > seed_gauge_min_y and sum(m) * y_cur < 0 with
|sum(m)| > seed_gauge_min_summ) are seed-masked (elig = 0, why =
'gauge_conflict') and their ids are collected in meta['gauge_conflict_ids']
for review. They stay node-supervised exactly as before (min-over-s tolerates
the gauge); only the SEED channel refuses them.

Seed source discipline: signs derive from the SAME labels dict that
attach_magmom_labels writes (site_moment_labels.build_magmom_labels), so
seed <-> label consistency holds BY CONSTRUCTION; verify_seed_vs_sidetable()
then re-derives a sample straight from the raw jsonl as an independent
tripwire (two build paths must agree, else HALT).

Accessors at the bottom are THE read path — decoration, losses, verifiers and
dumps all go through them (never a parallel attribute read).

Standalone dry run (no torch dataset, no cluster — node counts are proxied by
the side-table's own n_sites, which the real attach verifies against the
graphs):

    python spin_decoration.py --sidetable ./magmom_sidetable.jsonl --ordering ./ordering_sidetable.csv --manifest ./label_manifest.csv --collapse ./collapse_suspects.csv --conflict-csv seed_gauge_conflicts.csv
"""

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
import json
import math
import os

import torch

from site_moment_labels import build_magmom_labels

# ── Formal spin-only magnitudes (uB per atom) ──────────────────────────────────
# THE per-species table: identical to the formula-ceiling table of the
# shortlist scorer and to what the inference proposer emits. High-spin d/f
# counting; absent species (closed-shell / non-magnetic) are 0. When the
# earlier screening module `screen_stack` (not part of this release) is
# present, a consistency assert keeps the two copies from drifting; the
# embedded copy makes this module importable without it.
FORMAL_MAGNITUDE = {
    'Ti': 2, 'V': 3, 'Cr': 4, 'Mn': 5, 'Fe': 5, 'Co': 4, 'Ni': 2, 'Cu': 1,
    'Nb': 3, 'Mo': 3, 'Tc': 3, 'Ru': 4, 'Rh': 2, 'Pd': 1, 'W': 2, 'Re': 3,
    'Os': 2, 'Ir': 1, 'Ce': 1, 'Pr': 2, 'Nd': 3, 'Pm': 4, 'Sm': 5, 'Eu': 7,
    'Gd': 7, 'Tb': 6, 'Dy': 5, 'Ho': 4, 'Er': 3, 'Tm': 2, 'Yb': 1,
    'U': 3, 'Np': 4, 'Pu': 5,
}
try:                                           # drift guard vs the ceiling table
    from screen_stack import MAG_MAX as _CEILING_TABLE
    if _CEILING_TABLE != FORMAL_MAGNITUDE:
        raise RuntimeError(
            'FORMAL_MAGNITUDE differs from screen_stack.MAG_MAX — the seed '
            'magnitude channel and the physics ceiling must share ONE table. '
            'Sync the copies before training.')
except ImportError:                            # screen_stack not present (normal)
    pass

FMAG_MAX = float(max(FORMAL_MAGNITUDE.values()))          # 7.0 (Eu/Gd)

SEED_CHANNELS = 3                              # sign, formal magnitude, flag


def _f(s):
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _load_raw_fu(ordering_csv):
    """material_id -> raw per-f.u. value (None if absent) — same column the
    label builder reads for the curated-label reconstruction."""
    out = {}
    with open(ordering_csv, newline='') as fh:
        for row in csv.DictReader(fh):
            out[row['material_id']] = _f(row.get('jsonl_raw_fu'))
    return out


def _curated_y(mid, manifest, raw_fu, cap):
    """The curated per-f.u. label, EXACTLY as build_magmom_labels derives it
    (manifest override where present, else capped raw)."""
    act, new = ('', None)
    m_row = manifest.get(mid)
    if m_row is not None:
        act, new = m_row[0], m_row[1]
    raw = raw_fu.get(mid)
    if act not in ('', 'drop') and new is not None:
        return new
    if raw is not None:
        return max(-cap, min(cap, raw))
    return None


# ── build ──────────────────────────────────────────────────────────────────────
def build_seed_table(magmom_jsonl, ordering_csv, manifest, labels, node_counts,
                     eps=0.1, gauge_min_y=1.0, gauge_min_summ=0.1, cap=30.0):
    """Derive per-material seed records from the ALREADY-BUILT node-label dict.

    labels: the dict build_magmom_labels returned (the supervision authority —
            seed signs come from labels[mid]['m'], never re-parsed, so the seed
            convention IS the label convention on every decorated row).
    node_counts: {material_id: graph node count} (dataset master, or the
            side-table's n_sites in the standalone dry run).

    Returns (seeds, meta):
        seeds[mid] = {'elig': 0/1, 'why': str,
                      'sign': list[int]   (present iff elig),
                      'fmag': list[float] (formal uB per site; zeros if the
                              side-table has no species row for the id)}
        meta: censuses + the gauge-conflict id list + knobs (fingerprint food).
    """
    raw_fu = _load_raw_fu(ordering_csv)

    # one pass over the side-table for species (formal magnitudes)
    species_by_id = {}
    with open(magmom_jsonl) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            mid = str(rec.get('material_id', ''))
            if mid not in node_counts:
                continue
            sp = rec.get('species')
            if isinstance(sp, list):
                species_by_id[mid] = sp

    seeds = {}
    census = {}
    n_elig = n_nm = n_mixed = 0
    n_sites_dec = n_plus = n_minus = n_zero_sites = 0
    n_promotable = n_eps_band = 0
    gauge_conflicts = []

    def _mask(mid, why, fmag):
        seeds[mid] = {'elig': 0, 'why': why, 'fmag': fmag}
        census[why] = census.get(why, 0) + 1

    for mid, n_graph in node_counts.items():
        sp = species_by_id.get(mid)
        if sp is not None and len(sp) == n_graph:
            fmag = [float(FORMAL_MAGNITUDE.get(s, 0)) for s in sp]
        else:
            fmag = [0.0] * n_graph
            if sp is not None:                       # species there, wrong len
                _mask(mid, 'species_mismatch', fmag)
                continue
        rec = labels.get(mid)
        if rec is None or not rec.get('sup'):
            why = (rec or {}).get('why', 'missing_label_record')
            _mask(mid, f'unsup_{why}', fmag)
            continue
        if rec.get('why') == 'zero':                 # NM seed (all-zero signs)
            seeds[mid] = {'elig': 1, 'why': 'nm_seed',
                          'sign': [0] * n_graph, 'fmag': fmag}
            census['nm_seed'] = census.get('nm_seed', 0) + 1
            n_elig += 1
            n_nm += 1
            n_sites_dec += n_graph
            n_zero_sites += n_graph
            n_promotable += sum(1 for v in fmag if v > 0)
            continue
        m = rec.get('m')
        if not isinstance(m, list) or len(m) != n_graph:
            _mask(mid, 'label_len_mismatch', fmag)   # cannot happen post-attach
            continue
        # ── precondition 0.iv: the sign-convention (gauge) corner ──
        y_cur = _curated_y(mid, manifest, raw_fu, cap)
        s_m = sum(m)
        if (y_cur is not None and abs(y_cur) > gauge_min_y
                and abs(s_m) > gauge_min_summ and s_m * y_cur < 0):
            gauge_conflicts.append((mid, y_cur, s_m))
            _mask(mid, 'gauge_conflict', fmag)
            continue
        sign = []
        for v in m:
            if abs(v) > eps:
                sign.append(1 if v > 0 else -1)
            else:
                if abs(v) > 0:
                    n_eps_band += 1
                sign.append(0)
        seeds[mid] = {'elig': 1, 'why': 'ok', 'sign': sign, 'fmag': fmag}
        census['ok'] = census.get('ok', 0) + 1
        n_elig += 1
        n_sites_dec += n_graph
        n_plus += sum(1 for s in sign if s > 0)
        n_minus += sum(1 for s in sign if s < 0)
        n_zero_sites += sum(1 for s in sign if s == 0)
        n_promotable += sum(1 for s, v in zip(sign, fmag) if s == 0 and v > 0)
        if any(s > 0 for s in sign) and any(s < 0 for s in sign):
            n_mixed += 1

    meta = {
        'eps': eps, 'gauge_min_y': gauge_min_y,
        'gauge_min_summ': gauge_min_summ, 'cap': cap,
        'fmag_table_size': len(FORMAL_MAGNITUDE), 'fmag_max': FMAG_MAX,
        'census': dict(sorted(census.items(), key=lambda kv: -kv[1])),
        'n_rows': len(seeds), 'n_elig': n_elig, 'n_nm_seed': n_nm,
        'n_mixed_sign': n_mixed, 'n_sites_decorated': n_sites_dec,
        'site_signs': {'plus': n_plus, 'minus': n_minus,
                       'zero': n_zero_sites},
        'n_promotable_sites': n_promotable, 'n_eps_band_sites': n_eps_band,
        'n_gauge_conflicts': len(gauge_conflicts),
        'gauge_conflict_ids': [g[0] for g in gauge_conflicts],
        'gauge_conflict_rows': gauge_conflicts,
    }
    return seeds, meta


# ── independent tripwire (raw-jsonl re-derivation on a sample) ─────────────────
def verify_seed_vs_sidetable(seeds, magmom_jsonl, eps, sample_n=500):
    """Re-derive seed signs for up to sample_n eligible 'ok' rows STRAIGHT from
    the raw side-table and compare elementwise with the built table. The build
    path goes jsonl -> build_magmom_labels -> labels['m'] -> signs; this path
    goes jsonl -> signs. Any disagreement means the two paths diverged
    (indexing/ordering bug) -> RuntimeError before an epoch can train."""
    want = {m for m, r in seeds.items() if r.get('elig') and r['why'] == 'ok'}
    checked = 0
    with open(magmom_jsonl) as fh:
        for line in fh:
            if checked >= sample_n:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            mid = str(rec.get('material_id', ''))
            if mid not in want:
                continue
            magmom = rec.get('magmom')
            if not isinstance(magmom, list):
                raise RuntimeError(f'SEED TRIPWIRE: {mid} is seed-eligible but '
                                   f'its side-table row has no magmom list.')
            expect = [(1 if v > eps else (-1 if v < -eps else 0))
                      for v in (float(x) for x in magmom)]
            got = seeds[mid]['sign']
            if got != expect:
                bad = next(i for i, (a, b) in enumerate(zip(got, expect))
                           if a != b)
                raise RuntimeError(
                    f'SEED TRIPWIRE FAILED: {mid} site {bad} carries seed sign '
                    f'{got[bad]}, raw side-table re-derivation says '
                    f'{expect[bad]} (magmom {magmom[bad]}, eps {eps}) — the '
                    f'seed build path diverged from the label build path; do '
                    f'NOT train.')
            checked += 1
    if want and checked == 0:
        raise RuntimeError('SEED TRIPWIRE: no eligible row could be re-checked '
                           'against the side-table — wrong file?')
    return checked


# ── attach into the collated master (+ slices) ─────────────────────────────────
def attach_seed_channels(dataset, seeds):
    """Write seed_sign / seed_fmag [sum n_i] (node-aligned slices cloned from
    slices['x']) and the per-graph seed_elig int8 into the COLLATED MASTER, so
    every split view and every cached batch carries them — the same master-
    store discipline as attach_magmom_labels. Full-sweep length invariant
    inside. Returns (ids_in_master_order, n_elig_graphs)."""
    store = getattr(dataset, '_data', None)
    if store is None:
        store = dataset.data
    sl = getattr(dataset, 'slices', None)
    if sl is None:
        sl = getattr(dataset, '_slices', None)
    if sl is None or 'x' not in sl:
        raise RuntimeError("cannot locate slices['x'] on the dataset (PyG API "
                           "drift?) — node-aligned seed attach would break.")
    raw_ids = store.material_id
    ids = [str(x[0] if isinstance(x, (list, tuple)) else x) for x in raw_ids]
    xb = sl['x'].tolist()
    if len(xb) != len(ids) + 1:
        raise RuntimeError(f"slices['x'] has {len(xb)} offsets for {len(ids)} "
                           f'graphs — unexpected master layout.')
    n_nodes_total = int(xb[-1])
    sign_t = torch.zeros(n_nodes_total, dtype=torch.float32)
    fmag_t = torch.zeros(n_nodes_total, dtype=torch.float32)
    elig_t = torch.zeros(len(ids), dtype=torch.int8)
    n_elig = 0
    for j, mid in enumerate(ids):
        rec = seeds.get(mid)
        if rec is None:
            raise RuntimeError(f'SEED ATTACH FAILED: {mid} is in the dataset '
                               f'master but has no seed record — '
                               f'build_seed_table must cover every id.')
        lo, hi = xb[j], xb[j + 1]
        fmag = rec.get('fmag')
        if fmag is not None:
            if len(fmag) != hi - lo:
                raise RuntimeError(f'SEED ATTACH INVARIANT FAILED: {mid} has '
                                   f'{len(fmag)} formal magnitudes for a '
                                   f'{hi - lo}-node graph slice.')
            fmag_t[lo:hi] = torch.tensor(fmag, dtype=torch.float32)
        if rec.get('elig'):
            sign = rec['sign']
            if len(sign) != hi - lo:
                raise RuntimeError(f'SEED ATTACH INVARIANT FAILED: {mid} has '
                                   f'{len(sign)} seed signs for a '
                                   f'{hi - lo}-node graph slice.')
            sign_t[lo:hi] = torch.tensor(sign, dtype=torch.float32)
            elig_t[j] = 1
            n_elig += 1
    store.seed_sign = sign_t
    store.seed_fmag = fmag_t
    store.seed_elig = elig_t
    sl['seed_sign'] = sl['x'].clone()
    sl['seed_fmag'] = sl['x'].clone()
    sl['seed_elig'] = torch.arange(len(ids) + 1)
    return ids, n_elig


# ── THE read path ──────────────────────────────────────────────────────────────
def get_seed_sign(data):
    """Per-node seed sign [num_nodes] in {-1, 0, +1} (0 where ineligible)."""
    return data.seed_sign.reshape(-1)


def get_seed_fmag(data):
    """Per-node formal spin-only magnitude [num_nodes] in raw uB (0 for
    non-magnetic species / missing side-table rows)."""
    return data.seed_fmag.reshape(-1)


def get_seed_elig(data):
    """Per-graph seed eligibility flag [num_graphs] (int8, 1 = decoratable)."""
    return data.seed_elig.reshape(-1)


# ── cache verification (full sweep) ────────────────────────────────────────────
def verify_seed_in_cache(cache, seeds, tag):
    """End-to-end proof that the CACHED batches carry the built seed table,
    read through the same accessors decoration consumes. EVERY graph in EVERY
    batch is checked; raises on the first mismatch."""
    n_elig = n_graphs = 0
    for b in cache:
        mids = [str(x[0] if isinstance(x, (list, tuple)) else x)
                for x in b.material_id]
        elig = get_seed_elig(b).tolist()
        sign = get_seed_sign(b)
        fmag = get_seed_fmag(b)
        counts = torch.bincount(b.batch, minlength=len(mids)).tolist()
        if len(mids) != len(elig):
            raise RuntimeError(f'SEED VERIFY FAILED [{tag}]: {len(mids)} ids '
                               f'vs {len(elig)} seed_elig entries.')
        exp_sign, exp_fmag = [], []
        for m, e, c in zip(mids, elig, counts):
            rec = seeds.get(m)
            if rec is None:
                raise RuntimeError(f'SEED VERIFY FAILED [{tag}]: {m} has no '
                                   f'seed record.')
            if int(e) != int(rec.get('elig', 0)):
                raise RuntimeError(f'SEED VERIFY FAILED [{tag}]: {m} carries '
                                   f'seed_elig={int(e)}, build says '
                                   f'{int(rec.get("elig", 0))} '
                                   f'({rec.get("why")}).')
            fm = rec.get('fmag') or [0.0] * c
            if len(fm) != c:
                raise RuntimeError(f'SEED VERIFY FAILED [{tag}]: {m} has '
                                   f'{len(fm)} fmag entries but {c} nodes.')
            exp_fmag.extend(fm)
            if rec.get('elig'):
                exp_sign.extend(float(s) for s in rec['sign'])
                n_elig += 1
            else:
                exp_sign.extend([0.0] * c)
        n_graphs += len(mids)
        es = torch.tensor(exp_sign, dtype=sign.dtype)
        ef = torch.tensor(exp_fmag, dtype=fmag.dtype)
        if sign.numel() != es.numel() or not torch.equal(sign.cpu(), es):
            raise RuntimeError(f'SEED VERIFY FAILED [{tag}]: cached seed_sign '
                               f'differs from the built table.')
        if fmag.numel() != ef.numel() or not torch.allclose(
                fmag.cpu(), ef, atol=1e-6):
            raise RuntimeError(f'SEED VERIFY FAILED [{tag}]: cached seed_fmag '
                               f'differs from the built table.')
    print(f'    seed verify [{tag}]: {n_elig}/{n_graphs} graphs seed-eligible '
          f'— cached batches match the built seed table.')


# ── runtime decoration (train corruption / eval modes) ─────────────────────────
DEC_MODES = ('train', 'oracle', 'undecorated', 'shuffled', 'fm_formal')


def decorate_batch(data, mode, dcfg, gen=None):
    """Append the three decoration channels to data.x and (train mode) apply
    the corruption schedule + the global co-flip. Works on a per-step
    CLONE of a cached batch (CachedLoader semantics) on CPU or GPU; RNG draws
    happen on CPU through `gen` and are moved, so streams are device-agnostic
    and resume-reproducible.

    dcfg keys: base_channels (int, widening guard), p_undecorate, p_site_mask,
    p_promote, fmag_scale, coflip (bool).

    Modes:
        'train'        eligibility -> keep-prob (1 - p_undecorate); site mask;
                       promotion of zero-sign magnetic-species sites; co-flip
                       of seed + every signed supervision target.
        'oracle'       true seeds on every eligible graph; no corruption.
        'undecorated'  three zero channels (== the undecorated input + dead width).
        'shuffled'     oracle, then the seed channels PERMUTED within each
                       graph (leakage probe: the multiset survives, the
                       arrangement dies).
        'fm_formal'    the proposer's trivial FM hypothesis on EVERY graph:
                       sign = +1 on magnetic species (fmag > 0), flag = 1 on
                       all sites — the Xu-conditioned model's undecorated-mode stand-in when
                       cfg.xu_undecorated_policy == 'fm_formal' (its projected
                       head emits 0 on unseeded sites, so true zeros would be
                       degenerate), and a diagnostic mode for SD-ChemGNN.

    Sets on the batch:  seed_dec  [G] bool  (graph is decorated)
                        seed_flip [G] bool  (co-flip applied; train mode only)
    Mutates on flipped graphs (train mode): node_magmom, resid_target, y's
    magnetization column AND the named per-f.u. attribute — the read-path
    lesson: get_magnetization() prefers the named tensor, so BOTH are flipped.
    """
    if mode not in DEC_MODES:
        raise ValueError(f'decorate_batch: unknown mode {mode!r}')
    x = data.x
    base = int(dcfg['base_channels'])
    if x.size(1) != base:
        raise RuntimeError(f'decorate_batch: x has {x.size(1)} channels, '
                           f'expected the undecorated base {base} — double '
                           f'decoration or a stale cache.')
    dev = x.device
    N = x.size(0)
    batch_vec = data.batch if getattr(data, 'batch', None) is not None \
        else torch.zeros(N, dtype=torch.long, device=dev)
    G = int(data.num_graphs) if hasattr(data, 'num_graphs') \
        else int(batch_vec.max()) + 1

    sign = get_seed_sign(data).to(dev).clone()
    fmag = get_seed_fmag(data).to(dev)
    elig = get_seed_elig(data).to(dev).bool()

    flip = torch.zeros(G, dtype=torch.bool, device=dev)
    if mode == 'undecorated':
        dec = torch.zeros(G, dtype=torch.bool, device=dev)
    elif mode in ('oracle', 'shuffled'):
        dec = elig.clone()
    elif mode == 'fm_formal':
        dec = torch.ones(G, dtype=torch.bool, device=dev)
        sign = torch.where(fmag > 0, torch.ones_like(sign),
                           torch.zeros_like(sign))
    else:                                                   # 'train'
        keep = (torch.rand(G, generator=gen) >= float(dcfg['p_undecorate']))
        dec = elig & keep.to(dev)

    dec_nodes = dec[batch_vec]                              # [N] bool
    flag = dec_nodes.to(x.dtype)

    if mode == 'train':
        p_sm = float(dcfg['p_site_mask'])
        if p_sm > 0:
            site_mask = (torch.rand(N, generator=gen) < p_sm).to(dev) & dec_nodes
            flag = torch.where(site_mask, torch.zeros((), dtype=x.dtype,
                                                      device=dev), flag)
        p_pr = float(dcfg['p_promote'])
        if p_pr > 0:
            promotable = (sign == 0) & (fmag > 0) & (flag > 0)
            promote = (torch.rand(N, generator=gen) < p_pr).to(dev) & promotable
            rand_sign = torch.where(
                torch.rand(N, generator=gen).to(dev) < 0.5,
                torch.full((N,), -1.0, dtype=sign.dtype, device=dev),
                torch.full((N,), 1.0, dtype=sign.dtype, device=dev))
            sign = torch.where(promote, rand_sign, sign)
        if dcfg.get('coflip', True):
            flip = (torch.rand(G, generator=gen) < 0.5).to(dev) & dec
            flip_nodes = flip[batch_vec]
            sign = torch.where(flip_nodes, -sign, sign)
            # co-flip EVERY signed supervision target of the flipped graphs
            # (a global time reversal — the physics is identical, the stored
            # gauge is not; the reconstruction/total/resid/node losses must
            # see one consistent gauge or they fight):
            if getattr(data, 'node_magmom', None) is not None:
                nm = data.node_magmom.reshape(-1)
                data.node_magmom = torch.where(flip_nodes.to(nm.device),
                                               -nm, nm)
            if getattr(data, 'resid_target', None) is not None:
                rt = data.resid_target.reshape(-1)
                data.resid_target = torch.where(flip.to(rt.device), -rt, rt)
            named = getattr(data,
                            'total_magnetization_normalized_formula_units',
                            None)
            if named is not None:
                nv = named.reshape(-1)
                data.total_magnetization_normalized_formula_units = \
                    torch.where(flip.to(nv.device), -nv, nv).reshape(named.shape)
            if getattr(data, 'y', None) is not None and data.y.dim() == 2 \
                    and data.y.size(0) == G:
                col = 1 if data.y.size(1) > 1 else 0
                data.y[:, col] = torch.where(flip.to(data.y.device),
                                             -data.y[:, col], data.y[:, col])

    sign_ch = sign * flag                                   # no statement -> 0
    fmag_ch = (fmag / float(dcfg['fmag_scale'])) * flag

    if mode == 'shuffled':
        perm = torch.arange(N, device=dev)
        for gi in torch.nonzero(dec, as_tuple=True)[0].tolist():
            idx = torch.nonzero(batch_vec == gi, as_tuple=True)[0]
            if idx.numel() > 1:
                p = idx[torch.randperm(idx.numel(), generator=gen).to(dev)]
                perm[idx] = p
        sign_ch = sign_ch[perm]
        fmag_ch = fmag_ch[perm]
        flag = flag[perm]

    data.x = torch.cat([x, sign_ch.unsqueeze(1), fmag_ch.unsqueeze(1),
                        flag.unsqueeze(1)], dim=1)
    data.seed_dec = dec
    data.seed_flip = flip
    return data


# ── losses / metrics on top of the decoration ──────────────────────────────────
def node_pinned_loss(m_hat, m_true, batch_vec, num_graphs):
    """Per-graph node-mean L1 PINNED to the (possibly co-flipped) label gauge —
    no min-over-s. Returns L [num_graphs]."""
    e = (m_hat - m_true).abs()
    s = torch.zeros(num_graphs, dtype=m_hat.dtype,
                    device=m_hat.device).index_add_(0, batch_vec, e)
    cnt = torch.zeros(num_graphs, dtype=m_hat.dtype,
                      device=m_hat.device).index_add_(
        0, batch_vec, torch.ones_like(m_hat))
    return s / cnt.clamp(min=1)


def combined_node_loss(m_hat, m_true, batch_vec, num_graphs, dec_mask,
                       gauge_loss_fn):
    """Loss dispatch: gauge-PINNED on decorated graphs, min-over-s
    (gauge_loss_fn == site_moment_labels.node_gauge_loss) on undecorated
    ones. Returns (L [num_graphs], flip [num_graphs]) where flip is the
    gauge-resolution flag of the MIN branch, forced False on decorated graphs
    (their gauge is pinned, never resolved)."""
    L_min, flip = gauge_loss_fn(m_hat, m_true, batch_vec, num_graphs)
    L_pin = node_pinned_loss(m_hat, m_true, batch_vec, num_graphs)
    dec = dec_mask.to(torch.bool)
    L = torch.where(dec, L_pin, L_min)
    flip = flip & ~dec
    return L, flip


@torch.no_grad()
def self_consistency(m_hat, data, tiny=0.05):
    """Self-consistency readout: per-graph fraction of SEEDED sites (|sign channel| == 1
    after decoration) whose predicted sign reproduces the seed. Sites with
    |m_hat| <= tiny count as non-committal (excluded). Returns (frac [G] with
    NaN where a graph has no committal seeded sites, n_sites_checked)."""
    base = data.x.size(1) - SEED_CHANNELS
    sign_ch = data.x[:, base]
    batch_vec = data.batch
    G = int(data.num_graphs) if hasattr(data, 'num_graphs') \
        else int(batch_vec.max()) + 1
    seeded = sign_ch.abs() > 0.5
    committal = seeded & (m_hat.abs() > tiny)
    agree = committal & (torch.sign(m_hat) == torch.sign(sign_ch))
    ones = torch.ones_like(m_hat)
    n_c = torch.zeros(G, device=m_hat.device).index_add_(
        0, batch_vec, torch.where(committal, ones, torch.zeros_like(ones)))
    n_a = torch.zeros(G, device=m_hat.device).index_add_(
        0, batch_vec, torch.where(agree, ones, torch.zeros_like(ones)))
    frac = n_a / n_c.clamp(min=1)
    frac = torch.where(n_c > 0, frac, torch.full_like(frac, float('nan')))
    return frac, int(committal.sum())


# ── standalone dry run ─────────────────────────────────────────────────────────
def _main():
    ap = argparse.ArgumentParser(
        description='SD-ChemGNN seed-attach DRY RUN: build the seed table '
                    'against the real side-table + manifest (node counts '
                    'proxied by the side-table n_sites) and print the attach '
                    'census, the gauge-conflict corner and the tripwire '
                    'verdict. No dataset, no GPU, no training.')
    ap.add_argument('--sidetable', default='./magmom_sidetable.jsonl')
    ap.add_argument('--ordering', default='./ordering_sidetable.csv')
    ap.add_argument('--manifest', default='./label_manifest.csv')
    ap.add_argument('--collapse', default='./collapse_suspects.csv')
    ap.add_argument('--eps', type=float, default=0.1)
    ap.add_argument('--eps-alt', type=float, default=0.5,
                    help='second eps for the dead-zone sensitivity line')
    ap.add_argument('--gauge-min-y', type=float, default=1.0)
    ap.add_argument('--gauge-min-summ', type=float, default=0.1)
    ap.add_argument('--conflict-csv', default='',
                    help='write the gauge-conflict rows here (id, y_cur, sum_m)')
    a = ap.parse_args()

    # manifest, loaded the way the trainer loads it
    man = {}
    with open(a.manifest, newline='') as fh:
        for row in csv.DictReader(fh):
            try:
                new = float(row['new'])
            except (TypeError, ValueError, KeyError):
                new = None
            man[row['material_id']] = (row['action'], new)

    # node counts proxied by the side-table's own n_sites (the real attach
    # verifies graph == POSCAR sites; nsite_mismatch rows are ~0 by audit)
    node_counts = {}
    with open(a.sidetable) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ns = rec.get('n_sites')
            if isinstance(ns, int) and ns > 0:
                node_counts[str(rec['material_id'])] = ns
    # drops never reach the dataset — mirror that here
    for mid, (act, _n) in man.items():
        if act == 'drop':
            node_counts.pop(mid, None)
    print(f'dry run: {len(node_counts)} side-table rows as dataset proxy '
          f'(drops removed)')

    labels, lmeta = build_magmom_labels(
        a.sidetable, a.ordering, man, node_counts,
        collapse_path=(a.collapse if a.collapse and os.path.exists(a.collapse)
                       else None))
    print(f"labels: {lmeta['n_sup']}/{lmeta['n_rows']} supervised "
          f"({lmeta['n_zero']} zero rows, {lmeta['n_sup_nodes']} nodes); "
          f"census {lmeta['census']}")
    print(f"        sign sweep {lmeta['sign_sweep']}; residual recompute "
          f"{lmeta['resid_recompute']}")

    for eps in dict.fromkeys([a.eps, a.eps_alt]):
        seeds, meta = build_seed_table(
            a.sidetable, a.ordering, man, labels, node_counts,
            eps=eps, gauge_min_y=a.gauge_min_y,
            gauge_min_summ=a.gauge_min_summ)
        checked = verify_seed_vs_sidetable(seeds, a.sidetable, eps)
        tag = 'DEFAULT' if eps == a.eps else 'sensitivity'
        print(f'\n── seed table @ eps = {eps} ({tag}) ──')
        print(f"  eligible: {meta['n_elig']}/{meta['n_rows']} graphs "
              f"({meta['n_nm_seed']} NM seeds, {meta['n_mixed_sign']} "
              f"mixed-sign); {meta['n_sites_decorated']} decorated sites")
        s = meta['site_signs']
        print(f"  site signs  +{s['plus']}  -{s['minus']}  0:{s['zero']}   "
              f"eps-band sites {meta['n_eps_band_sites']}   promotable "
              f"(zero-sign magnetic species) {meta['n_promotable_sites']}")
        print(f"  gauge conflicts (precondition 0.iv corner): "
              f"{meta['n_gauge_conflicts']} rows seed-masked")
        print(f"  census: {meta['census']}")
        print(f'  tripwire: {checked} rows re-derived from the raw side-table '
              f'— consistent')
        if eps == a.eps and a.conflict_csv:
            with open(a.conflict_csv, 'w', newline='') as fh:
                w = csv.writer(fh)
                w.writerow(['material_id', 'y_curated', 'sum_magmom'])
                for mid, y, sm in meta['gauge_conflict_rows']:
                    w.writerow([mid, f'{y:.4f}', f'{sm:.4f}'])
            print(f"  wrote {a.conflict_csv} "
                  f"({meta['n_gauge_conflicts']} rows) for review")


if __name__ == '__main__':
    _main()
