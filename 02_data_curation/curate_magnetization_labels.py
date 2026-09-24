#!/usr/bin/env python3
"""
curate_magnetization_labels.py — physics-checked label curation for the MP magnetization set,
plus a side experiment: rescore an existing run's predictions under the cleaned
labels (no retraining) to measure how much error was label artifact all along.

Usage (needs torch + the dataset; the shipped label_manifest.csv is the
audited product of these rules and does not need regenerating):
    python curate_magnetization_labels.py <path-to-a-run's *_regression.txt | run folder>

Outputs:
    label_manifest.csv   material_id, formula, split, old, new, action, reason
    console summary      entries touched per rule + before/after rescore

Rules (constants below; every decision lands in the manifest for review):
  R1  PLAUSIBILITY CEILING: |M| > ceiling(formula) + TOL is an artifact.
      ceiling = sum over the formula of MAX_SPIN[element] (generous high-spin
      upper bounds, uB/atom; DFT collinear totals are spin-dominated). Action:
      relabel to 0 when the chemistry is closed-shell (ceiling <= TOL: the
      compound is a valid NONMAGNET example); DROP when a magnetic chemistry
      merely exceeds its ceiling (which of its labels to trust is ambiguous).
  R2  STRADDLE RESOLUTION: a formula group whose R1-surviving labels span the
      2 uB threshold is an FM/NM convergence lottery. Screening semantics: the
      chemistry's moment is what it carries WHEN IT ORDERS -> relabel every
      entry in the group to the median of its |M|>2 members (never above the
      ceiling). Set R2_MODE='off' to disable, 'drop' to drop minority instead.
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
import csv
import glob
import os.path as osp
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

from sdchemgnn_config import config_magnetic as cfg
from build_crystal_graph_dataset import MPDataset

BULK_CUTOFF = float(cfg.bulk_cutoff)
TARGET_CAP  = float(cfg.target_cap)
TOL         = 2.0          # uB/f.u. slack: itinerant excess, O2/NO radical units
R2_MODE     = 'ordered-median'   # 'ordered-median' | 'drop' | 'off'

# Max SPIN moment per atom (uB), generous upper bounds. 0 for elements that do
# not carry moments in compounds (closed shell / always-quenched): catching
# LLZO@11 requires Zr=0 even though atomic Zr is d2 — Zr magnets don't exist.
# Edit freely; every flag cites the computed ceiling so review is easy.
MAX_SPIN = {
    # 3d
    'Sc': 1, 'Ti': 2, 'V': 3, 'Cr': 4, 'Mn': 5, 'Fe': 5, 'Co': 4, 'Ni': 2,
    'Cu': 1, 'Zn': 0,
    # 4d / 5d (moments rare: small ceilings; early d0-in-compounds set to 0)
    'Y': 0, 'Zr': 0, 'Nb': 1, 'Mo': 3, 'Tc': 3, 'Ru': 3, 'Rh': 2, 'Pd': 1,
    'Ag': 1, 'Cd': 0, 'Hf': 0, 'Ta': 1, 'W': 3, 'Re': 4, 'Os': 2, 'Ir': 1,
    'Pt': 0, 'Au': 0, 'Hg': 0,
    # 4f (spin-only; orbital parts are absent from collinear DFT totals)
    'La': 0, 'Ce': 1, 'Pr': 2, 'Nd': 3, 'Pm': 4, 'Sm': 5, 'Eu': 7, 'Gd': 7,
    'Tb': 6, 'Dy': 5, 'Ho': 4, 'Er': 3, 'Tm': 2, 'Yb': 1, 'Lu': 0,
    # 5f
    'Ac': 0, 'Th': 1, 'Pa': 2, 'U': 3, 'Np': 4, 'Pu': 5, 'Am': 6,
}   # every element not listed (s/p-block, noble gases): 0


def parse_formula(f):
    """MP formula string -> element Counter, handling parentheses+multipliers:
    'MgFe8(OF)8' -> {Mg:1, Fe:8, O:8, F:8}."""
    tokens = re.findall(r'[A-Z][a-z]?|\d+|\(|\)', f)
    stack = [Counter()]
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == '(':
            stack.append(Counter())
            i += 1
        elif t == ')':
            grp = stack.pop()
            i += 1
            mult = 1
            if i < len(tokens) and tokens[i].isdigit():
                mult = int(tokens[i])
                i += 1
            for el, c in grp.items():
                stack[-1][el] += c * mult
        elif t[0].isalpha():
            el = t
            i += 1
            c = 1
            if i < len(tokens) and tokens[i].isdigit():
                c = int(tokens[i])
                i += 1
            stack[-1][el] += c
        else:
            i += 1
    return stack[0]


def ceiling_of(formula):
    return float(sum(MAX_SPIN.get(el, 0) * c for el, c in parse_formula(formula).items()))


def scan_dataset():
    """Replicate the training split exactly; return per-entry records."""
    dataset = MPDataset(root=cfg.dataset_root, jsonl_path='')
    torch.manual_seed(cfg.seed)
    perm = torch.randperm(len(dataset))
    dataset = dataset[perm]
    n = len(dataset)
    n_train, n_val = int(n * cfg.train_p), int(n * cfg.val_p)
    recs = []
    for i in range(n):
        d = dataset[i]
        f = getattr(d, 'formula', None)
        f = str(f[0] if isinstance(f, (list, tuple)) else f).strip()
        mid = getattr(d, 'material_id', None)
        mid = str(mid[0] if isinstance(mid, (list, tuple)) else mid)
        yv = d.y.view(-1)
        m = float(yv[1]) if yv.numel() > 1 else float(yv[0])
        m = max(-TARGET_CAP, min(TARGET_CAP, m))
        split = 'train' if i < n_train else ('val' if i < n_train + n_val else 'test')
        test_row = i - n_train - n_val if split == 'test' else -1
        recs.append({'mid': mid, 'formula': f, 'split': split, 'row': test_row,
                     'old': m, 'new': m, 'action': 'keep', 'reason': ''})
        if (i + 1) % 20000 == 0:
            print(f'  [scan] {i + 1}/{n}')
    return recs


def apply_rules(recs):
    # R1: plausibility ceiling
    ceil_cache = {}
    for r in recs:
        c = ceil_cache.setdefault(r['formula'], ceiling_of(r['formula']))
        if abs(r['old']) > c + TOL:
            if c <= TOL:
                r['new'], r['action'] = 0.0, 'relabel0'
                r['reason'] = f'R1 closed-shell: |M|={abs(r["old"]):.2f} > ceiling {c:g}+{TOL:g}'
            else:
                r['action'] = 'drop'
                r['reason'] = f'R1 exceeds ceiling: |M|={abs(r["old"]):.2f} > {c:g}+{TOL:g}'
    # R2: straddle resolution on R1 survivors
    if R2_MODE != 'off':
        groups = defaultdict(list)
        for r in recs:
            if r['action'] != 'drop':
                groups[r['formula']].append(r)
        for f, g in groups.items():
            if len(g) < 2:
                continue
            ms = [abs(r['new']) for r in g]
            if not (min(ms) <= BULK_CUTOFF < max(ms)):
                continue
            mag = sorted(m for m in ms if m > BULK_CUTOFF)
            target = min(mag[len(mag) // 2], ceil_cache[f])
            for r in g:
                if R2_MODE == 'ordered-median' and abs(abs(r['new']) - target) > 1e-9:
                    r['new'] = target
                    if r['action'] == 'keep':
                        r['action'] = 'relabelR2'
                    r['reason'] = (r['reason'] + '; ' if r['reason'] else '') + \
                        f'R2 straddle group -> ordered median {target:.2f}'
                elif R2_MODE == 'drop' and abs(r['new']) <= BULK_CUTOFF:
                    r['action'], r['reason'] = 'drop', 'R2 straddle minority'
    return recs


def metrics(pred, true):
    err, at = np.abs(pred - true), np.abs(true)
    tail = at > BULK_CUTOFF
    out = {'MAE': err.mean(), 'tail': err[tail].mean() if tail.any() else float('nan')}
    edges = [BULK_CUTOFF, 5.0, 10.0, TARGET_CAP]
    for i in range(3):
        lo, hi = edges[i], edges[i + 1]
        m = (at > lo if i == 0 else at >= lo) & (at <= hi if i == 2 else at < hi)
        out[f'[{lo:g}-{hi:g}]'] = err[m].mean() if m.any() else float('nan')
    t, p = true[tail], pred[tail]
    out['R2'] = 1 - ((t - p) ** 2).sum() / max(((t - t.mean()) ** 2).sum(), 1e-12)
    order = np.argsort(-np.abs(pred))
    out['P@1000'] = (at[order[:1000]] > BULK_CUTOFF).mean()
    fp = np.abs(pred) > BULK_CUTOFF
    out['prec'] = (fp & tail).sum() / max(fp.sum(), 1)
    out['rec'] = (fp & tail).sum() / max(tail.sum(), 1)
    return out


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    path = sys.argv[1]
    if osp.isdir(path):
        path = sorted(glob.glob(osp.join(path, '*_regression.txt')))[-1]
    arr = np.loadtxt(path)
    pred, true_file = arr[:, 0], arr[:, 1]

    print('[curate] scanning dataset + replicating split...')
    recs = apply_rules(scan_dataset())

    acts = Counter(r['action'] for r in recs)
    print(f'\n[curate] actions over {len(recs)} entries: {dict(acts)}')
    with open('label_manifest.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['material_id', 'formula', 'split', 'old', 'new', 'action', 'reason'])
        for r in recs:
            if r['action'] != 'keep':
                w.writerow([r['mid'], r['formula'], r['split'],
                            f'{r["old"]:.4f}', f'{r["new"]:.4f}', r['action'], r['reason']])
    print('[curate] label_manifest.csv written (non-keep rows only)')

    # ── Side experiment: rescore the run's predictions under cleaned TEST labels ──
    test = sorted((r for r in recs if r['split'] == 'test'), key=lambda r: r['row'])
    assert len(test) == len(pred), (len(test), len(pred))
    mism = sum(abs(r['old'] - t) > 1e-3 for r, t in zip(test, true_file))
    print(f'[verify] test-row label match vs regression file: '
          f'{len(test) - mism}/{len(test)} exact' + (' — OK' if mism == 0 else ' *** MISMATCH ***'))
    keep = np.array([r['action'] != 'drop' for r in test])
    clean = np.array([r['new'] for r in test])
    before = metrics(pred, true_file)
    after = metrics(pred[keep], clean[keep])
    print(f'\n── predictions rescored (dropped {int((~keep).sum())} test entries) ──')
    print(f'{"metric":10s} {"original":>10s} {"cleaned":>10s}')
    for k in before:
        print(f'{k:10s} {before[k]:10.4f} {after[k]:10.4f}')


if __name__ == '__main__':
    main()
