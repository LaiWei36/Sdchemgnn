#!/usr/bin/env python
"""The composition-descriptor forest of the paper's Table 2 (SD-ChemGNN's
comparator): a random forest on 78 Ward-scheme composition statistics,
trained on the same 123,044 training structures as SD-ChemGNN and scored on
the same 15,377 test structures.

    MAE 0.620 / tail MAE 1.140 / per-atom MAE 0.057 uB   (random_state 23647)

It uses no structure, only the reduced formula: composition-weighted mean,
max, min, range and standard deviation of 14 per-element properties from
pymatgen's periodic table (Z, electronegativity, row, group, atomic mass,
atomic radius, Mendeleev number, valence s/p/d/f counts, transition-metal
and lanthanide/actinide flags, the formal spin-only moment) plus eight
composition-level terms (element count, atoms per formula unit, magnetic
fraction, magnetic-species count, formal ceiling per f.u. and per atom,
largest formal moment, log atoms per f.u.). The forest is scikit-learn's
RandomForestRegressor with 300 trees and min_samples_leaf 1.

Inputs (all produced by the earlier stages; run from the working directory):
    --train-meta   phase0d_train_meta.csv   of the MagSieve scoring pass
    --test-meta    phase0d_test_meta.csv    (score_configurations_with_sdchemgnn.py
                   --parts 0d --splits train,val,test): material_id and the
                   curated, capped label (truth_capped) of every structure of
                   the split, and for the test split SD-ChemGNN's undecorated
                   prediction (pred_undec), which is printed on the same table
    --sidetable    ordering_sidetable.csv (stage 1): the formula of every id

Every metric follows the paper's conventions: labels and predictions are
signed and capped at +/-30 uB/f.u.; the tail is truth > 2; per-atom error
divides by atoms per formula unit of the reduced formula; the shortlist is
the top 1,000 by |prediction| with no ceiling clamp, and a false positive is
a shortlisted structure with truth <= 2. The false-positive count is printed
for completeness; Table 2 reports MAE, tail MAE and per-atom MAE.

One third-decimal detail. The SD-ChemGNN row printed here (from pred_undec)
reads 0.543 / 0.962 / 0.053 against the paper's 0.543 / 0.960 / 0.053: the
paper's tail value was read from the trainer's own dump, where 16 labels of
exactly 2 uB/f.u. carry float32 noise (2.0000002) and fall on the tail side
of the strict threshold (4,777 tail structures there, 4,761 here). MAE and
per-atom error are unaffected, and the forest row is the paper's.

    python composition_forest_comparator.py --train-meta ./magsieve_features/phase0d_train_meta.csv --test-meta ./magsieve_features/phase0d_test_meta.csv --sidetable ./ordering_sidetable.csv

About 10 minutes on two cores for the fit. --seed 11311 and --seed 48907
give 0.619 / 1.138 / 0.057 and 0.617 / 1.135 / 0.057 (the seed spread quoted
in the project's notes); --pred-out writes the test predictions.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import time
from collections import Counter

import numpy as np

# The spin-only formal maximum table — the same one the formula ceiling and
# the decoration channels use, so "magnetic species" means one thing
# everywhere.
MAG_MAX = {'Ti': 2, 'V': 3, 'Cr': 4, 'Mn': 5, 'Fe': 5, 'Co': 4, 'Ni': 2,
           'Cu': 1, 'Nb': 3, 'Mo': 3, 'Tc': 3, 'Ru': 4, 'Rh': 2, 'Pd': 1,
           'W': 2, 'Re': 3, 'Os': 2, 'Ir': 1, 'Ce': 1, 'Pr': 2, 'Nd': 3,
           'Pm': 4, 'Sm': 5, 'Eu': 7, 'Gd': 7, 'Tb': 6, 'Dy': 5, 'Ho': 4,
           'Er': 3, 'Tm': 2, 'Yb': 1, 'U': 3, 'Np': 4, 'Pu': 5}

CAP = 30.0
POS = 2.0
TOPK = 1000
CLAMP_SLACK = 2.0

# Parentheses are NOT optional here: 3,700 of the 15,377 test formulas carry a
# group multiplier (Na2Ca(AsO3)4, Ba2CaAl2(Fe4O7)2, ...). A flat element-count
# regex silently drops the multiplier and undercounts those cells by a factor of
# up to nine, which lands squarely on the per-atom normalization the paper
# compares against the literature. The mean atoms/f.u. over the test set is
# the check: it must come out at 14.34.
_TOK = re.compile(r'([A-Z][a-z]?)(\d*\.?\d*)|(\()|(\)\s*(\d*\.?\d*))')


def parse_formula(f):
    c, stack, cur = Counter(), [], Counter()
    for m in _TOK.finditer(str(f)):
        el, num, op, _cl, mult = m.groups()
        if op:
            stack.append(cur)
            cur = Counter()
        elif _cl is not None:
            k = float(mult) if mult else 1.0
            inner = Counter({e: v * k for e, v in cur.items()})
            cur = stack.pop() if stack else Counter()
            cur.update(inner)
        elif el:
            cur[el] += float(num) if num else 1.0
    while stack:                       # unbalanced '(' — keep what we parsed
        prev = stack.pop()
        prev.update(cur)
        cur = prev
    c.update(cur)
    return c


def reduced(c):
    """Reduce a cell composition to its formula unit. Returns (comp, atoms_fu).

    Per-atom error is per-f.u. error divided by atoms per formula unit, and Z
    cancels, so the reduced formula is all that is needed.
    """
    vals = [v for v in c.values() if v > 0]
    if not vals:
        return c, 1.0
    ints = [int(round(v)) for v in vals]
    if all(abs(v - i) < 1e-6 for v, i in zip(vals, ints)) and all(ints):
        g = ints[0]
        for i in ints[1:]:
            g = math.gcd(g, i)
        g = max(g, 1)
        c = Counter({k: v / g for k, v in c.items()})
    return c, float(sum(c.values()))


# ══════════════════════════════════════════════════════════════════════════════
#  features — Magpie-style composition statistics from pymatgen element data
# ══════════════════════════════════════════════════════════════════════════════
def element_table():
    """Per-element property table. pymatgen is the source where available; the
    fallback keeps the script runnable without it, at reduced feature quality
    (it says so on stdout rather than silently degrading)."""
    try:
        from pymatgen.core.periodic_table import Element
    except Exception as exc:
        print(f'  pymatgen unavailable ({type(exc).__name__}) — falling back to '
              f'Z and the formal-moment table only. The forest will be weaker '
              f'than the published one and the comparison correspondingly '
              f'softer; install pymatgen for the real feature set.')
        return None, ['Z', 'mag_max']

    props = ['Z', 'X', 'row', 'group', 'atomic_mass', 'atomic_radius',
             'mendeleev_no', 'n_val_s', 'n_val_p', 'n_val_d', 'n_val_f',
             'is_tm', 'is_lanth_act', 'mag_max']
    tab = {}
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)   # pymatgen's
    # "no electronegativity / Mendeleev number for X" notes on noble gases and
    # superheavy elements, none of which occur in the corpus; they read as 0.
    for el in Element:
        try:
            fes = el.full_electronic_structure
        except Exception:
            fes = []
        nv = {'s': 0.0, 'p': 0.0, 'd': 0.0, 'f': 0.0}
        maxn = max((n for n, _, _ in fes), default=0)
        for n, orb, occ in fes:
            if orb == 'd' and n == maxn - 1:
                nv['d'] += occ
            elif orb == 'f' and n == maxn - 2:
                nv['f'] += occ
            elif n == maxn and orb in ('s', 'p'):
                nv[orb] += occ
        tab[el.symbol] = [
            float(el.Z),
            float(el.X) if el.X and not math.isnan(el.X) else 0.0,
            float(el.row), float(el.group), float(el.atomic_mass),
            float(el.atomic_radius) if el.atomic_radius else 0.0,
            float(el.mendeleev_no) if el.mendeleev_no else 0.0,
            nv['s'], nv['p'], nv['d'], nv['f'],
            1.0 if el.is_transition_metal else 0.0,
            1.0 if (el.is_lanthanoid or el.is_actinoid) else 0.0,
            float(MAG_MAX.get(el.symbol, 0.0)),
        ]
    return tab, props


def featurise(comps, tab, props):
    """Composition-weighted mean / max / min / range / std of every element
    property, plus composition-level terms (the formal ceiling, the magnetic
    fraction, the element count): 14 properties x 5 statistics + 8 = 78."""
    names, rows = [], []
    for p in props:
        names += [f'{p}_mean', f'{p}_max', f'{p}_min', f'{p}_range', f'{p}_std']
    names += ['n_elements', 'atoms_per_fu', 'frac_magnetic_sites',
              'n_magnetic_species', 'ceiling_fu', 'ceiling_per_atom',
              'max_mag_max', 'log_atoms_per_fu']
    npr = len(props)
    for c in comps:
        c, atoms = reduced(Counter(c))
        els = [e for e in c if (tab is None or e in tab)]
        if not els:
            rows.append(np.zeros(npr * 5 + 8))
            continue
        w = np.array([c[e] for e in els], float)
        w = w / w.sum()
        if tab is None:
            P = np.array([[float(MAG_MAX.get(e, 0.0)) * 0 + 1.0,
                           float(MAG_MAX.get(e, 0.0))] for e in els])
        else:
            P = np.array([tab[e] for e in els], float)
        mean = w @ P
        std = np.sqrt(np.maximum(w @ (P - mean) ** 2, 0.0))
        blk = np.concatenate([mean, P.max(0), P.min(0),
                              P.max(0) - P.min(0), std])
        mm = np.array([MAG_MAX.get(e, 0.0) for e in els], float)
        n_mag_sites = sum(c[e] for e in els if MAG_MAX.get(e, 0.0) > 0)
        ceil = float(sum(c[e] * MAG_MAX.get(e, 0.0) for e in els))
        rows.append(np.concatenate([blk, [
            float(len(els)), atoms,
            n_mag_sites / max(atoms, 1e-9),
            float(sum(1 for e in els if MAG_MAX.get(e, 0.0) > 0)),
            ceil, ceil / max(atoms, 1e-9), float(mm.max()),
            math.log1p(atoms)]]))
    return np.asarray(rows), names


# ══════════════════════════════════════════════════════════════════════════════
def screening(tag, pred, y, atoms, ceil, note=''):
    e = np.abs(pred - y)
    band = (y >= 5) & (y < 10)
    neg = y <= POS
    top = np.argsort(-np.abs(pred), kind='mergesort')[:TOPK]
    per_atom = float((e / np.maximum(atoms, 1e-9)).mean())
    print('  %-32s %7.4f %7.4f %7.4f %8.4f %5d %8.3f %8d   %s'
          % (tag, e.mean(), e[y > POS].mean(), e[~(y > POS)].mean(), per_atom,
             int((e[band] >= 3).sum()), float((y[top] > POS).mean()),
             int((y[top] <= POS).sum()), note))
    return {'mae': float(e.mean()), 'tail': float(e[y > POS].mean()),
            'per_atom': per_atom, 'p1000': float((y[top] > POS).mean()),
            'fp1k': int((y[top] <= POS).sum())}


HDR = ('  %-32s %7s %7s %7s %8s %5s %8s %8s'
       % ('model', 'MAE', 'tail', 'bulk', 'per-atom', 'cat', 'P@1000', 'FP@1000'))


def read_meta(path):
    rows = list(csv.DictReader(open(path)))
    ids = [r['material_id'] for r in rows]
    y = np.clip(np.array([float(r['truth_capped']) for r in rows]), -CAP, CAP)
    pred = (np.array([float(r['pred_undec']) for r in rows])
            if rows and 'pred_undec' in rows[0] else None)
    return ids, y, pred


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--train-meta', default='./magsieve_features/phase0d_train_meta.csv')
    ap.add_argument('--test-meta', default='./magsieve_features/phase0d_test_meta.csv')
    ap.add_argument('--sidetable', default='./ordering_sidetable.csv',
                    help='ordering_sidetable.csv: material_id -> formula')
    ap.add_argument('--trees', type=int, default=300)
    ap.add_argument('--seed', type=int, default=23647,
                    help='random_state of the forest (the paper: 23647)')
    ap.add_argument('--jobs', type=int, default=-1)
    ap.add_argument('--pred-out', default=None,
                    help='write material_id, forest prediction, label for the test set')
    a = ap.parse_args()

    form = {}
    for r in csv.DictReader(open(a.sidetable)):
        if r.get('formula'):
            form[r['material_id']] = r['formula']
    tr_ids, y_tr, _ = read_meta(a.train_meta)
    te_ids, y_te, undec = read_meta(a.test_meta)
    for name, ids in (('train', tr_ids), ('test', te_ids)):
        miss = [i for i in ids if i not in form]
        if miss:
            sys.exit(f'{len(miss)} {name} ids have no formula in {a.sidetable} '
                     f'(first: {miss[0]}); the side-table must cover the dataset')
    print(f'Composition-descriptor forest — train {len(tr_ids):,} / test '
          f'{len(te_ids):,} structures, {a.trees} trees, random_state {a.seed}')

    tab, props = element_table()
    t0 = time.time()
    X_tr, names = featurise([parse_formula(form[i]) for i in tr_ids], tab, props)
    X_te, _ = featurise([parse_formula(form[i]) for i in te_ids], tab, props)
    atoms = np.array([reduced(parse_formula(form[i]))[1] for i in te_ids])
    ceil = np.array([X_te[i, names.index('ceiling_fu')] for i in range(len(X_te))])
    print(f'  features: {X_tr.shape[1]} composition descriptors '
          f'({time.time() - t0:.1f}s); mean atoms/f.u. on the test set '
          f'{atoms.mean():.2f}')

    from sklearn.ensemble import RandomForestRegressor
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=a.trees, min_samples_leaf=1,
                               random_state=a.seed, n_jobs=a.jobs)
    rf.fit(X_tr, y_tr)
    pred = rf.predict(X_te)
    print(f'  fitted on {len(y_tr):,} training structures in '
          f'{time.time() - t0:.0f}s')

    print(f'\n{HDR}')
    print('  ' + '-' * 96)
    screening('null (predict zero)', np.zeros_like(y_te), y_te, atoms, ceil)
    screening('formal ceiling as a predictor', ceil, y_te, atoms, ceil,
              note='the physics prior alone')
    screening('composition-descriptor forest', pred, y_te, atoms, ceil,
              note='Table 2 row')
    if undec is not None:
        screening('SD-ChemGNN, undecorated', undec, y_te, atoms, ceil,
                  note='from the test meta (pred_undec)')

    if a.pred_out:
        with open(a.pred_out, 'w') as fh:
            fh.write('material_id,forest_pred,y_capped\n')
            for i, p, t in zip(te_ids, pred, y_te):
                fh.write(f'{i},{p:.6f},{t:.6f}\n')
        print(f'\n  test predictions -> {a.pred_out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
