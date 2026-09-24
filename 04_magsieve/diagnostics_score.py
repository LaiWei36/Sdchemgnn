#!/usr/bin/env python
"""Offline scorer for the diagnostics parts (0a, 0b, 0c) of
`score_configurations_with_sdchemgnn.py`. Pure numpy: NO torch, NO GPU, NO
dataset. It reads the files that job writes and answers three questions:

    0a  Would a self-consistency filter discard anything?
    0b  Can the auxiliary ordering head route WITHOUT a decoration, on the
        population that needs routing?
    0c  Does the spread of predictions across candidate arrangements separate
        compensated structures from real magnets — and at what cost in false
        negatives?

The screening conventions are the paper's: rank on the absolute predicted
moment; a shortlist is the top 1000; a false positive is a shortlisted row
with truth <= 2 uB/f.u.; catastrophic means |err| >= 3 inside the [5, 10)
band; the formula ceiling clamp is min(pred, ceiling + slack) and is applied
only when formulas are supplied.

USAGE
    python diagnostics_score.py --dir ./magsieve_diagnostics [--ids test_ids.csv] [--baseline .../regression_undec.txt]
"""

from __future__ import annotations

import argparse
import csv
import os.path as osp
import re
import sys

import numpy as np

TOPK = 1000
POS = 2.0          # truth > POS  is "worth pursuing"
MONEY = 5.0        # truth >= MONEY is the high-moment band the screen exists for
CLAMP_SLACK = 2.0


# ══════════════════════════════════════════════════════════════════════════════
def rule(title=''):
    print('\n' + '=' * 92)
    if title:
        print(title)
        print('=' * 92)


def auc(pos, neg):
    """Mann-Whitney AUC: P(a random `pos` scores above a random `neg`), ties
    at 0.5. Rank-based, so it needs no calibration and no threshold — the
    right summary for "does this quantity separate the two populations".
    """
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float('nan')
    allv = np.concatenate([pos, neg])
    r = np.empty(allv.size, float)
    order = np.argsort(allv, kind='mergesort')
    sv = allv[order]
    i = 0
    while i < sv.size:                       # average ranks within ties
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((r[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0)
                 / (pos.size * neg.size))


# ── the project's physics ceiling (spin-only formal maximum per f.u.) ─────────
MAG_MAX = {'Ti': 2, 'V': 3, 'Cr': 4, 'Mn': 5, 'Fe': 5, 'Co': 4, 'Ni': 2,
           'Cu': 1, 'Nb': 3, 'Mo': 3, 'Tc': 3, 'Ru': 4, 'Rh': 2, 'Pd': 1,
           'W': 2, 'Re': 3, 'Os': 2, 'Ir': 1, 'Ce': 1, 'Pr': 2, 'Nd': 3,
           'Pm': 4, 'Sm': 5, 'Eu': 7, 'Gd': 7, 'Tb': 6, 'Dy': 5, 'Ho': 4,
           'Er': 3, 'Tm': 2, 'Yb': 1, 'U': 3, 'Np': 4, 'Pu': 5}


# Parentheses are NOT optional: 3,700 of the 15,377 test formulas carry a group
# multiplier (Na2Ca(AsO3)4, Ba2CaAl2(Fe4O7)2, ...). Dropping it undercounts
# those cells by up to a factor of nine and the ceiling clamp with them. This is
# the same grammar the earlier screening module implemented, checked to agree with
# it on all 15,377 rows.
_TOK = re.compile(r'([A-Z][a-z]?)(\d*\.?\d*)|(\()|(\)\s*(\d*\.?\d*))')


def _parse_formula(f):
    stack, cur = [], {}

    def merge(dst, src, k=1.0):
        for e, v in src.items():
            dst[e] = dst.get(e, 0.0) + v * k
        return dst

    for m in _TOK.finditer(str(f)):
        el, num, op, close, mult = m.groups()
        if op:
            stack.append(cur)
            cur = {}
        elif close is not None:
            inner, cur = cur, (stack.pop() if stack else {})
            merge(cur, inner, float(mult) if mult else 1.0)
        elif el:
            cur[el] = cur.get(el, 0.0) + (float(num) if num else 1.0)
    while stack:                        # unbalanced '(': keep what was parsed
        cur = merge(stack.pop(), cur)
    return cur


def ceiling(formula):
    try:
        c = _parse_formula(formula)
    except Exception:
        return 1e9
    tot = sum(n * MAG_MAX.get(el, 0) for el, n in c.items())
    return float(tot) if tot > 0 else 0.5


# ══════════════════════════════════════════════════════════════════════════════
def screening(tag, pred, y, rank=None, ceil=None, note=''):
    """One scoreboard line, plus the numbers the aggregation question needs.

    `pred` is the value whose error is measured; `rank` is the value the
    shortlist is ordered by, which is NOT the same thing under a demotion rule
    — demotion changes the ranking without inventing a new prediction, and
    that distinction is the whole point of aggregating rather than selecting.
    """
    pred = np.asarray(pred, float)
    r = np.abs(pred) if rank is None else np.asarray(rank, float)
    if ceil is not None:
        pred = np.minimum(pred, ceil + CLAMP_SLACK)
        r = np.minimum(r, ceil + CLAMP_SLACK)
    e = np.abs(pred - y)
    band = (y >= 5) & (y < 10)
    neg = y <= POS
    top = np.argsort(-r, kind='mergesort')[:TOPK]
    money = y >= MONEY
    rec = float(money[top].sum() / max(money.sum(), 1))
    print('  %-36s %7.4f %7.3f %5d %8.3f %7.3f %8d %8.3f   %s'
          % (tag, e.mean(), e[y > POS].mean() if (y > POS).any() else np.nan,
             int((e[band] >= 3).sum()) if band.any() else 0,
             float((y[top] > POS).mean()),
             float(((np.abs(pred) > POS) & neg).sum() / max(neg.sum(), 1)),
             int((y[top] <= POS).sum()), rec, note))
    return {'tag': tag, 'mae': float(e.mean()), 'p1000': float((y[top] > POS).mean()),
            'fp1k': int((y[top] <= POS).sum()), 'recall_money': rec}


HDR = ('  %-36s %7s %7s %5s %8s %7s %8s %8s'
       % ('candidate', 'MAE', 'tail', 'cat', 'P@1000', 'FPR', 'FP@1000', 'Rec@1000'))


# ══════════════════════════════════════════════════════════════════════════════
def load_index(d, ids_csv=None):
    p = osp.join(d, 'phase0_test_ids.csv')
    if not osp.exists(p):
        sys.exit(f'{p} not found — run score_configurations_with_sdchemgnn.py first.')
    rows = list(csv.DictReader(open(p)))
    mid = np.array([r['material_id'] for r in rows])
    y = np.array([float(r['truth_capped']) for r in rows])
    ceil = None
    if ids_csv:
        by = {}
        for r in csv.DictReader(open(ids_csv)):
            f = r.get('formula')
            if f:
                by[r['material_id']] = ceiling(f)
        miss = sum(1 for m in mid if m not in by)
        if miss:
            print(f'  NOTE: {miss}/{len(mid)} rows have no formula in '
                  f'{osp.basename(ids_csv)} — ceiling clamp disabled for them.')
        ceil = np.array([by.get(m, 1e9) for m in mid])
    return mid, y, ceil


# ══════════════════════════════════════════════════════════════════════════════
def score_0a(d):
    p = osp.join(d, 'phase0a_selfconsistency.txt')
    if not osp.exists(p):
        return
    rule('0a — does the self-consistency filter discard anything?')
    a = np.loadtxt(p)
    names = ['oracle', 'shuffled', 'fm_formal']
    for i, n in enumerate(names[:a.shape[1]]):
        c = a[:, i]
        f = c[np.isfinite(c)]
        print(f'  {n:<10} committal on {f.size:6d} graphs   mean {f.mean():.4f}   '
              f'>= 0.90 on {100.0 * (f >= 0.90).mean():5.1f}%   '
              f'== 1.00 on {100.0 * (f >= 0.9999).mean():5.1f}%')
    if a.shape[1] >= 2:
        o, s = a[:, 0], a[:, 1]
        m = np.isfinite(o) & np.isfinite(s)
        sep = auc(o[m], s[m])
        surv = float((s[m] >= 0.90).mean())
        print(f'\n  separation AUC(oracle > shuffled)  {sep:.4f}   '
              f'(0.50 = the filter is blind)')
        print(f'  a shuffled arrangement survives an sc >= 0.90 filter '
              f'{100 * surv:.1f}% of the time')
        if sep < 0.60 or surv > 0.5:
            print('\n  VERDICT: the filter does NOT discriminate. Selector v0\'s '
                  'primary filter would\n           pass wrong arrangements at '
                  'essentially the rate it passes right ones,\n           so it '
                  'cannot be the mechanism that protects the fallback. This is '
                  'the\n           anticipated case: abstention has to '
                  'come from somewhere else.')
        else:
            print('\n  VERDICT: the filter separates. Self-consistency is usable '
                  'as selector v0\'s\n           first stage — but check the '
                  'survival rate above before trusting it alone.')


def score_0b(d, y):
    p = osp.join(d, 'phase0b_modes_undec.txt')
    if not osp.exists(p):
        return
    rule('0b — decoration-free routing by the auxiliary head')
    a = np.loadtxt(p)
    P, lab = a[:, :3], a[:, 6].astype(int)
    pred = P.argmax(1)
    cls = ['net~0', 'partial', 'aligned']
    yy = y[:len(lab)]
    for scope, m in (('all labelled rows', lab >= 0),
                     ('magnetic rows (|truth| > 0.5)', (lab >= 0) & (np.abs(yy) > 0.5)),
                     # the population routing actually has work on: rows the
                     # moments of which do NOT simply add up, so the
                     # arrangement is what decides the answer
                     ('arrangement matters (label != aligned)',
                      (lab >= 0) & (lab != 2)),
                     ('high-moment band (truth >= 5)', (lab >= 0) & (yy >= MONEY))):
        if not m.any():
            continue
        acc = float((pred[m] == lab[m]).mean())
        maj = float(np.bincount(lab[m], minlength=3).max() / m.sum())
        print(f'  {scope:<32s} n={int(m.sum()):6d}   accuracy {acc:.4f}   '
              f'majority {maj:.4f}   {"beats" if acc > maj else "LOSES TO"} a constant')
    m = lab >= 0
    print('\n  per class (all labelled):')
    for k, c in enumerate(cls):
        mk = m & (lab == k)
        if mk.any():
            print(f'    {c:<9} n={int(mk.sum()):6d}   recall '
                  f'{float((pred[mk] == k).mean()):.4f}')
    print('\n  BREAK-EVEN, scope stated explicitly (the rev-7 audit finding):')
    print('    a substituting selector would need 59.7% arrangement accuracy on the FULL split merely to')
    print('    beat not trying. Compare that against the "all labelled rows" line')
    print('    above — not against a hard-subset number. Routing a 3-way ordering')
    print('    mode is also a WEAKER task than picking one of K arrangements, so')
    print('    this accuracy is an UPPER bound on what a mode-head selector could do.')


def aggregate_candidates(path, n):
    """Collapse the (structure, arrangement) table into per-structure statistics.

    THE `nm` CANDIDATE IS EXCLUDED FROM THE ENSEMBLE STATISTICS, deliberately.
    `nm` is not an alternative arrangement of the moments; it is the separate
    hypothesis that the material carries none. It predicts near zero for
    everything, so including it drives min/max to ~0 for every structure and
    the collapse ratio degenerates into a constant — it would measure "is the
    nm hypothesis in the candidate set", which is always yes. The question
    aggregation asks is narrower and only meaningful among the MAGNETIC
    arrangements: given that this material has moments, can they cancel?
    `nm` is kept as its own column instead, where a large value is a genuine
    signal that the residual head is driving the prediction regardless of the
    arrangement — the failure shape of a sign-projected head.

    Returns a dict of length-n arrays plus `n_pairs` and `K`.
    """
    A = [[] for _ in range(n)]           # |pred| over MAGNETIC arrangements
    B = [[] for _ in range(n)]           # per-f.u. site budget, same set
    SCF = [[] for _ in range(n)]         # self-consistency, same set
    fm = np.full(n, np.nan)
    nm = np.full(n, np.nan)
    n_pairs, K = 0, 0
    with open(path) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            f = line.split()
            if len(f) < 12:
                continue
            row, name, k = int(f[0]), f[3], int(f[2])
            pred, sabs, z, sc = float(f[6]), float(f[7]), float(f[10]), float(f[11])
            if row >= n:
                continue
            n_pairs += 1
            K = max(K, k + 1)
            if name == 'nm':
                nm[row] = abs(pred)
                continue
            if name == 'fm':
                fm[row] = abs(pred)
            A[row].append(abs(pred))
            B[row].append(sabs / max(z, 1.0))
            SCF[row].append(sc)
    nk = np.array([len(v) for v in A])
    a_min = np.array([min(v) if v else np.nan for v in A])
    a_max = np.array([max(v) if v else np.nan for v in A])
    a_med = np.array([float(np.median(v)) if v else np.nan for v in A])
    budget = np.array([float(np.median(v)) if v else np.nan for v in B])
    sc_med = np.array([float(np.nanmedian(v)) if v and np.isfinite(v).any()
                       else np.nan for v in SCF])
    return {'n_pairs': n_pairs, 'K': K, 'nk': nk, 'min': a_min, 'max': a_max,
            'med': a_med, 'budget': budget, 'sc': sc_med, 'fm': fm, 'nm': nm,
            'collapse': a_min / np.maximum(a_max, 1e-9),
            'spread': a_max - a_min}


def score_0c(d, mid, y, ceil, baseline=None):
    p = osp.join(d, 'phase0c_proposed.txt')
    if not osp.exists(p):
        return
    rule('0c — aggregation: does the ensemble separate compensation from magnetism?')
    n = len(mid)
    st = aggregate_candidates(p, n)
    if st['n_pairs'] == 0:
        print('  no candidate rows — nothing to score.')
        return
    a_min, a_max, a_med = st['min'], st['max'], st['med']
    budget, collapse, spread = st['budget'], st['collapse'], st['spread']
    FMPRED, NMPRED, nk = st['fm'], st['nm'], st['nk']
    print(f'  {st["n_pairs"]:,} scored (structure, arrangement) pairs over '
          f'{n:,} structures; up to K = {st["K"]}')
    print(f'  magnetic arrangements per structure: mean {nk.mean():.2f}, '
          f'{int((nk == 0).sum())} with none, {int((nk == 1).sum())} with one')
    print('  (the nm hypothesis is held out of the ensemble statistics — see '
          'aggregate_candidates)')
    print(f'  collapse ratio  median {np.nanmedian(collapse):.3f}   '
          f'spread median {np.nanmedian(spread):.3f} uB/f.u.')

    # ---- THE decision number ----
    # A shortlist false positive is a structure that looks magnetic but is not.
    # If aggregation works, those are exactly the structures whose predictions
    # COLLAPSE under some arrangement, while real magnets hold up under all of
    # them. Restricting the negative class to rows the model itself calls
    # magnetic is deliberate: separating true magnets from obvious non-magnets
    # is not the problem, and scoring against all negatives would flatter the
    # result.
    looks_mag = np.nan_to_num(a_max, nan=0.0) > POS
    compensated = looks_mag & (y <= POS)
    real = y >= MONEY
    print(f'\n  populations: {int(compensated.sum()):,} compensated '
          f'(some arrangement predicts > {POS:g} but truth <= {POS:g}), '
          f'{int(real.sum()):,} real magnets (truth >= {MONEY:g})')
    # Each statistic is oriented so that HIGH means "real magnet", then scored
    # as P(a random real magnet outranks a random compensated structure).
    print('\n  AUC — P(a real magnet outranks a compensated structure):')
    for nm_, q in (('collapse ratio  (holds up under every arrangement)', collapse),
                   ('negative spread (arrangement-insensitive)', -spread),
                   ('median site budget per f.u.', budget),
                   ('the fm hypothesis alone (the no-aggregation control)', FMPRED),
                   ('the nm hypothesis (residual-only prediction)', NMPRED)):
        print(f'    {nm_:<52s} {auc(q[real], q[compensated]):.4f}')
    print('\n    0.50 means the ensemble sees no difference and aggregation fails on its')
    print('    own terms. The budget line is the control that matters most: it is high')
    print('    for compensated structures too, which is exactly why ranking on the net')
    print('    moment alone has never separated them. If the collapse ratio beats it by')
    print('    a wide margin, the ensemble is carrying information the single-pass model')
    print('    does not expose.')

    # ---- candidate ranking scores ----
    rule('0c — candidate ranking scores on the project\'s screening protocol')
    print(HDR)
    print('  ' + '-' * 96)
    out = []
    if baseline is not None:
        out.append(screening('undecorated baseline', baseline, y, ceil=ceil,
                             note='<- the deployable, unscreened ranking'))
    out.append(screening('fm hypothesis alone (no aggregation)',
                         np.nan_to_num(FMPRED), y, ceil=ceil))
    out.append(screening('ensemble median', np.nan_to_num(a_med), y, ceil=ceil))
    out.append(screening('ensemble min (conservative)', np.nan_to_num(a_min),
                         y, ceil=ceil))
    out.append(screening('ensemble max (optimistic)', np.nan_to_num(a_max),
                         y, ceil=ceil))
    out.append(screening('median, ranked by budget', np.nan_to_num(a_med), y,
                         rank=np.nan_to_num(budget), ceil=ceil))

    rule('0c — collapse demotion: the false-negative exchange rate')
    print('  Demotion moves a row DOWN the shortlist without changing its '
          'prediction, so it\n  can only trade false positives for false '
          'negatives — never for a worse number.\n  The exchange rate is what '
          'that must be measured rather than argued about.')
    print()
    print(HDR)
    print('  ' + '-' * 96)
    base = screening('median, no demotion', np.nan_to_num(a_med), y, ceil=ceil)
    for tau in (0.10, 0.25, 0.40, 0.55, 0.70):
        dem = np.nan_to_num(collapse, nan=1.0) < tau
        rk = np.where(dem, -1e9, np.nan_to_num(a_med))
        r = screening(f'median, demote collapse < {tau:.2f}',
                      np.nan_to_num(a_med), y, rank=rk, ceil=ceil,
                      note=f'{int(dem.sum())} demoted')
        d_fp = base['fp1k'] - r['fp1k']
        d_tp = (base['recall_money'] - r['recall_money']) * max(int((y >= MONEY).sum()), 1)
        if d_fp == 0 and abs(d_tp) < 0.5:
            verdict = 'no change — the shortlist did not contain them'
        elif d_fp <= 0:
            verdict = 'a bad trade — it loses ground and buys nothing'
        elif d_tp <= 0:
            verdict = 'strictly better — false positives removed at no cost'
        else:
            verdict = (f'exchange rate {d_tp / d_fp:.2f} high-moment hits lost '
                       f'per false positive removed')
        print(f'      -> removes {d_fp:+d} false positives at a cost of '
              f'{d_tp:+.0f} high-moment true positives: {verdict}')
        out.append(r)
    return out


# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', required=True, help='the scoring pass --out directory')
    ap.add_argument('--ids', default=None,
                    help='test_ids_15377.csv (adds the physics ceiling clamp)')
    ap.add_argument('--baseline', default=None,
                    help='regression_undec.txt from the same run (the unscreened ranking)')
    a = ap.parse_args()

    mid, y, ceil = load_index(a.dir, a.ids)
    print(f'Diagnostics scorer — {len(mid):,} rows from {a.dir}')
    print(f'  truth: {int((y > POS).sum()):,} above {POS:g}, '
          f'{int((y >= MONEY).sum()):,} in the high-moment band (>= {MONEY:g})')
    if ceil is None:
        print('  no --ids: the physics ceiling clamp is OFF, so these numbers '
              'are not directly\n  comparable with clamped numbers. Supply '
              'test_ids_15377.csv to enable it.')

    base = None
    if a.baseline:
        b = np.loadtxt(a.baseline)
        b = b[:, 0] if b.ndim == 2 else b
        if b.size != y.size:
            print(f'  WARNING: baseline has {b.size} rows vs {y.size} scored — '
                  f'ignoring it rather than\n  aligning by position (that '
                  f'assumption once produced an off-by-one).')
        else:
            base = b

    score_0a(a.dir)
    score_0b(a.dir, y)
    score_0c(a.dir, mid, y, ceil, base)
    print()


if __name__ == '__main__':
    main()
