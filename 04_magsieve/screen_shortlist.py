#!/usr/bin/env python
"""Shortlist scoring for MagSieve: applies a router to the candidate table,
runs the demote-only selector, and scores the top-1,000 shortlist.

Pure numpy: no torch, no GPU, no cluster. Conventions (the same as
`diagnostics_score.py`): rank on the absolute predicted moment, shortlist =
top 1000, a false positive is a shortlisted row with truth <= 2 uB/f.u.,
catastrophic means |err| >= 3 inside the [5, 10) band. A formula-based
ceiling clamp min(pred, ceiling + 2) is applied only to rows whose formula the
--ids file supplies; pass an ids file without a formula column (such as
phase0_test_ids.csv) for the paper's unclamped convention.

It always prints the controls beside the result, because the numbers only
mean something against them:

    unscreened           the undecorated prediction with no selector at all
    always-FM / -AFM     constant routers, the two constant-routing controls
    true-ordering router the reference ranking routed from the true ordering
                         class (not deployable)

and it asserts the two selector invariants rather than trusting them.

    python screen_shortlist.py --dir <features dir> --ids phase0_test_ids.csv --proba <features dir>/router_test_proba.txt [--abstain 0.0]
"""

from __future__ import annotations

import argparse
import csv
import os.path as osp
import sys

import numpy as np

import magsieve_selector as SEL

POS, MONEY, TOPK, CLAMP_SLACK = 2.0, 5.0, 1000, 2.0
MAG_MAX = {'Ti': 2, 'V': 3, 'Cr': 4, 'Mn': 5, 'Fe': 5, 'Co': 4, 'Ni': 2,
           'Cu': 1, 'Nb': 3, 'Mo': 3, 'Tc': 3, 'Ru': 4, 'Rh': 2, 'Pd': 1,
           'W': 2, 'Re': 3, 'Os': 2, 'Ir': 1, 'Ce': 1, 'Pr': 2, 'Nd': 3,
           'Pm': 4, 'Sm': 5, 'Eu': 7, 'Gd': 7, 'Tb': 6, 'Dy': 5, 'Ho': 4,
           'Er': 3, 'Tm': 2, 'Yb': 1, 'U': 3, 'Np': 4, 'Pu': 5}


def ceiling(formula):
    """The spin-only formal maximum per formula unit. Group multipliers are
    NOT optional — 3,700 of the 15,377 test formulas carry one, and dropping
    it undercuts those cells by up to a factor of nine."""
    import re
    tok = re.compile(r'([A-Z][a-z]?)(\d*\.?\d*)|(\()|(\)\s*(\d*\.?\d*))')
    stack, cur = [], {}

    def merge(dst, src, k=1.0):
        for e, v in src.items():
            dst[e] = dst.get(e, 0.0) + v * k
        return dst

    for m in tok.finditer(str(formula)):
        el, num, op, close, mult = m.groups()
        if op:
            stack.append(cur)
            cur = {}
        elif close is not None:
            inner, cur = cur, (stack.pop() if stack else {})
            merge(cur, inner, float(mult) if mult else 1.0)
        elif el:
            cur[el] = cur.get(el, 0.0) + (float(num) if num else 1.0)
    while stack:
        cur = merge(stack.pop(), cur)
    tot = sum(v * MAG_MAX.get(k, 0) for k, v in cur.items())
    return float(tot) if tot > 0 else 0.5


HDR = ('  %-38s %7s %7s %5s %8s %8s %8s'
       % ('candidate', 'MAE', 'tail', 'cat', 'P@1000', 'FP@1000', 'Rec@1000'))


def screen(tag, pred, rank, y, ceil, note=''):
    pred = np.minimum(pred, ceil + CLAMP_SLACK)
    r = np.minimum(np.asarray(rank, float), ceil + CLAMP_SLACK)
    e = np.abs(pred - y)
    band = (y >= 5) & (y < 10)
    top = np.argsort(-r, kind='mergesort')[:TOPK]
    money = y >= MONEY
    out = {'tag': tag, 'mae': float(e.mean()),
           'tail': float(e[y > POS].mean()), 'cat': int((e[band] >= 3).sum()),
           'p1000': float((y[top] > POS).mean()),
           'fp1k': int((y[top] <= POS).sum()),
           'rec': float(money[top].sum() / max(money.sum(), 1))}
    print('  %-38s %7.4f %7.3f %5d %8.4f %8d %8.4f   %s'
          % (tag, out['mae'], out['tail'], out['cat'], out['p1000'],
             out['fp1k'], out['rec'], note))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', required=True)
    ap.add_argument('--ids', required=True, help='test_ids_15377.csv (formulas)')
    ap.add_argument('--proba', default=None, help='router probabilities [N,4]')
    ap.add_argument('--abstain', type=float, default=0.0)
    ap.add_argument('--fallback', default='physical',
                    choices=('physical', 'abstain'),
                    help='what to do when the routed class has no candidate; '
                         'worth 4 shortlist false positives on one test set')
    ap.add_argument('--split', default='test')
    a = ap.parse_args()

    meta = osp.join(a.dir, f'phase0d_{a.split}_meta.csv')
    if not osp.exists(meta):
        sys.exit(f'{meta} not found — run score_configurations_with_sdchemgnn.py '
                 f'with --parts 0d first.')
    rows = list(csv.DictReader(open(meta)))
    n = len(rows)
    mid = [r['material_id'] for r in rows]
    y = np.clip(np.array([float(r['truth_capped']) for r in rows]), -30, 30)
    undec = np.array([float(r['pred_undec']) for r in rows])
    true_cls = [r.get('ordering_raw', '') for r in rows]

    form = {}
    for r in csv.DictReader(open(a.ids)):
        if r.get('formula'):
            form[r['material_id']] = r['formula']
    miss = sum(1 for m in mid if m not in form)
    if miss:
        print(f'  NOTE: {miss}/{n} rows have no formula; ceiling clamp off for '
              f'them.')
    ceil = np.array([ceiling(form[m]) if m in form else 1e9 for m in mid])

    cpath = osp.join(a.dir, f'phase0d_{a.split}_cand.txt')
    if not osp.exists(cpath):
        cpath = osp.join(a.dir, 'phase0c_proposed.txt')
    table = SEL.CandidateTable.from_file(cpath, n)
    print(f'Shortlist scoring — {n:,} rows, {table.n_pairs:,} (structure, arrangement) '
          f'pairs')
    print(f'  candidate coverage by class: {table.coverage()}')

    import ordering_router as RT
    norm = [RT.norm_class(c) for c in true_cls]
    print(f'  4-class truth available on '
          f'{sum(1 for c in norm if c):,}/{n:,} rows\n')

    print(HDR)
    print('  ' + '-' * 92)
    res = []
    res.append(screen('unscreened: undecorated prediction', undec, np.abs(undec),
                      y, ceil, '<- reported values; must not be made worse'))

    fb = (SEL.FALLBACK_PHYSICAL if a.fallback == 'physical'
          else SEL.FALLBACK_ABSTAIN)
    def run(tag, router, note='', demote_only=True):
        s = SEL.select(undec, table, router, abstain_below=a.abstain,
                       demote_only=demote_only, fallback=fb)
        SEL.assert_invariants(s, undec, demote_only=demote_only)
        r = screen(tag, s['reported'], s['rank'], y, ceil, note)
        r['summary'] = SEL.summarise(s)
        return r

    for c in ('FM', 'AFM'):
        res.append(run(f'constant router: always {c}', SEL.ConstantRouter(n, c),
                       'control'))
    if a.proba and osp.exists(a.proba):
        P = np.loadtxt(a.proba)
        if P.shape[0] != n:
            sys.exit(f'router probabilities have {P.shape[0]} rows for {n}')
        res.append(run('MagSieve (learned router)  <== the result', SEL.ProbRouter(P),
                       f'abstain<{a.abstain:g}, fb={a.fallback}'))
    else:
        print('  (no --proba: skipping the learned router)')
    if any(norm):
        res.append(run('true-ordering router (reference)',
                       SEL.OracleRouter([c or '' for c in norm]),
                       'reference: true classes, not deployable'))

    floor = res[0]
    print('\n  STRUCTURAL CHECKS (these must hold exactly, not approximately)')
    ok = True
    for r in res[1:]:
        same = (abs(r['mae'] - floor['mae']) < 1e-9
                and abs(r['tail'] - floor['tail']) < 1e-9
                and r['cat'] == floor['cat'])
        ok &= same
        print(f'    {r["tag"]:<40} MAE/tail/cat identical to unscreened: '
              f'{"yes" if same else "NO — RULE 2 BROKEN"}')
    print(f'\n  {"all structural checks passed" if ok else "STRUCTURAL CHECK FAILED"}')

    for r in res[1:]:
        s = r.get('summary')
        if s:
            print(f'\n  {r["tag"]}: acted on {s["acted"]}, abstained '
                  f'{s["abstained"]}, routed-without-demotion '
                  f'{s["routed_no_demotion"]}; by class {s["by_class"]}')

    learned = next((r for r in res if 'MagSieve' in r['tag']), None)
    if learned:
        print('\n  RESULT')
        if learned['fp1k'] < floor['fp1k']:
            print(f'    MagSieve leaves {learned["fp1k"]} false positives in the '
                  f'top 1,000 against {floor["fp1k"]} unscreened; every '
                  f'reported value is unchanged.')
        else:
            print(f'    NO GAIN: {learned["fp1k"]} false positives against '
                  f'{floor["fp1k"]} unscreened. The two rules guarantee the '
                  f'reported values are unchanged.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
