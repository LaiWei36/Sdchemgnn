#!/usr/bin/env python
"""MagSieve's ordering ROUTER.

Fits a 4-class Materials Project ordering classifier (NM / FM / AFM / FiM) on
the features the scoring pass (part 0d) dumps, and emits per-row class probabilities for
the selector. Pure numpy + scikit-learn: no torch, no GPU, no cluster.

    fit on TRAIN  ->  tune on VAL  ->  touch TEST exactly once

THIS IS NOT THE AUXILIARY HEAD. Part 0b measured the auxiliary
ordering-diagnostic head: a 3-class OUTCOME-FAMILY classifier trained as an auxiliary loss for a
different objective, with cross-entropy weights balancing the global class
census. It is right on 80% of all labelled rows and 14% of the rows that decide
the shortlist. Three things differ here, each aimed at that measured failure:

  1. 4-class ORDERING labels, not 3-class outcome families. The routing
     decision is which arrangement to score, and the ordering class names one
     directly; the outcome family {NM, compensated AFM} merges cases needing
     different seeds.
  2. TWO reweightings, because the failure has two causes. `--class-weight`
     corrects the class census: on the real side-table the labelled population
     is NM 62,556 / FM 52,203 / FiM 10,362 / AFM 3,271, so AFM is 2.5% and FiM
     8.1% — and those two classes ARE the compensated structures that make up
     every shortlist false positive. `--budget-weight` separately up-weights
     rows with a large predicted moment budget, the population a shortlist is
     drawn from irrespective of class.
  3. ARRANGEMENT-RESPONSE features, not only the pooled embedding. Collapse
     and spread come from running the model on K arrangements and are not
     functions of the trunk, so they carry information from outside the
     magnetization model's blind spot. A supervised combination of magnitude
     with collapse and spread already reached 16.8 +/- 0.7 false positives
     in the top 1,000 (cross-validated) against 20.8 for magnitude alone, with
     a shuffled-feature null at 21.2 +/- 0.5 that never reached 18.

Honest prior: the pooled embedding carries the blind spot, so items 2 and 3 are
the only reasons to expect better. Judge on the SHORTLIST column that `report()`
prints, never on overall accuracy — the diagnostics showed a head can be right
80% of the time overall and 14% of the time where it decides anything.

LEAKAGE NOTE. The ordering side-table also carries `is_magnetic` and
`num_magnetic_sites`. Neither may be used as a router FEATURE: both are
Materials Project annotations derived from a converged DFT calculation, and a
newly generated structure has neither. The magnetic-site count computed from
COMPOSITION is a different quantity, is available at inference, and the
proposer already derives it.

    python ordering_router.py --dir ./magsieve_features [--features base,mode,arr,emb] [--model gbm] [--class-weight balanced|rare|none] [--budget-weight 8] [--ablate]
"""

from __future__ import annotations

import argparse
import csv
import json
import os.path as osp
import sys

import numpy as np

CLASSES = ('NM', 'FM', 'AFM', 'FiM')
CI = {c: i for i, c in enumerate(CLASSES)}
GROUPS = ('base', 'mode', 'arr', 'emb')


def norm_class(s):
    """Map a Materials Project `ordering` string onto our four classes."""
    t = str(s).strip().upper().replace('-', '').replace('_', '')
    if t in ('NM', 'NONMAGNETIC', 'NONE'):
        return 'NM'
    if t in ('FM', 'FERROMAGNETIC'):
        return 'FM'
    if t in ('AFM', 'ANTIFERROMAGNETIC'):
        return 'AFM'
    if t in ('FIM', 'FERRIMAGNETIC'):
        return 'FiM'
    return None                                   # 'Unknown' and blanks


# ══════════════════════════════════════════════════════════════════════════════
def load_split(d, name):
    """Read one split's 0d artifacts and build the feature matrix."""
    meta = osp.join(d, f'phase0d_{name}_meta.csv')
    if not osp.exists(meta):
        return None
    rows = list(csv.DictReader(open(meta)))
    n = len(rows)
    f = lambda k: np.array([float(r[k]) for r in rows])
    mid = np.array([r['material_id'] for r in rows])
    y = f('truth_capped')
    pred = f('pred_undec')
    sabs, smm = f('sum_abs_m'), f('sum_m')
    z = np.maximum(f('z_fu'), 1.0)
    resid = f('resid_fu')
    pm = np.column_stack([f('p_net0'), f('p_partial'), f('p_aligned')])
    mode = np.array([int(r['mode_label']) for r in rows])
    cls = np.array([norm_class(r.get('ordering_raw', '')) for r in rows],
                   dtype=object)

    budget = sabs / z
    align = np.abs(smm) / np.maximum(sabs, 1e-9)
    base = np.column_stack([np.log1p(np.abs(pred)), budget, align,
                            np.abs(resid) / np.maximum(np.abs(pred), 1e-9),
                            np.log1p(z)])

    # arrangement response, from the candidate table
    import magsieve_selector as SEL
    cpath = osp.join(d, f'phase0d_{name}_cand.txt')
    if not osp.exists(cpath):
        cpath = osp.join(d, 'phase0c_proposed.txt')       # test-split fallback
    if osp.exists(cpath):
        tab = SEL.CandidateTable.from_file(cpath, n)
        mags = np.full((n, len(CLASSES)), np.nan)
        for i in range(n):
            for j, c in enumerate(CLASSES):
                v = tab.pred(i, c)
                if v is not None:
                    mags[i, j] = abs(v)
        mg = mags[:, 1:]                       # magnetic arrangements only
        with np.errstate(invalid='ignore'):
            amin = np.nanmin(mg, axis=1)
            amax = np.nanmax(mg, axis=1)
        amin = np.nan_to_num(amin, nan=0.0)
        amax = np.nan_to_num(amax, nan=0.0)
        arr = np.column_stack([amin, amax, amax - amin,
                               amin / np.maximum(amax, 1e-9),
                               np.isfinite(mags).sum(1),
                               np.nan_to_num(mags, nan=-1.0)])
    else:
        tab = None
        arr = np.zeros((n, 9))

    h = np.load(osp.join(d, f'phase0d_{name}_h.npy'))
    if h.shape[0] != n:
        sys.exit(f'{name}: embedding has {h.shape[0]} rows, meta has {n}')
    return {'name': name, 'n': n, 'mid': mid, 'y': y, 'pred': pred,
            'budget': budget, 'cls': cls, 'mode': mode, 'table': tab,
            'groups': {'base': base, 'mode': pm, 'arr': arr, 'emb': h}}


def matrix(sp, groups):
    return np.column_stack([sp['groups'][g] for g in groups])


CLASS_WEIGHT_MODES = ('none', 'balanced', 'rare')


def class_weights(ycl, mode):
    """Per-row weights from the class census.

    THE CENSUS IS THE PROBLEM. On the real side-table the labelled population
    is NM 62,556 / FM 52,203 / FiM 10,362 / AFM 3,271 — AFM is 2.5% and FiM
    8.1%. Those two classes are precisely the compensated structures that make
    up every shortlist false positive. A classifier fitted to maximise accuracy
    on that census will hardly ever emit AFM, which is the most likely
    mechanism behind the auxiliary head being right on 80% of rows overall and
    14% of the rows that decide the shortlist.

        none      no reweighting — the control, and what the old head
                  effectively had for this purpose
        balanced  inverse frequency; every class contributes equally
        rare      inverse frequency with AFM and FiM emphasised further, on
                  the grounds that a missed AFM costs a false positive while a
                  missed NM costs nothing the ranking notices
    """
    n_cls = len(CLASSES)
    cnt = np.bincount(ycl, minlength=n_cls).astype(float)
    if mode == 'none':
        return np.ones(ycl.size)
    inv = np.where(cnt > 0, cnt.sum() / np.maximum(cnt, 1) / n_cls, 0.0)
    if mode == 'rare':
        inv = inv * np.array([1.0, 1.0, 2.0, 2.0])      # NM FM AFM FiM
    return inv[ycl]


def fit_router(tr, groups, model='gbm', budget_weight=0.0, budget_cut=1.0,
               class_weight='balanced', seed=23647):
    """Fit on rows with a usable 4-class label.

    Two independent reweightings, because the measured failure has two causes
    and they are not the same one:

      `class_weight`  corrects the CLASS census — AFM at 2.5% and FiM at 8.1%
                      of labelled rows. Without it the classifier barely emits
                      the two classes that matter.
      `budget_weight` up-weights rows whose predicted site budget exceeds
                      `budget_cut` uB/f.u. — the population a shortlist is
                      actually drawn from, irrespective of class.

    Both are swept on val rather than guessed.
    """
    lab = np.array([CI[c] if c in CI else -1 for c in tr['cls']])
    m = lab >= 0
    if m.sum() < 100:
        sys.exit(f'only {int(m.sum())} labelled training rows — the ordering '
                 f'side-table did not provide 4-class labels.')
    X, ycl = matrix(tr, groups)[m], lab[m]
    w = class_weights(ycl, class_weight)
    if budget_weight:
        w = w * (1.0 + budget_weight * (tr['budget'][m] > budget_cut))
    if model == 'gbm':
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(random_state=seed,
                                             max_iter=300, learning_rate=0.08)
    else:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=4000, C=1.0))
    try:
        clf.fit(X, ycl, sample_weight=w)
    except TypeError:                              # pipelines want the step name
        clf.fit(X, ycl, logisticregression__sample_weight=w)
    return clf, int(m.sum())


def proba(clf, sp, groups):
    """Class probabilities in CLASSES order, filling absent classes with 0."""
    p = clf.predict_proba(matrix(sp, groups))
    out = np.zeros((sp['n'], len(CLASSES)))
    cols = clf.classes_ if hasattr(clf, 'classes_') else clf[-1].classes_
    for j, c in enumerate(cols):
        out[:, int(c)] = p[:, j]
    return out


def report(tag, sp, P, shortlist_k=1000):
    """Accuracy overall AND on the shortlist population, separately — the diagnostics
    showed the second is the one that decides anything."""
    lab = np.array([CI[c] if c in CI else -1 for c in sp['cls']])
    m = lab >= 0
    pred = P.argmax(1)
    acc = float((pred[m] == lab[m]).mean()) if m.any() else float('nan')
    maj = float(np.bincount(lab[m], minlength=4).max() / max(m.sum(), 1))
    top = np.argsort(-np.abs(sp['pred']), kind='mergesort')[:shortlist_k]
    fp = top[sp['y'][top] <= 2]
    mf = np.array([i for i in fp if lab[i] >= 0], dtype=int)
    accs = float((pred[mf] == lab[mf]).mean()) if mf.size else float('nan')
    print(f'  {tag:<26} n={int(m.sum()):6d}  acc {acc:.4f}  majority {maj:.4f}'
          f'   |  on the {len(fp)} shortlist FPs: acc {accs:.4f} (n={mf.size})')
    return {'acc': acc, 'majority': maj, 'acc_shortlist_fp': accs}


# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', required=True, help='the scoring pass (part 0d) output dir')
    ap.add_argument('--features', default='base,mode,arr,emb')
    ap.add_argument('--model', default='gbm', choices=('gbm', 'logistic'))
    ap.add_argument('--budget-weight', type=float, default=0.0)
    ap.add_argument('--class-weight', default='balanced',
                    choices=CLASS_WEIGHT_MODES,
                    help='corrects the AFM 2.5%% / FiM 8.1%% census; '
                         '"none" is the control')
    ap.add_argument('--budget-cut', type=float, default=1.0)
    ap.add_argument('--ablate', action='store_true',
                    help='sweep feature groups and budget weights on VAL')
    ap.add_argument('--out', default=None, help='write test probabilities here')
    a = ap.parse_args()
    groups = [g.strip() for g in a.features.split(',') if g.strip()]
    bad = set(groups) - set(GROUPS)
    if bad:
        sys.exit(f'unknown feature groups {sorted(bad)}; choose from {GROUPS}')

    sp = {n: load_split(a.dir, n) for n in ('train', 'val', 'test')}
    missing = [k for k, v in sp.items() if v is None]
    if missing:
        sys.exit(f'missing 0d artifacts for {missing} — the router must be '
                 f'fitted off the evaluation set, so train and val are not '
                 f'optional. Re-run the scoring pass with --parts 0d.')
    for k, v in sp.items():
        lab = sum(1 for c in v['cls'] if c in CI)
        print(f'  {k:<6} {v["n"]:7,d} rows, {lab:7,d} with a 4-class label')

    if a.ablate:
        print('\n  ABLATION on VAL — feature groups and budget weighting')
        print('  (judge on the shortlist column, not the overall one)\n')
        for gs in (['base'], ['base', 'mode'], ['base', 'arr'],
                   ['base', 'arr', 'mode'], ['base', 'emb'], list(GROUPS)):
            for bw in (0.0, a.budget_weight or 8.0):
                for cw in CLASS_WEIGHT_MODES:
                    clf, _ = fit_router(sp['train'], gs, a.model, bw,
                                        a.budget_cut, cw)
                    report(f'{"+".join(gs)} bw={bw:g} cw={cw}', sp['val'],
                           proba(clf, sp['val'], gs))
        return 0

    print(f'\n  fitting: features={groups} model={a.model} '
          f'budget_weight={a.budget_weight} cut={a.budget_cut} '
          f'class_weight={a.class_weight}')
    clf, n_fit = fit_router(sp['train'], groups, a.model, a.budget_weight,
                            a.budget_cut, a.class_weight)
    print(f'  fitted on {n_fit:,} labelled train rows\n')
    for k in ('train', 'val', 'test'):
        report(k, sp[k], proba(clf, sp[k], groups))
    P = proba(clf, sp['test'], groups)
    out = a.out or osp.join(a.dir, 'router_test_proba.txt')
    np.savetxt(out, P, header=' '.join(CLASSES), fmt='%.6f')
    with open(osp.join(a.dir, 'router_meta.json'), 'w') as fh:
        json.dump({'features': groups, 'model': a.model,
                   'budget_weight': a.budget_weight,
                   'budget_cut': a.budget_cut,
                   'class_weight': a.class_weight, 'n_fit': n_fit}, fh, indent=1)
    print(f'\n  -> {out}   (score with screen_shortlist.py)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
