#!/usr/bin/env python
"""Offline test suite for the MagSieve selector, router and shortlist scorer.

Exercises `magsieve_selector.py`, `ordering_router.py` and
`screen_shortlist.py` with no torch, no GPU, no cluster and no dataset. Most of
it is about the two selector rules, because a safety guarantee that is only
argued for is not a guarantee.

No pytest dependency. Exit status is 0 only if every check passed.

    python test_selector_offline.py
"""

from __future__ import annotations

import os
import os.path as osp
import subprocess
import sys
import tempfile
import traceback

import numpy as np

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
    except AssertionError as exc:
        FAIL.append(name)
        print(f'  FAIL  {name}\n        {exc}')
    except Exception:
        FAIL.append(name)
        print(f'  FAIL  {name}\n{traceback.format_exc(limit=3)}')
    else:
        PASS.append(name)
        print(f'  ok    {name}')


# ══════════════════════════════════════════════════════════════════════════════
#  a fixture whose answer is known by construction
# ══════════════════════════════════════════════════════════════════════════════
#   REAL   truth 8   undecorated predicts ~8, and an AFM arrangement exists
#          that collapses it — the situation that makes a naive minimum
#          dangerous and that rule 1 alone does NOT protect against.
#   COMP   truth 0   undecorated predicts ~8 (the false positives), and the
#          AFM arrangement collapses it. Only the ROUTER separates these two.
#   NON    truth 0   undecorated predicts ~0.
N_REAL, N_COMP, N_NON = 1400, 300, 2300


def fixture(d, split, rng, informative=True):
    """Write one split's 0d artifacts. `informative=False` decouples the
    features from the class, so a router fitted on it can only guess — the
    anti-test that catches a scorer reading the populations instead of the
    routing."""
    kinds = ['REAL'] * N_REAL + ['COMP'] * N_COMP + ['NON'] * N_NON
    n = len(kinds)
    truth = {'REAL': 8.0, 'COMP': 0.0, 'NON': 0.0}
    cls = {'REAL': 'FM', 'COMP': 'AFM', 'NON': 'NM'}
    y = np.array([truth[k] for k in kinds])
    mid = [f'mp-{split}-{i:05d}' for i in range(n)]
    undec = np.array([8.0 if k in ('REAL', 'COMP') else 0.05
                      for k in kinds]) + rng.normal(0, .02, n)
    # the ONE feature that separates REAL from COMP, and only when informative
    tell = np.array([1.0 if k == 'REAL' else 0.0 for k in kinds])
    if not informative:
        tell = rng.permutation(tell)
    tell = tell + rng.normal(0, .05, n)

    h = np.column_stack([tell, rng.normal(0, 1, (n, 3))]).astype(np.float32)
    np.save(osp.join(d, f'phase0d_{split}_h.npy'), h)
    with open(osp.join(d, f'phase0d_{split}_meta.csv'), 'w') as fh:
        fh.write('row,material_id,truth_capped,pred_undec,sum_abs_m,sum_m,'
                 'resid_fu,z_fu,p_net0,p_partial,p_aligned,mode_label,'
                 'ordering_raw\n')
        for i, k in enumerate(kinds):
            fh.write('%d,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.1f,%.4f,%.4f,%.4f,%d,%s\n'
                     % (i, mid[i], y[i], undec[i], abs(undec[i]), undec[i],
                        0.0, 1.0, 0.3, 0.2, 0.5, 0, cls[k]))
    with open(osp.join(d, f'phase0d_{split}_cand.txt'), 'w') as fh:
        fh.write('# row material_id cand_idx cand_name exact net_formal pred '
                 'sum_abs_m sum_m resid_fu z_fu sc_frac\n')
        for i, k in enumerate(kinds):
            cands = [('nm', 0.02), ('fm', abs(undec[i])),
                     ('afm_bipartite', 0.03)]
            if k == 'COMP':
                cands.append(('species_split_1', 0.04))
            for j, (nm, v) in enumerate(cands):
                fh.write('%d %s %d %s 1 0.0 %.6f %.6f %.6f 0.0 1.0 1.0\n'
                         % (i, mid[i], j, nm, v, abs(v), v))
    return y, undec, [cls[k] for k in kinds]


# ══════════════════════════════════════════════════════════════════════════════
def tier_rules(SEL):
    print('\ntier 1 — the two selector rules, as invariants')
    rng = np.random.default_rng(7)
    n = 400
    undec = rng.uniform(0, 10, n)
    t = SEL.CandidateTable(n)
    for i in range(n):
        t._by_cls[i] = {'NM': 0.01, 'FM': undec[i] * 1.7,   # a PROMOTION
                        'AFM': undec[i] * 0.1}              # a demotion
        t._names[i] = {c: c.lower() for c in t._by_cls[i]}

    def t_reported_never_changes():
        for cls in ('NM', 'FM', 'AFM'):
            s = SEL.select(undec, t, SEL.ConstantRouter(n, cls))
            assert np.array_equal(s['reported'], undec), cls
            SEL.assert_invariants(s, undec)
    check('rule 2: no router can change a reported prediction',
          t_reported_never_changes)

    def t_never_promotes():
        # FM is 1.7x the undecorated value on every row; under rule 1 not one
        # of them may raise the ranking score.
        s = SEL.select(undec, t, SEL.ConstantRouter(n, 'FM'))
        assert np.all(s['rank'] <= np.abs(undec) + 1e-12)
        assert s['acted'].sum() == 0, 'a promotion was acted on'
        assert np.allclose(s['rank'], np.abs(undec))
    check('rule 1: a larger candidate is refused, so nothing is promoted',
          t_never_promotes)

    def t_demotion_is_applied():
        s = SEL.select(undec, t, SEL.ConstantRouter(n, 'AFM'))
        assert s['acted'].all(), int(s['acted'].sum())
        assert np.allclose(s['rank'], np.abs(undec) * 0.1)
        SEL.assert_invariants(s, undec)
    check('a smaller candidate does demote the ranking score',
          t_demotion_is_applied)

    def t_ablation_is_honest():
        # demote_only=False must ACTUALLY promote, or the ablation that shows
        # rule 1 is load-bearing would be measuring nothing.
        s = SEL.select(undec, t, SEL.ConstantRouter(n, 'FM'), demote_only=False)
        assert np.any(s['rank'] > np.abs(undec) + 1e-9), 'nothing was promoted'
        try:
            SEL.assert_invariants(s, undec, demote_only=True)
        except AssertionError:
            return
        raise AssertionError('assert_invariants failed to catch a promotion')
    check('the demote_only=False ablation really promotes, and is caught',
          t_ablation_is_honest)

    def t_abstain_reproduces_the_floor_exactly():
        s = SEL.select(undec, t, SEL.ProbRouter(np.full((n, 4), 0.25)),
                       abstain_below=0.99)
        assert s['abstained'].all()
        assert np.array_equal(s['rank'], np.abs(undec)), 'unscreened ranking not reproduced'
        assert s['acted'].sum() == 0
    check('a router that abstains everywhere reproduces the unscreened ranking exactly',
          t_abstain_reproduces_the_floor_exactly)

    def t_missing_candidate_abstains():
        t2 = SEL.CandidateTable(n)                 # empty table
        s = SEL.select(undec, t2, SEL.ConstantRouter(n, 'AFM'))
        assert s['abstained'].all()
        assert np.array_equal(s['rank'], np.abs(undec))
    check('a class with no candidate abstains rather than guessing',
          t_missing_candidate_abstains)

    def t_fallback_is_explicit_and_measurable():
        # A class the proposer produced no candidate for is a real decision.
        # ABSTAIN does nothing; PHYSICAL substitutes the nearest available
        # arrangement. On one real test set the two differ by 4 shortlist
        # false positives, so this must never be a silent dict.get chain.
        t3 = SEL.CandidateTable(3)
        for i in range(3):
            t3._by_cls[i] = {'AFM': 0.5}          # no FiM candidate anywhere
            t3._names[i] = {'AFM': 'afm_bipartite'}
        u3 = np.array([4.0, 4.0, 4.0])
        a = SEL.select(u3, t3, SEL.ConstantRouter(3, 'FiM'),
                       fallback=SEL.FALLBACK_ABSTAIN)
        assert a['abstained'].all() and np.allclose(a['rank'], u3)
        b = SEL.select(u3, t3, SEL.ConstantRouter(3, 'FiM'),
                       fallback=SEL.FALLBACK_PHYSICAL)
        assert b['acted'].all(), 'FiM should fall back to the AFM candidate'
        assert np.allclose(b['rank'], 0.5)
        assert list(b['cls']) == ['AFM'] * 3, 'the class acted on must be the '\
                                              'substitute, not the request'
        SEL.assert_invariants(b, u3)
    check('an unavailable class falls back explicitly, and records what it used',
          t_fallback_is_explicit_and_measurable)

    def t_demotion_can_still_admit_a_false_positive():
        # The correction. An earlier version of this suite asserted
        # that a demotion-only selector "cannot invent a false positive", and
        # it passed only because its fixture demoted every row equally and so
        # preserved the ordering. The claim is FALSE in general: a fixed-size
        # top-K means demoting a true positive frees a slot, and whatever sat
        # at rank K+1 moves in with its own score unchanged. This test pins the
        # true behaviour so the wrong claim cannot come back.
        u = np.array([9.0, 8.0, 7.0])
        y = np.array([9.0, 8.0, 0.0])          # row 2 is a false positive
        tt = SEL.CandidateTable(3)
        tt._by_cls[1] = {'AFM': 0.1}           # demote the TRUE magnet
        tt._names[1] = {'AFM': 'afm_bipartite'}
        s = SEL.select(u, tt, SEL.ConstantRouter(3, 'AFM'))
        SEL.assert_invariants(s, u)            # no rule is violated
        before = np.argsort(-np.abs(u), kind='mergesort')[:2]
        after = np.argsort(-s['rank'], kind='mergesort')[:2]
        assert int((y[before] <= 2).sum()) == 0
        assert int((y[after] <= 2).sum()) == 1, (
            'the whole point of this test is that a false positive IS admitted')
    check('demoting a true positive admits the next row, which may be a false '
          'positive', t_demotion_can_still_admit_a_false_positive)

    def t_what_the_rules_do_guarantee():
        # What survives the correction: no row outranks its unscreened position, and
        # no reported value moves.
        y = np.concatenate([np.full(200, 8.0), np.zeros(200)])
        u = np.concatenate([np.full(200, 8.0), np.full(200, 7.0)])
        tt = SEL.CandidateTable(400)
        for i in range(400):
            tt._by_cls[i] = {'AFM': 0.01}
            tt._names[i] = {'AFM': 'afm_bipartite'}
        s = SEL.select(u, tt, SEL.ConstantRouter(400, 'AFM'))
        SEL.assert_invariants(s, u)
        assert np.all(s['rank'] <= np.abs(u) + 1e-12)
        assert np.array_equal(s['reported'], u)
    check('what does survive: no row outranks its unscreened score, no value moves',
          t_what_the_rules_do_guarantee)


def tier_table(SEL):
    print('\ntier 2 — candidate table and family mapping')

    def t_family_map():
        for nm, fam in (('nm', 'NM'), ('fm', 'FM'), ('afm_bipartite', 'AFM'),
                        ('component_flip_2', 'AFM'), ('species_split_1', 'FiM'),
                        ('orbit_split_3', 'FiM')):
            assert SEL.family_of(nm) == fam, (nm, SEL.family_of(nm))
        assert SEL.family_of('something_else') is None
    check('every proposer candidate maps to exactly one ordering class',
          t_family_map)

    def t_first_of_family_wins():
        # priority order is the proposer's; taking the best of a family would
        # be a second, hidden selector
        with tempfile.TemporaryDirectory() as d:
            p = osp.join(d, 'c.txt')
            with open(p, 'w') as fh:
                fh.write('# hdr\n')
                fh.write('0 mp-x 0 afm_bipartite 1 0.0 5.0 5 5 0 1.0 1.0\n')
                fh.write('0 mp-x 1 component_flip_1 1 0.0 0.1 1 1 0 1.0 1.0\n')
            t = SEL.CandidateTable.from_file(p, 1)
            assert t.pred(0, 'AFM') == 5.0, t.pred(0, 'AFM')
            assert t.name(0, 'AFM') == 'afm_bipartite'
    check('the first candidate of a family is used, not the best',
          t_first_of_family_wins)

    def t_coverage():
        t = SEL.CandidateTable(3)
        t._by_cls[0] = {'FM': 1.0}
        t._by_cls[1] = {'FM': 1.0, 'AFM': 0.1}
        assert t.coverage() == {'NM': 0, 'FM': 2, 'AFM': 1, 'FiM': 0}
    check('coverage reports how often each class can be acted on at all',
          t_coverage)


def tier_router(RT):
    print('\ntier 3 — router label handling')

    def t_norm_class():
        for s, e in (('FM', 'FM'), ('fm', 'FM'), ('AFM', 'AFM'),
                     ('FiM', 'FiM'), ('ferrimagnetic', 'FiM'),
                     ('Non-magnetic', 'NM'), ('NM', 'NM')):
            assert RT.norm_class(s) == e, (s, RT.norm_class(s))
        for s in ('Unknown', '', 'garbage'):
            assert RT.norm_class(s) is None, s
    check('MP ordering strings map to the four classes; Unknown stays unlabelled',
          t_norm_class)


def tier_end_to_end(SEL, RT):
    print('\ntier 4 — end to end on a fixture whose answer is known')

    def build(seed, informative=True):
        d = tempfile.mkdtemp(prefix='sel-')
        rng = np.random.default_rng(seed)
        for sp in ('train', 'val', 'test'):
            y, u, cls = fixture(d, sp, rng, informative)
        with open(osp.join(d, 'ids.csv'), 'w') as fh:
            fh.write('material_id,formula\n')
            for i in range(len(y)):
                fh.write(f'mp-test-{i:05d},Fe8\n')
        return d, y, u

    def t_oracle_router_clears_the_shortlist():
        d, y, u = build(1)
        n = len(y)
        tab = SEL.CandidateTable.from_file(
            osp.join(d, 'phase0d_test_cand.txt'), n)
        import csv as _csv
        cls = [r['ordering_raw'] for r in
               _csv.DictReader(open(osp.join(d, 'phase0d_test_meta.csv')))]
        fp = lambda r: int((y[np.argsort(-r, kind='mergesort')[:1000]] <= 2).sum())
        base = fp(np.abs(u))
        s = SEL.select(u, tab, SEL.OracleRouter(cls))
        SEL.assert_invariants(s, u)
        # REAL and COMP are tied near the same predicted value, so the 1000-row
        # shortlist draws them in proportion: ~1000 * 300/1700 ~= 176.
        expect = 1000 * N_COMP / (N_REAL + N_COMP)
        assert 0.7 * expect <= base <= 1.3 * expect, (
            f'fixture baseline {base} is not the ~{expect:.0f} the tie '
            f'arithmetic predicts — the fixture changed')
        assert fp(s['rank']) == 0, (
            f'the oracle router should clear every compensated row from the '
            f'shortlist; {fp(s["rank"])} remain')
    check('the oracle router removes every compensated false positive',
          t_oracle_router_clears_the_shortlist)

    def t_learned_router_recovers_it():
        d, y, u = build(2)
        r = subprocess.run([sys.executable, 'ordering_router.py',
                            '--dir', d, '--features', 'base,emb',
                            '--model', 'gbm'],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        P = np.loadtxt(osp.join(d, 'router_test_proba.txt'))
        n = len(y)
        tab = SEL.CandidateTable.from_file(
            osp.join(d, 'phase0d_test_cand.txt'), n)
        s = SEL.select(u, tab, SEL.ProbRouter(P))
        SEL.assert_invariants(s, u)
        fp = lambda r_: int((y[np.argsort(-r_, kind='mergesort')[:1000]] <= 2).sum())
        assert fp(s['rank']) < 0.25 * fp(np.abs(u)), (
            f'the fixture carries a clean separating feature, so a fitted '
            f'router should remove most false positives: '
            f'{fp(np.abs(u))} -> {fp(s["rank"])}')
    check('a router fitted on train recovers the separation on test',
          t_learned_router_recovers_it)

    def t_anti_test():
        d, y, u = build(3, informative=False)
        r = subprocess.run([sys.executable, 'ordering_router.py',
                            '--dir', d, '--features', 'base,emb',
                            '--model', 'gbm'],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        P = np.loadtxt(osp.join(d, 'router_test_proba.txt'))
        n = len(y)
        tab = SEL.CandidateTable.from_file(
            osp.join(d, 'phase0d_test_cand.txt'), n)
        s = SEL.select(u, tab, SEL.ProbRouter(P))
        SEL.assert_invariants(s, u)
        fp = lambda r_: int((y[np.argsort(-r_, kind='mergesort')[:1000]] <= 2).sum())
        # With the feature decoupled the router cannot separate REAL from COMP.
        # The two rules mean that costs recall, never precision — so what must
        # be asserted is that nothing BLEW UP, not that it improved.
        assert fp(s['rank']) <= fp(np.abs(u)), (
            'with no usable signal the selector must not make precision worse')
    check('with the signal removed the selector degrades safely, never worse',
          t_anti_test)

    def t_scorer_runs_and_asserts():
        d, y, u = build(4)
        r = subprocess.run([sys.executable, 'ordering_router.py',
                            '--dir', d, '--features', 'base,emb'],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        r = subprocess.run([sys.executable, 'screen_shortlist.py', '--dir', d,
                            '--ids', osp.join(d, 'ids.csv'),
                            '--proba', osp.join(d, 'router_test_proba.txt')],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        assert 'all structural checks passed' in r.stdout, r.stdout[-2500:]
        assert 'true-ordering router' in r.stdout
    check('the scorer runs end to end and its structural checks pass',
          t_scorer_runs_and_asserts)


def main():
    here = osp.dirname(osp.abspath(__file__))
    sys.path.insert(0, here)
    os.chdir(here)
    import magsieve_selector as SEL
    import ordering_router as RT
    print('Selector offline test suite')
    print(f'  numpy {np.__version__}')
    tier_rules(SEL)
    tier_table(SEL)
    tier_router(RT)
    tier_end_to_end(SEL, RT)
    print('\n' + '=' * 72)
    print(f'{len(PASS)} passed, {len(FAIL)} failed')
    for n in FAIL:
        print(f'  FAILED : {n}')
    print('=' * 72)
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
