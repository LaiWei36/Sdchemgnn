#!/usr/bin/env python
"""Offline test suite for the proposer, the diagnostics scorer and the decoration contract.

Exercises `propose_spin_configurations.py` and `diagnostics_score.py` without a GPU,
without the dataset, and without the cluster, and — where torch is importable
— exercises the seed-injection contract against the SHIPPED decoration path
in `spin_decoration.decorate_batch`.

Runs in two tiers so it is useful in both places:

    tier 1  numpy only. The proposer, the scorer, the aggregation statistics
            and the screening arithmetic. Runs anywhere.
    tier 2  torch. The injection round trip: what `score_with_signs` writes
            into a batch is what the model reads out of it. Skipped with a
            printed notice when torch or the shipped modules are unavailable.

No pytest dependency — this has to run on a login node with whatever is
loaded. Exit status is 0 only if every check passed.

    python test_proposer_offline.py [--code DIR]
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
import os
import os.path as osp
import sys
import tempfile
import traceback

import numpy as np

PASS, FAIL, SKIP = [], [], []


def check(name, fn):
    try:
        fn()
    except AssertionError as exc:
        FAIL.append((name, str(exc) or 'assertion failed'))
        print(f'  FAIL  {name}\n        {exc}')
    except Exception:
        FAIL.append((name, traceback.format_exc(limit=2)))
        print(f'  FAIL  {name}\n{traceback.format_exc(limit=3)}')
    else:
        PASS.append(name)
        print(f'  ok    {name}')


def skip(name, why):
    SKIP.append((name, why))
    print(f'  skip  {name}   ({why})')


def near(a, b, tol=1e-6):
    assert abs(float(a) - float(b)) <= tol, f'{a} != {b} (tol {tol})'


# ══════════════════════════════════════════════════════════════════════════════
#  fixtures
# ══════════════════════════════════════════════════════════════════════════════
def chain_edges(n):
    e = [[], []]
    for i in range(n - 1):
        e[0] += [i, i + 1]
        e[1] += [i + 1, i]
    return np.array(e)


def ring_edges(n):
    e = [[], []]
    for i in range(n):
        j = (i + 1) % n
        e[0] += [i, j]
        e[1] += [j, i]
    return np.array(e)


# ══════════════════════════════════════════════════════════════════════════════
#  tier 1a — the proposer
# ══════════════════════════════════════════════════════════════════════════════
def tier1_proposer(PROP):
    print('\ntier 1a — proposer')

    def by_name(cands):
        return {c['name']: c for c in cands}

    def t_nonmagnetic_is_nm_only():
        c = PROP.propose(['Na', 'Cl'], chain_edges(2))
        assert len(c) == 1, f'expected 1 candidate, got {[x["name"] for x in c]}'
        assert c[0]['name'] == 'nm'
        assert not c[0]['sign'].any(), 'the nm hypothesis must be all zeros'
    check('a non-magnetic composition yields nm and nothing else',
          t_nonmagnetic_is_nm_only)

    def t_fm_always_present():
        c = by_name(PROP.propose(['Fe', 'O', 'Fe', 'O'], chain_edges(4)))
        assert 'fm' in c
        assert c['fm']['net_formal'] == 10.0, c['fm']['net_formal']  # 2 x Fe(5)
        assert c['fm']['n_down'] == 0
    check('fm is proposed whenever magnetic sites exist, with the formal net',
          t_fm_always_present)

    def t_superexchange_is_what_finds_the_oxide():
        # Mn-O-Mn-O-...: the Mn sites are never nearest neighbours, so a plain
        # magnetic subgraph is empty and every AFM candidate disappears. This
        # is the physics bug the proposer was written around; if it regresses,
        # the AFM family silently vanishes on most oxides.
        sp = ['Mn', 'O'] * 4
        ei = chain_edges(8)
        mag_idx = np.flatnonzero(PROP.magnetic_mask(sp))
        plain = PROP._magnetic_adjacency(ei, mag_idx, len(sp), superexchange=False)
        supex = PROP._magnetic_adjacency(ei, mag_idx, len(sp), superexchange=True)
        assert sum(len(a) for a in plain) == 0, 'the direct subgraph should be empty'
        assert sum(len(a) for a in supex) > 0, 'superexchange found no coupling'
    check('superexchange adjacency is what makes an oxide chain connected',
          t_superexchange_is_what_finds_the_oxide)

    def t_bipartite_afm_alternates():
        sp = ['Mn', 'O'] * 4
        c = by_name(PROP.propose(sp, chain_edges(8)))
        assert 'afm_bipartite' in c, list(c)
        a = c['afm_bipartite']
        assert a['exact'], 'a chain is bipartite; exact must be True'
        s = a['sign'][[0, 2, 4, 6]]
        assert list(s) == [1, -1, 1, -1] or list(s) == [-1, 1, -1, 1], list(s)
        near(a['net_formal'], 0.0)
    check('the AFM candidate alternates on a bipartite chain and nets zero',
          t_bipartite_afm_alternates)

    def t_frustration_is_flagged_not_hidden():
        c = by_name(PROP.propose(['Mn'] * 3, ring_edges(3)))
        assert 'afm_bipartite' in c
        assert not c['afm_bipartite']['exact'], \
            'an odd cycle is frustrated and must be reported approximate'
    check('a frustrated triangle is flagged approximate rather than silently wrong',
          t_frustration_is_flagged_not_hidden)

    def t_even_ring_is_exact():
        c = by_name(PROP.propose(['Mn'] * 4, ring_edges(4)))
        assert c['afm_bipartite']['exact']
        near(c['afm_bipartite']['net_formal'], 0.0)
    check('an even ring is bipartite and exact', t_even_ring_is_exact)

    def t_species_split_makes_a_ferrimagnet():
        # Gd(7) up, Fe(5) down -> |net| = 2 x 7 - 2 x 5 = 4. The only family
        # that can produce a LARGE net moment from opposed sublattices, which
        # is the population every shortlist false positive comes from.
        c = PROP.propose(['Gd', 'Fe', 'Gd', 'Fe'], ring_edges(4))
        nets = sorted({abs(x['net_formal']) for x in c})
        assert 4.0 in nets, nets
    check('a species split reproduces the ferrimagnetic net exactly',
          t_species_split_makes_a_ferrimagnet)

    def t_component_flip_when_disconnected():
        c = by_name(PROP.propose(['Fe'] * 4,
                                 np.array([[0, 1, 2, 3], [1, 0, 3, 2]])))
        assert any(k.startswith('component_flip') for k in c), list(c)
    check('independent components get their free relative sign as a candidate',
          t_component_flip_when_disconnected)

    def t_no_sign_on_a_nonmagnetic_site():
        for sp, ei in ((['Mn', 'O'] * 4, chain_edges(8)),
                       (['Gd', 'Fe', 'O', 'O'], ring_edges(4)),
                       (['Fe', 'Fe', 'Cl', 'Cl'], ring_edges(4))):
            mag = PROP.magnetic_mask(sp)
            for c in PROP.propose(sp, ei):
                assert not np.any(c['sign'][~mag]), \
                    f'{c["name"]} put a sign on a non-magnetic site of {sp}'
    check('no candidate ever puts a sign on a non-magnetic site',
          t_no_sign_on_a_nonmagnetic_site)

    def t_dedup_up_to_global_flip():
        # s and -s are one physical hypothesis under SD-ChemGNN's gauge
        # co-flip; if the dedup key ignored that, half the K budget would be
        # spent re-scoring arrangements the model cannot tell apart.
        s = np.array([1, 0, -1, 0], dtype=np.int8)
        assert PROP.canonical(s).tobytes() == PROP.canonical(-s).tobytes()
        for sp, ei in ((['Mn', 'O'] * 4, chain_edges(8)),
                       (['Gd', 'Fe', 'Gd', 'Fe'], ring_edges(4)),
                       (['Fe'] * 4, ring_edges(4))):
            keys = [PROP.canonical(c['sign']).tobytes()
                    for c in PROP.propose(sp, ei, k_max=32)]
            assert len(keys) == len(set(keys)), f'duplicate candidates for {sp}'
    check('candidates are deduplicated up to a global flip',
          t_dedup_up_to_global_flip)

    def t_k_max_respected_and_shapes_sound():
        sp = ['Gd', 'Fe', 'Mn', 'Ni', 'Co', 'O', 'O', 'O']
        for k in (1, 2, 3, 8, 16):
            c = PROP.propose(sp, ring_edges(8), k_max=k)
            assert len(c) <= k, f'k_max={k} returned {len(c)}'
            for x in c:
                assert x['sign'].shape == (len(sp),), x['sign'].shape
                assert x['sign'].dtype == np.int8, x['sign'].dtype
                assert set(np.unique(x['sign'])) <= {-1, 0, 1}
    check('k_max is respected and every sign vector is a well-formed int8 row',
          t_k_max_respected_and_shapes_sound)

    def t_determinism():
        sp = ['Mn', 'O', 'Fe', 'O', 'Mn', 'O']
        ei = chain_edges(6)
        a = [c['sign'].tobytes() for c in PROP.propose(sp, ei, k_max=8)]
        b = [c['sign'].tobytes() for c in PROP.propose(sp, ei, k_max=8)]
        assert a == b, 'the proposer is not deterministic'
    check('the proposer is deterministic', t_determinism)

    def t_no_edges_degrades_gracefully():
        c = PROP.propose(['Gd', 'Fe', 'Gd', 'Fe'], None, k_max=8)
        names = {x['name'] for x in c}
        assert 'nm' in names and 'fm' in names
        assert not any(n.startswith('afm') for n in names), \
            'without edges there is no neighbour graph to bipartition'
        assert any(n.startswith('species_split') for n in names), \
            'composition-only candidates must survive'
    check('without an edge index the composition-only families still work',
          t_no_edges_degrades_gracefully)

    def t_single_magnetic_site():
        c = PROP.propose(['Fe', 'O'], chain_edges(2), k_max=8)
        names = [x['name'] for x in c]
        assert names[:2] == ['nm', 'fm'], names
        assert len(c) <= 3, names          # nothing to split or bipartition
    check('a lone magnetic site yields nm and fm and little else',
          t_single_magnetic_site)


# ══════════════════════════════════════════════════════════════════════════════
#  tier 1b — the scorer
# ══════════════════════════════════════════════════════════════════════════════
def tier1_scorer(SC):
    print('\ntier 1b — scorer primitives')

    def t_auc_extremes():
        near(SC.auc([1, 2, 3], [0, 0, 0]), 1.0)
        near(SC.auc([0, 0, 0], [1, 2, 3]), 0.0)
        near(SC.auc([1, 1], [1, 1]), 0.5)          # all ties
        near(SC.auc([2, 0], [1, 1]), 0.5)          # symmetric
    check('AUC is 1 / 0 / 0.5 on separated, inverted and tied populations',
          t_auc_extremes)

    def t_auc_ties_are_half():
        # four pairs: (1,1) ties = 0.5, (1,0), (2,1), (2,0) win = 3 -> 3.5/4
        near(SC.auc([1, 2], [1, 0]), 0.875)
        # two pairs, one tie one win -> 1.5/2
        near(SC.auc([1, 2], [1]), 0.75)
    check('AUC scores a tied pair at one half', t_auc_ties_are_half)

    def t_auc_ignores_nan():
        near(SC.auc([1, 2, np.nan], [0, 0]), 1.0)
    check('AUC drops non-finite entries rather than propagating NaN',
          t_auc_ignores_nan)

    def t_ceiling_is_the_formal_maximum():
        near(SC.ceiling('Fe2O3'), 10.0)            # 2 x Fe(5)
        near(SC.ceiling('Mn2Fe'), 15.0)            # 2 x Mn(5) + Fe(5)
        near(SC.ceiling('Gd'), 7.0)
        near(SC.ceiling('NaCl'), 0.5)              # no magnetic species
    check('the physics ceiling is the spin-only formal maximum per f.u.',
          t_ceiling_is_the_formal_maximum)

    def t_group_multipliers_are_applied():
        # 3,700 of the 15,377 test formulas carry a group multiplier. Dropping
        # it undercounts the cell by up to a factor of nine and takes the
        # ceiling clamp — and the per-atom normalization §4.8 compares against
        # the literature — down with it. The first version of this parser
        # multiplied only the last element of the group.
        assert SC._parse_formula('Ba2CaAl2(Fe4O7)2') == \
            {'Ba': 2.0, 'Ca': 1.0, 'Al': 2.0, 'Fe': 8.0, 'O': 14.0}
        assert SC._parse_formula('Ni(C2N3)3') == {'Ni': 1.0, 'C': 6.0, 'N': 9.0}
        near(SC.ceiling('Ba2CaAl2(Fe4O7)2'), 40.0)   # 8 x Fe(5)
        near(SC.ceiling('MgCo6(OF)6'), 24.0)         # 6 x Co(4)
        near(SC.ceiling('Fe(FeO2)2'), 15.0)          # 3 x Fe(5), nested count
    check('a parenthesised group multiplier is applied to the whole group',
          t_group_multipliers_are_applied)

    def t_screening_arithmetic():
        # 1000 rows: the top 1000 IS everything, so P@1000 is just the
        # positive rate and the false-positive count the negative count — a case where the
        # answer is arithmetic rather than a judgement call.
        y = np.concatenate([np.full(300, 6.0), np.zeros(700)])
        r = SC.screening('exact predictor', y.copy(), y)
        near(r['mae'], 0.0)
        near(r['p1000'], 0.3)
        assert r['fp1k'] == 700, r['fp1k']
        near(r['recall_money'], 1.0)
    check('the screening line reproduces P@1000, false positives and recall by hand',
          t_screening_arithmetic)

    def t_demotion_cannot_change_the_prediction():
        y = np.concatenate([np.full(50, 6.0), np.zeros(50)])
        pred = y.copy()
        a = SC.screening('ranked by prediction', pred, y)
        b = SC.screening('ranked by noise', pred, y,
                         rank=np.random.default_rng(0).normal(size=100))
        near(a['mae'], b['mae'])          # ranking never touches the error
        assert a['p1000'] == b['p1000'], 'top-100 of 100 rows must be identical'
    check('a ranking change moves the shortlist, never the prediction error',
          t_demotion_cannot_change_the_prediction)


# ══════════════════════════════════════════════════════════════════════════════
#  tier 1c — the scorer end to end, on a fixture whose answer is known
# ══════════════════════════════════════════════════════════════════════════════
# Three response profiles over the arrangement families, one per population.
# Deliberately more rows than the 1000-row shortlist, and more real magnets
# than fit in it, so P@1000 and the false-positive count are real selections rather than the
# arithmetic identity they collapse to when n <= TOPK.
N_REAL, N_COMP, N_NON = 1200, 400, 2400
PROFILES = {
    #                       nm    fm    afm   ferri     truth
    'real': ([0.20, 6.00, 5.60, 5.80], 6.0),   # holds up under every arrangement
    'comp': ([0.10, 6.00, 0.05, 0.20], 0.1),   # collapses under the AFM one
    'non':  ([0.00, 0.10, 0.05, 0.02], 0.0),   # nothing there at all
}
CAND_NAMES = ['nm', 'fm', 'afm_bipartite', 'species_split_1']


def write_fixture(d, rng, informative=True):
    """A fixture whose answer is known by construction.

    `informative=True` couples each structure's arrangement response to its
    truth class, which is the hypothesis the diagnostics test.

    `informative=False` is the ANTI-TEST: every structure draws its response
    profile at random, independent of its truth. The marginal distribution of
    predictions is unchanged and the three truth populations are unchanged —
    only the coupling between them is gone. A statistic that still separates
    on that fixture is reading the populations, not the arrangement response,
    and would report a false success on the real data.

    Note the earlier version of this anti-test permuted the values WITHIN a
    structure, which does nothing: min and max are invariant under a
    permutation, so the collapse ratio survived it untouched and the test
    passed a broken scorer. It is written this way because that version
    failed to fail.
    """
    y = np.concatenate([np.full(N_REAL, PROFILES['real'][1]),
                        np.full(N_COMP, PROFILES['comp'][1]),
                        np.full(N_NON, PROFILES['non'][1])])
    kinds = ['real'] * N_REAL + ['comp'] * N_COMP + ['non'] * N_NON
    if not informative:
        kinds = list(rng.permutation(kinds))
    n = len(y)
    mid = [f'mp-{i:05d}' for i in range(n)]
    with open(osp.join(d, 'phase0_test_ids.csv'), 'w') as fh:
        fh.write('row,material_id,truth_capped\n')
        for i, (m, t) in enumerate(zip(mid, y)):
            fh.write(f'{i},{m},{t:.6f}\n')
    with open(osp.join(d, 'phase0c_proposed.txt'), 'w') as fh:
        fh.write('# row material_id cand_idx cand_name exact net_formal pred '
                 'sum_abs_m sum_m resid_fu z_fu sc_frac\n')
        for i in range(n):
            vals = np.asarray(PROFILES[kinds[i]][0]) + rng.normal(0, 0.01, 4)
            for k, (nm_, v) in enumerate(zip(CAND_NAMES, vals)):
                fh.write('%d %s %d %s 1 0.0 %.6f %.6f %.6f 0.0 1.0 1.0\n'
                         % (i, mid[i], k, nm_, v, abs(v) * 2, v))
    return y


def tier1_scorer_e2e(SC):
    print('\ntier 1c — scorer end to end, against a construction-known answer')

    def build(seed, informative=True):
        """Returns (dir handle, y, stats) using the SCORER'S OWN aggregation —
        not a reimplementation, so a change to aggregate_candidates is caught
        here rather than on the cluster."""
        d = tempfile.mkdtemp(prefix='phase0fix-')
        write_fixture(d, np.random.default_rng(seed), informative)
        mid, y, _ = SC.load_index(d)
        return d, y, SC.aggregate_candidates(osp.join(d, 'phase0c_proposed.txt'),
                                             len(mid))

    def t_nm_is_held_out_of_the_ensemble():
        # The nm hypothesis predicts ~0 for everything, so including it would
        # drive min/max to ~0 for every structure and flatten the collapse
        # ratio into a constant. This is the flaw the fixture exposed.
        _, y, st = build(1)
        assert np.isfinite(st['nm']).all(), 'nm must be recorded separately'
        assert st['nk'].max() == len(CAND_NAMES) - 1, \
            f'expected the nm candidate held out, got nk={st["nk"].max()}'
        assert np.nanmedian(st['collapse'][y >= SC.MONEY]) > 0.8, \
            'with nm held out, a real magnet must hold up across arrangements'
    check('the nm hypothesis is held out of the ensemble statistics',
          t_nm_is_held_out_of_the_ensemble)

    def t_informative_fixture_separates():
        _, y, st = build(1)
        comp = (st['max'] > SC.POS) & (y <= SC.POS)
        real = y >= SC.MONEY
        assert int(comp.sum()) == N_COMP, int(comp.sum())
        assert int(real.sum()) == N_REAL, int(real.sum())
        a = SC.auc(st['collapse'][real], st['collapse'][comp])
        assert a > 0.95, f'collapse ratio should separate cleanly, got {a:.3f}'
    check('on the informative fixture the collapse ratio separates (AUC > 0.95)',
          t_informative_fixture_separates)

    def t_anti_test_finds_nothing():
        _, y, st = build(2, informative=False)
        comp = (st['max'] > SC.POS) & (y <= SC.POS)
        real = y >= SC.MONEY
        a = SC.auc(st['collapse'][real], st['collapse'][comp])
        assert 0.35 < a < 0.65, (
            f'the anti-test fixture has the same marginals and no coupling '
            f'between arrangement response and truth; an AUC of {a:.3f} means '
            f'the statistic is reading the populations, not the ensemble')
    check('the anti-test fixture, same marginals without the coupling, scores ~0.5',
          t_anti_test_finds_nothing)

    def t_min_beats_fm_on_false_positives():
        _, y, st = build(3)
        fm = SC.screening('fm alone', np.nan_to_num(st['fm']), y)
        mn = SC.screening('ensemble min', np.nan_to_num(st['min']), y)
        assert fm['fp1k'] > mn['fp1k'], (
            f'aggregation must remove the compensated false positives the FM '
            f'hypothesis admits: fm {fm["fp1k"]} vs min {mn["fp1k"]}')
        assert mn['fp1k'] == 0, (
            f'{N_REAL} real magnets exist for a {SC.TOPK}-row shortlist, so a '
            f'working conservative score should fill it with them; got '
            f'{mn["fp1k"]} false positives')
    check('taking the ensemble minimum removes every compensated false positive',
          t_min_beats_fm_on_false_positives)

    def t_demotion_selects_the_right_rows():
        _, y, st = build(4)
        dem = np.nan_to_num(st['collapse'], nan=1.0) < 0.25
        comp = (st['max'] > SC.POS) & (y <= SC.POS)
        real = y >= SC.MONEY
        assert bool(dem[comp].all()), (
            f'{int((~dem[comp]).sum())} compensated rows escaped demotion')
        assert not bool(dem[real].any()), (
            f'{int(dem[real].sum())} real magnets were demoted — this fixture '
            f'has none that collapse, so any is a false negative the rule '
            f'invented')
        # The non-magnetic population collapses too and is demoted with them.
        # That is harmless (it was never near the shortlist) but it is worth
        # pinning: the rule fires on "can cancel", not on "is a false positive",
        # and on real data most of what it demotes will be uninteresting rows.
        assert int(dem.sum()) > N_COMP, int(dem.sum())
    check('collapse demotion catches every compensated row and no real magnet',
          t_demotion_selects_the_right_rows)

    def t_demotion_repairs_the_no_aggregation_baseline():
        # The realistic case: the FM hypothesis alone admits compensated rows
        # into the shortlist, and demoting on the collapse ratio removes them
        # WITHOUT changing a single prediction — the trade that must
        # be measured rather than argued about.
        _, y, st = build(5)
        fm = np.nan_to_num(st['fm'])
        dem = np.nan_to_num(st['collapse'], nan=1.0) < 0.25
        base = SC.screening('fm alone', fm, y)
        r = SC.screening('fm + collapse demotion', fm, y,
                         rank=np.where(dem, -1e9, fm))
        assert r['fp1k'] < base['fp1k'], (base['fp1k'], r['fp1k'])
        assert r['recall_money'] >= base['recall_money'] - 1e-9, (
            'no real magnet in this fixture collapses, so demotion must cost '
            'nothing in the high-moment band')
        assert abs(r['mae'] - base['mae']) < 1e-12, \
            'demotion is a ranking operation and must not move the error'
    check('demotion repairs the no-aggregation baseline at no high-moment cost',
          t_demotion_repairs_the_no_aggregation_baseline)


# ══════════════════════════════════════════════════════════════════════════════
#  tier 2 — the injection contract, against the shipped decoration path
# ══════════════════════════════════════════════════════════════════════════════
class FakeBatch:
    """The subset of a PyG Batch that decorate_batch and self_consistency
    touch. Deliberately not a PyG object: the point is to pin the CONTRACT
    (which attributes are read, which are written) rather than to re-test PyG.
    """
    def __init__(self, x, batch, sign, fmag, elig):
        self.x, self.batch = x, batch
        self.seed_sign, self.seed_fmag, self.seed_elig = sign, fmag, elig
        self.num_graphs = int(batch.max()) + 1


def tier2_injection(S, torch):
    print('\ntier 2 — seed injection against the shipped decoration path')
    base = 6
    dcfg = {'base_channels': base, 'p_undecorate': 0.0, 'p_site_mask': 0.0,
            'p_promote': 0.0, 'fmag_scale': 7.0, 'coflip': False}

    def mk(signs, elig):
        n = len(signs)
        return FakeBatch(torch.zeros(n, base),
                         torch.zeros(n, dtype=torch.long),
                         torch.tensor(signs, dtype=torch.float32),
                         torch.tensor([5.0] * n),
                         torch.tensor(elig, dtype=torch.int8))

    def t_oracle_writes_the_injected_sign():
        d = S.decorate_batch(mk([1.0, -1.0, 0.0, 1.0], [1]), 'oracle', dcfg)
        assert d.x.size(1) == base + S.SEED_CHANNELS, d.x.size(1)
        got = d.x[:, base].tolist()
        assert got == [1.0, -1.0, 0.0, 1.0], got
        near(d.x[0, base + 1], 5.0 / 7.0)
        assert d.x[:, base + 2].tolist() == [1.0] * 4, 'flag must be set'
    check('an injected sign vector reaches the sign channel unchanged',
          t_oracle_writes_the_injected_sign)

    def t_inelig_is_the_undecorated_fallback():
        # This is the contract score_with_signs relies on for structures the
        # proposer could not seed: elig = 0 must give three zero channels, the
        # same input the undecorated mode produces — not a silent FM guess.
        d = S.decorate_batch(mk([1.0, -1.0, 1.0, -1.0], [0]), 'oracle', dcfg)
        assert d.x[:, base:].abs().sum().item() == 0.0, d.x[:, base:]
        u = S.decorate_batch(mk([1.0, -1.0, 1.0, -1.0], [0]), 'undecorated', dcfg)
        assert torch.equal(d.x, u.x), 'elig=0 must equal the undecorated input'
    check('an ineligible graph decorates to exactly the undecorated input',
          t_inelig_is_the_undecorated_fallback)

    def t_double_decoration_is_refused():
        d = S.decorate_batch(mk([1.0, 1.0], [1]), 'oracle', dcfg)
        try:
            S.decorate_batch(d, 'oracle', dcfg)
        except RuntimeError:
            return
        raise AssertionError('decorating an already-decorated batch must raise '
                             '— this is the guard that catches a stale cache')
    check('decorating twice raises rather than silently widening x',
          t_double_decoration_is_refused)

    def t_self_consistency_reads_the_seed():
        d = S.decorate_batch(mk([1.0, -1.0, 1.0, -1.0], [1]), 'oracle', dcfg)
        agree = torch.tensor([2.0, -2.0, 2.0, -2.0])
        opp = -agree
        f1, _ = S.self_consistency(agree, d, tiny=0.05)
        f0, _ = S.self_consistency(opp, d, tiny=0.05)
        near(f1[0], 1.0)
        near(f0[0], 0.0)
    check('self-consistency is 1 when the output reproduces the seed and 0 when '
          'it inverts it', t_self_consistency_reads_the_seed)

    def t_non_committal_sites_are_excluded():
        d = S.decorate_batch(mk([1.0, -1.0], [1]), 'oracle', dcfg)
        f, ncom = S.self_consistency(torch.tensor([0.001, -0.001]), d, tiny=0.05)
        assert ncom == 0, ncom
        assert bool(torch.isnan(f[0])), \
            'a graph with no committal site must report NaN, not a false 0 or 1'
    check('sites below the commitment threshold are excluded, not counted as '
          'failures', t_non_committal_sites_are_excluded)

    def t_shuffled_preserves_the_multiset():
        g = torch.Generator(); g.manual_seed(1729)
        d = S.decorate_batch(mk([1.0, -1.0, 1.0, -1.0], [1]), 'shuffled', dcfg, g)
        got = sorted(d.x[:, base].tolist())
        assert got == [-1.0, -1.0, 1.0, 1.0], got
    check('the shuffled probe permutes the arrangement and keeps the multiset',
          t_shuffled_preserves_the_multiset)

    def t_fm_formal_seeds_every_magnetic_site():
        b = mk([0.0, 0.0, 0.0, 0.0], [0])
        b.seed_fmag = torch.tensor([5.0, 0.0, 5.0, 0.0])
        d = S.decorate_batch(b, 'fm_formal', dcfg)
        assert d.x[:, base].tolist() == [1.0, 0.0, 1.0, 0.0], d.x[:, base]
    check('the fm_formal fallback seeds magnetic species and only those',
          t_fm_formal_seeds_every_magnetic_site)


class FakeCollated:
    """Three graphs of 2, 3 and 2 sites collated the way PyG lays them out:
    nodes contiguous per graph, edge_index in the batch's global numbering."""
    def __init__(self, torch):
        self.batch = torch.tensor([0, 0, 1, 1, 1, 2, 2])
        self.ptr = torch.tensor([0, 2, 5, 7])
        #        g0: 0-1        g1: 2-3, 3-4        g2: 5-6
        self.edge_index = torch.tensor([[0, 1, 2, 3, 3, 4, 5, 6],
                                        [1, 0, 3, 2, 4, 3, 6, 5]])


def tier2_batch_geometry(torch):
    print('\ntier 2b — batch geometry helpers')
    try:
        import score_configurations_with_sdchemgnn as D
    except Exception as exc:
        skip('batch geometry helpers', f'{type(exc).__name__}: {exc}')
        return
    b = FakeCollated(torch)

    def t_ptr_from_the_batch_vector():
        assert D._node_ptr(b) == [0, 2, 5, 7]
        nob = FakeCollated(torch)
        del nob.ptr                     # the bincount fallback path
        assert D._node_ptr(nob) == [0, 2, 5, 7]
    check('node offsets come out right with and without a ptr attribute',
          t_ptr_from_the_batch_vector)

    def t_edges_are_split_and_localised():
        ptr, edges = D._split_edges(b)
        assert ptr == [0, 2, 5, 7]
        assert edges[0].tolist() == [[0, 1], [1, 0]], edges[0].tolist()
        # graph 1 is sites 2,3,4 globally -> 0,1,2 locally
        assert sorted(map(tuple, edges[1].T.tolist())) == \
            [(0, 1), (1, 0), (1, 2), (2, 1)], edges[1].T.tolist()
        assert edges[2].tolist() == [[0, 1], [1, 0]], edges[2].tolist()
        for e in edges:
            assert e.min() >= 0, 'a local index went negative'
    check('edges are split per graph and reindexed to local site numbers',
          t_edges_are_split_and_localised)

    def t_isolated_graph_yields_no_edges():
        c = FakeCollated(torch)
        c.edge_index = torch.tensor([[0, 1], [1, 0]])      # only graph 0 bonded
        _, edges = D._split_edges(c)
        assert edges[0] is not None and edges[1] is None and edges[2] is None
    check('a graph with no edges of its own reports None, not a wrong slice',
          t_isolated_graph_yields_no_edges)

    def t_empty_edge_index():
        c = FakeCollated(torch)
        c.edge_index = torch.zeros((2, 0), dtype=torch.long)
        ptr, edges = D._split_edges(c)
        assert ptr == [0, 2, 5, 7] and edges == [None, None, None]
    check('an empty edge index degrades to no edges rather than raising',
          t_empty_edge_index)


# ══════════════════════════════════════════════════════════════════════════════
def load_shipped(code_dir):
    """Import spin_decoration. In the full repository this just works; when only
    this folder is present, `site_moment_labels` is stubbed — spin_decoration
    imports one name from it and uses that name only inside build_seed_table,
    which tier 2 does not touch.
    """
    sys.path.insert(0, code_dir)
    try:
        import spin_decoration as S
        return S, False
    except ModuleNotFoundError as exc:
        if 'site_moment_labels' not in str(exc):
            raise
    stub = tempfile.mkdtemp(prefix='phase0stub-')
    with open(osp.join(stub, 'site_moment_labels.py'), 'w') as fh:
        fh.write('def build_magmom_labels(*a, **k):\n'
                 '    raise NotImplementedError(\n'
                 '        "test stub: the real site_moment_labels is not in "\n'
                 '        "this folder. Tier 2 exercises decorate_batch and "\n'
                 '        "self_consistency, neither of which calls this.")\n')
    sys.path.insert(0, stub)
    import spin_decoration as S
    return S, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--code', default=osp.dirname(osp.abspath(__file__)),
                    help='directory holding propose_spin_configurations.py etc.')
    ap.add_argument('--shipped', default=None,
                    help='directory holding spin_decoration.py (tier 2)')
    a = ap.parse_args()
    sys.path.insert(0, a.code)

    print('Proposer and diagnostics offline test suite')
    print(f'  code    {a.code}')
    print(f'  numpy   {np.__version__}')

    import propose_spin_configurations as PROP
    tier1_proposer(PROP)
    # diagnostics_score is the laptop-side scorer and does not have to be present
    # alongside the scoring job. Skip its tiers rather than aborting: this
    # suite can run as a preflight, and killing a job over a missing offline
    # convenience file is exactly the wrong trade.
    try:
        import diagnostics_score as SC
    except ImportError as exc:
        skip('tier 1b (scorer primitives)', f'diagnostics_score not importable: {exc}')
        skip('tier 1c (scorer end to end)', 'same')
    else:
        tier1_scorer(SC)
        tier1_scorer_e2e(SC)

    shipped = a.shipped or os.environ.get('BS_CODE_DIR') or a.code
    try:
        import torch
    except ImportError:
        skip('tier 2 (injection contract)', 'torch not importable')
    else:
        try:
            S, stubbed = load_shipped(shipped)
        except Exception as exc:
            skip('tier 2 (injection contract)',
                 f'spin_decoration not importable from {shipped}: '
                 f'{type(exc).__name__}: {exc}')
        else:
            if stubbed:
                print('\n  (site_moment_labels stubbed — it is not in the '
                      'folder and tier 2 does not use it)')
            print(f'  torch   {torch.__version__}')
            tier2_injection(S, torch)
        tier2_batch_geometry(torch)

    print('\n' + '=' * 72)
    print(f'{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped')
    for n, why in SKIP:
        print(f'  skipped: {n} — {why}')
    for n, why in FAIL:
        print(f'  FAILED : {n}')
    print('=' * 72)
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
