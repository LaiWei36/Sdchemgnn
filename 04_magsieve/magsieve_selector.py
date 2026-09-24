#!/usr/bin/env python
"""MagSieve's demote-only SELECTOR.

Takes the proposer's candidate arrangements, a router that names an ordering
class per structure, and the undecorated prediction, and produces a ranking.
Pure numpy: no torch, no GPU, no dataset. It consumes the artifacts the
scoring pass (score_configurations_with_sdchemgnn.py, part 0d) writes and
nothing else.

THE TWO RULES
-------------
    Rule 1 — the selector may only DEMOTE.  A routed candidate enters the
             ranking score only when its magnitude is SMALLER than the
             undecorated one.
    Rule 2 — the selector never changes a REPORTED number.  The value reported
             for every structure is the undecorated prediction, always.

They are enforced structurally here rather than checked afterwards: `select()`
copies the undecorated vector into `reported` and never writes to it again, and
the ranking score is built with an explicit minimum. `assert_invariants()` is
provided so callers can re-verify, and the offline suite does.

WHY THIS SHAPE (measured, not assumed)
    * With a perfect ordering-class router: P@1000 1.0000, no false positive
      in the top 1,000, and MAE, tail and catastrophic count all exactly the
      unscreened ranking's.
    * Full substitution — reporting the chosen configuration's prediction
      instead — reaches 0.9990 / 1 but degrades MAE from 0.5403 to 0.7966.
    * MAE, tail and catastrophic are the unscreened ranking's BY CONSTRUCTION,
      since no predicted value ever changes.
    * A router that abstains everywhere reproduces the unscreened ranking
      EXACTLY, so the design degrades gracefully to doing nothing.

CORRECTION. An earlier version of this docstring claimed "a router error
cannot manufacture a false positive; it can only cost recall". THAT IS FALSE
and the claim was load-bearing, so it is recorded rather than quietly removed.

Rule 1 stops a row being PROMOTED above where the unscreened ranking put it.
It does not stop one being ADMITTED. The shortlist is a fixed-size top-K, so
demoting a true positive frees a slot and whatever sat at rank K+1 moves in —
its own score unchanged, no rule violated, and it may be a false positive.
Minimal counterexample: scores (9, 8, 7) with truths (9, 8, 0) and a top-2
shortlist; demote row 1 and the shortlist goes from {0,1} with no false
positives to {0,2} with one.

Measured consequence on a real test set: a constant AFM router — which
demotes indiscriminately — roughly doubles the false positives in the top
1,000 (21 to 45 on one run, clamped convention). Over-demotion is a real
failure mode and the two rules do not exclude it.

What the rules DO still guarantee, and these are verified by the offline suite:
    * MAE, tail and catastrophic count are the unscreened ranking's exactly,
      since no predicted value ever changes (rule 2).
    * No individual row is ranked above where the unscreened ranking ranked
      it (rule 1).
    * Abstaining everywhere reproduces the unscreened ranking exactly.
MAE cannot degrade here, but shortlist precision can, so a router must still
be better than nothing rather than merely harmless.

NOT A PIPELINE FILE. Nothing in the trainer imports this.
"""

from __future__ import annotations

import numpy as np

# The four Materials Project ordering classes, in the order the router's
# probability columns are expected to be in.
CLASSES = ('NM', 'FM', 'AFM', 'FiM')
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}
ABSTAIN = -1


def family_of(cand_name):
    """Which ordering class a proposer candidate realises.

    The mapping is one-to-one by construction: the proposer's candidate
    families were built to be exactly the MP ordering classes, which is what
    makes a 4-class router able to name a configuration directly.
    """
    n = str(cand_name)
    if n == 'nm':
        return 'NM'
    if n == 'fm':
        return 'FM'
    if n.startswith('afm_bipartite') or n.startswith('component_flip'):
        return 'AFM'
    if n.startswith('species_split') or n.startswith('orbit_split'):
        return 'FiM'
    return None


class CandidateTable:
    """The (structure, arrangement) table from a scoring-pass dump, indexed for
    lookup by row and ordering class.

    Candidates keep the proposer's priority order within a family, so
    `pred(row, cls)` returns the family's FIRST candidate — the most physically
    motivated one — rather than an arbitrary or a best-of member. Picking the
    best of a family would be a second, hidden selector.
    """

    def __init__(self, n_rows):
        self.n = int(n_rows)
        self._by_cls = [dict() for _ in range(self.n)]
        self._names = [dict() for _ in range(self.n)]
        self.n_pairs = 0

    @classmethod
    def from_file(cls, path, n_rows):
        t = cls(n_rows)
        with open(path) as fh:
            for line in fh:
                if line.startswith('#'):
                    continue
                f = line.split()
                if len(f) < 12:
                    continue
                row, name, pred = int(f[0]), f[3], float(f[6])
                if row >= t.n:
                    continue
                fam = family_of(name)
                t.n_pairs += 1
                if fam is None or fam in t._by_cls[row]:
                    continue                      # keep the first of a family
                t._by_cls[row][fam] = pred
                t._names[row][fam] = name
        return t

    def pred(self, row, cls):
        return self._by_cls[row].get(cls)

    def name(self, row, cls):
        return self._names[row].get(cls, '')

    def coverage(self):
        """How many rows carry a candidate for each class — the ceiling on how
        often the router can be acted on at all."""
        return {c: sum(1 for r in range(self.n) if c in self._by_cls[r])
                for c in CLASSES}


# ══════════════════════════════════════════════════════════════════════════════
#  routers
# ══════════════════════════════════════════════════════════════════════════════
class ProbRouter:
    """Wraps an [N, 4] class-probability matrix. Confidence is the max
    posterior; rows below `abstain_below` abstain."""

    def __init__(self, proba, classes=CLASSES):
        self.p = np.asarray(proba, float)
        if self.p.ndim != 2 or self.p.shape[1] != len(classes):
            raise ValueError(f'expected [N, {len(classes)}], got {self.p.shape}')
        self.classes = tuple(classes)

    def route(self):
        return ([self.classes[i] for i in self.p.argmax(1)], self.p.max(1))


class ConstantRouter:
    """Always names the same class. The control that says how much of any gain
    is routing rather than a global rule — the 'always-min kills true
    FMs, always-max reproduces the FP factory' check, made runnable."""

    def __init__(self, n, cls):
        self.n, self.cls = int(n), cls

    def route(self):
        return ([self.cls] * self.n, np.ones(self.n))


class OracleRouter:
    """Routes from the TRUE ordering class. NOT DEPLOYABLE — it reads labels.
    Present so the ceiling is computed by the same code path as everything
    else rather than by a one-off script."""

    def __init__(self, true_cls):
        self.t = list(true_cls)

    def route(self):
        return (list(self.t), np.ones(len(self.t)))


# ══════════════════════════════════════════════════════════════════════════════
#  the selector
# ══════════════════════════════════════════════════════════════════════════════
# What to do when the router names a class the proposer produced no candidate
# for. This is a real decision and it moved the measured result, so it is an
# explicit, named policy rather than a silent `dict.get(a, b)` chain.
#
#   ABSTAIN     do nothing — the conservative default. A class that is not on
#               the menu for this structure cannot be acted on.
#   PHYSICAL    substitute the nearest available arrangement, on physical
#               grounds: a ferrimagnet with no species split (one magnetic
#               species) still has opposed sublattices available through the
#               bipartition; an antiferromagnet with no magnetic subgraph has
#               no coupled magnetic sites at all, which is the non-magnetic
#               hypothesis.
#
# On one real test set the difference is 21 vs 17 false positives in the top 1,000 under the
# measured router and 1 vs 0 under a perfect one, so it is not cosmetic.
FALLBACK_ABSTAIN = {}
FALLBACK_PHYSICAL = {'FiM': ('AFM',), 'AFM': ('NM',)}


def select(undec, table, router, *, abstain_below=0.0, demote_only=True,
           fallback=None):
    """Apply a router to the candidate table and return the selection.

    Returns a dict with
        reported   [N] float  — the undecorated prediction, UNCHANGED (rule 2)
        rank       [N] float  — the ranking score, min(|undec|, |chosen|) (rule 1)
        cls        [N] object — the class acted on, or '' where abstained
        acted      [N] bool   — a candidate actually changed the ranking
        abstained  [N] bool   — no class, low confidence, or no such candidate

    `demote_only=False` is available for the ablation that shows rule 1 is
    load-bearing; it is not a deployment option.
    """
    undec = np.asarray(undec, float)
    n = undec.size
    if table.n != n:
        raise ValueError(f'table has {table.n} rows, undec has {n}')
    reported = undec.copy()                       # RULE 2: written once, here
    rank = np.abs(undec).copy()
    cls_out = np.full(n, '', dtype=object)
    acted = np.zeros(n, bool)
    abstained = np.zeros(n, bool)

    routed, conf = router.route()
    conf = np.asarray(conf, float)
    for i in range(n):
        c = routed[i]
        if c is None or c == ABSTAIN or c == '' or conf[i] < abstain_below:
            abstained[i] = True
            continue
        v = table.pred(i, c)
        if v is None:
            for alt in (fallback or {}).get(c, ()):     # explicit, not silent
                v = table.pred(i, alt)
                if v is not None:
                    c = alt
                    break
        if v is None:
            abstained[i] = True
            continue
        a = abs(float(v))
        if demote_only and a >= rank[i]:
            cls_out[i] = c                        # routed, but no demotion
            continue
        rank[i] = a                               # RULE 1: strictly smaller
        cls_out[i] = c
        acted[i] = True
    return {'reported': reported, 'rank': rank, 'cls': cls_out,
            'acted': acted, 'abstained': abstained}


def assert_invariants(sel, undec, demote_only=True):
    """Re-verify the two rules on a result. Cheap, and the offline suite and
    the scorer both call it — a guarantee that is only argued for is not a
    guarantee."""
    undec = np.asarray(undec, float)
    if not np.array_equal(sel['reported'], undec):
        raise AssertionError('RULE 2 VIOLATED: a reported prediction changed. '
                             'MAE, tail and catastrophic count are supposed to '
                             'be the unscreened ranking\'s by construction.')
    if demote_only and np.any(sel['rank'] > np.abs(undec) + 1e-12):
        bad = int(np.sum(sel['rank'] > np.abs(undec) + 1e-12))
        raise AssertionError(f'RULE 1 VIOLATED: {bad} rows were promoted. A '
                             f'promotion can manufacture a false positive, '
                             f'which is the one failure this design excludes.')
    if np.any(sel['acted'] & sel['abstained']):
        raise AssertionError('a row is both acted on and abstained')
    return True


def summarise(sel):
    n = sel['reported'].size
    from collections import Counter
    return {'n': n,
            'acted': int(sel['acted'].sum()),
            'abstained': int(sel['abstained'].sum()),
            'routed_no_demotion': int(n - sel['acted'].sum()
                                      - sel['abstained'].sum()),
            'by_class': dict(Counter(c for c in sel['cls'] if c))}
