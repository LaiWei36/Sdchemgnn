#!/usr/bin/env python
"""MagSieve's PROPOSER: candidate spin configurations from the structure alone.

Generates candidate magnetic arrangements for a structure at INFERENCE time,
where no labels exist. Its output is the same object the training-time seed
derivation produces: a per-site sign vector in {-1, 0, +1}, which
`spin_decoration.decorate_batch` turns into the three decoration channels.

    training  :  labels  ->  spin_decoration.derive_seeds   ->  sign vector
    inference :  structure ->  propose_spin_configurations.propose   ->  sign vectors (K of them)

WHY THIS DOES NOT USE pymatgen's MagneticStructureEnumerator
------------------------------------------------------------
pymatgen's `MagneticStructureEnumerator` was the obvious candidate. It cannot
be used here: it delegates to `EnumlibAdaptor`, which requires the
compiled enumlib binaries (`enum.x` / `multienum.x` and `makestr.x` /
`makeStr.py`) on PATH. They are not installed in the analysis environment and
should not be assumed on the cluster — this was verified, not guessed.

That turns out to be the right thing to discover, because the enumerator also
solves a harder problem than ours. It generates derivative *structures*:
magnetic supercells with new lattices. We are not generating structures. We
already have the graph — whatever cell Materials Project supplied, with the
sites and neighbour list the network consumes — and we need only to assign
signs to the sites that are already in it. Sign assignment over a fixed site
set needs no supercell machinery.

So the proposer works from the crystal graph itself: the species list and the
edge index the model already receives. `pymatgen` is OPTIONAL and used only to
add symmetry-orbit candidates when a `Structure` happens to be available; the
default path has NO new dependency at all.

CANDIDATE FAMILIES (priority order — the cap keeps the earliest)
    nm            every site 0. The non-magnetic hypothesis.
    fm            +1 on every magnetic-species site. The ferromagnetic one.
    afm_bipartite 2-colouring of the magnetic-site subgraph. Antiferromagnetic
                  superexchange favours antiparallel nearest neighbours, so a
                  bipartition of the neighbour graph is the physically natural
                  antiferromagnetic candidate. Exact when the subgraph is
                  bipartite; greedy conflict-minimising when it is frustrated,
                  and the return value says which.
    component_*   when the magnetic subgraph has several connected components
                  their relative sign is free; each independent flip is a
                  distinct arrangement.
    species_*     one magnetic species up, another down. The ferrimagnetic
                  family, and the only one that can produce a large net moment
                  from opposed sublattices.
    orbit_*       symmetry-orbit sign patterns (pymatgen only, optional).

All candidates are deduplicated UP TO A GLOBAL FLIP, because SD-ChemGNN trains
under a gauge co-flip and s and -s are the same physical hypothesis.

NOT A PIPELINE FILE. Nothing here is imported by the trainer. It is consumed
by `score_configurations_with_sdchemgnn.py`.
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
import json
import sys
from collections import deque

import numpy as np

# ── one formal-magnitude table, shared with the training-time seed builder ────
try:
    from spin_decoration import FORMAL_MAGNITUDE
except Exception:                                       # standalone / test use
    try:
        from screen_stack import MAG_MAX as FORMAL_MAGNITUDE   # earlier module, optional
    except Exception:
        FORMAL_MAGNITUDE = {
            'Ti': 2, 'V': 3, 'Cr': 4, 'Mn': 5, 'Fe': 5, 'Co': 4, 'Ni': 2,
            'Cu': 1, 'Nb': 3, 'Mo': 3, 'Tc': 3, 'Ru': 4, 'Rh': 2, 'Pd': 1,
            'W': 2, 'Re': 3, 'Os': 2, 'Ir': 1, 'Ce': 1, 'Pr': 2, 'Nd': 3,
            'Pm': 4, 'Sm': 5, 'Eu': 7, 'Gd': 7, 'Tb': 6, 'Dy': 5, 'Ho': 4,
            'Er': 3, 'Tm': 2, 'Yb': 1, 'U': 3, 'Np': 4, 'Pu': 5,
        }

DEFAULT_K = 8


# ══════════════════════════════════════════════════════════════════════════════
#  site classification
# ══════════════════════════════════════════════════════════════════════════════
def magnetic_mask(species):
    """Boolean mask over sites: True where the species carries a formal moment.

    Same table the seed channels use, so 'magnetic' means exactly what it means
    at training time and in the physics-ceiling machinery.
    """
    return np.array([float(FORMAL_MAGNITUDE.get(str(s), 0.0)) > 0.0
                     for s in species], dtype=bool)


def formal_magnitudes(species):
    """Per-site spin-only formal magnitude in raw μB (0 on non-magnetic sites)."""
    return np.array([float(FORMAL_MAGNITUDE.get(str(s), 0.0))
                     for s in species], dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════════
#  magnetic subgraph
# ══════════════════════════════════════════════════════════════════════════════
def _magnetic_adjacency(edge_index, mag_idx, n_sites, superexchange=True):
    """Adjacency lists over the magnetic sites only, keyed by position in
    `mag_idx`. `edge_index` is the PyG [2, E] array over ALL sites.

    Two magnetic sites are treated as coupled if they are directly bonded OR
    — when `superexchange` is set, which is the default — if they share a
    common non-magnetic neighbour. The second case is the important one and
    the reason this is not a plain subgraph: in most magnetic oxides the
    cations are not nearest neighbours at all, and the dominant interaction is
    superexchange through the bridging anion, M–O–M. Restricting to direct
    magnetic-magnetic bonds leaves the subgraph of a typical oxide empty, and
    with it every antiferromagnetic candidate.
    """
    pos = {int(g): i for i, g in enumerate(mag_idx)}
    adj = [set() for _ in mag_idx]
    if edge_index is None or len(mag_idx) == 0:
        return [sorted(a) for a in adj]
    src = np.asarray(edge_index)[0].tolist()
    dst = np.asarray(edge_index)[1].tolist()

    # direct magnetic-magnetic bonds
    for u, v in zip(src, dst):
        iu, iv = pos.get(int(u)), pos.get(int(v))
        if iu is not None and iv is not None and iu != iv:
            adj[iu].add(iv)
            adj[iv].add(iu)

    # superexchange: magnetic sites sharing a non-magnetic bridge
    if superexchange:
        bridge = [[] for _ in range(int(n_sites))]
        for u, v in zip(src, dst):
            iu = pos.get(int(u))
            if iu is not None and int(v) not in pos:
                bridge[int(v)].append(iu)
        for shared in bridge:
            if len(shared) < 2:
                continue
            uniq = sorted(set(shared))
            for a_i in range(len(uniq)):
                for b_i in range(a_i + 1, len(uniq)):
                    adj[uniq[a_i]].add(uniq[b_i])
                    adj[uniq[b_i]].add(uniq[a_i])
    return [sorted(a) for a in adj]


def _components(adj):
    """Connected components of the magnetic subgraph, as index lists."""
    n = len(adj)
    seen = [False] * n
    out = []
    for s in range(n):
        if seen[s]:
            continue
        comp, q = [], deque([s])
        seen[s] = True
        while q:
            u = q.popleft()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
        out.append(sorted(comp))
    return out


def two_colour(adj, comp):
    """2-colour one component. Returns (colours dict, bipartite flag).

    BFS assigns alternating colours. On an odd cycle the graph is not
    bipartite (frustration); we keep the BFS assignment and then run a short
    greedy repair that flips any site sitting on more same-sign edges than
    opposite-sign ones, which lowers the frustrated-bond count without
    pretending the structure is bipartite. The flag is returned so the caller
    can record that this candidate is approximate.
    """
    col = {comp[0]: +1}
    q = deque([comp[0]])
    bipartite = True
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in col:
                col[v] = -col[u]
                q.append(v)
            elif col[v] == col[u]:
                bipartite = False
    if not bipartite:
        for _ in range(4):                              # bounded greedy repair
            moved = False
            for u in comp:
                same = sum(1 for v in adj[u] if col.get(v) == col[u])
                diff = sum(1 for v in adj[u] if col.get(v) == -col[u])
                if same > diff:
                    col[u] = -col[u]
                    moved = True
            if not moved:
                break
    return col, bipartite


# ══════════════════════════════════════════════════════════════════════════════
#  canonicalisation
# ══════════════════════════════════════════════════════════════════════════════
def canonical(sign):
    """Canonical form of a sign vector up to a GLOBAL FLIP.

    The branch trains under a gauge co-flip, so s and -s are one hypothesis.
    Fix the gauge by the first non-zero entry.
    """
    sign = np.asarray(sign, dtype=np.int8)
    nz = np.flatnonzero(sign)
    if nz.size and sign[nz[0]] < 0:
        return (-sign).astype(np.int8)
    return sign


def _key(sign):
    return canonical(sign).tobytes()


# ══════════════════════════════════════════════════════════════════════════════
#  the proposer
# ══════════════════════════════════════════════════════════════════════════════
def propose(species, edge_index=None, *, k_max=DEFAULT_K, structure=None,
            include_nm=True):
    """Propose up to `k_max` candidate arrangements for one structure.

    Parameters
    ----------
    species     : sequence of element symbols, one per site, in graph order.
    edge_index  : [2, E] neighbour list over all sites (PyG convention), or
                  None. Without it only the composition-only candidates (nm,
                  fm, species splits) can be produced.
    k_max       : cap on returned candidates.
    structure   : optional pymatgen Structure, used ONLY to add symmetry-orbit
                  candidates. Never required.
    include_nm  : emit the all-zero non-magnetic hypothesis.

    Returns
    -------
    list of dicts, each {'name', 'sign' (int8 array over ALL sites),
    'n_up', 'n_down', 'net_formal', 'exact'}, in priority order. `net_formal`
    is the formal-magnitude net Σ sᵢ·μᵢ, a cheap prior on how magnetic the
    arrangement claims to be. `exact` is False for candidates that required a
    frustrated (non-bipartite) approximation.
    """
    species = list(species)
    n = len(species)
    mag = magnetic_mask(species)
    fmag = formal_magnitudes(species)
    mag_idx = np.flatnonzero(mag)

    out, seen = [], set()

    def add(name, sign, exact=True):
        sign = np.asarray(sign, dtype=np.int8)
        if sign.shape != (n,):
            raise ValueError(f'propose: candidate {name!r} has shape '
                             f'{sign.shape}, expected ({n},)')
        if np.any(sign[~mag] != 0):
            raise ValueError(f'propose: candidate {name!r} puts a sign on a '
                             f'non-magnetic site')
        k = _key(sign)
        if k in seen:
            return False
        seen.add(k)
        out.append({'name': name, 'sign': canonical(sign),
                    'n_up': int((sign > 0).sum()), 'n_down': int((sign < 0).sum()),
                    'net_formal': float(np.dot(sign, fmag)), 'exact': bool(exact)})
        return True

    # 1 ── the non-magnetic hypothesis
    if include_nm:
        add('nm', np.zeros(n, dtype=np.int8))

    # 2 ── the ferromagnetic hypothesis
    if mag_idx.size:
        s = np.zeros(n, dtype=np.int8)
        s[mag_idx] = 1
        add('fm', s)

    # nothing further to say without magnetic sites
    if mag_idx.size == 0:
        return out[:k_max]

    adj = _magnetic_adjacency(edge_index, mag_idx, n)
    comps = _components(adj)

    # 3 ── antiferromagnetic bipartition of the neighbour graph
    if edge_index is not None and any(len(a) for a in adj):
        col, bip_all = {}, True
        for c in comps:
            cc, bip = two_colour(adj, c)
            col.update(cc)
            bip_all = bip_all and bip
        s = np.zeros(n, dtype=np.int8)
        for i, g in enumerate(mag_idx):
            s[g] = col.get(i, 1)
        add('afm_bipartite', s, exact=bip_all)

        # 4 ── independent component flips (their relative sign is free)
        if len(comps) > 1:
            for ci in range(1, min(len(comps), k_max)):
                s2 = s.copy()
                for i in comps[ci]:
                    s2[mag_idx[i]] *= -1
                add(f'component_flip_{ci}', s2, exact=bip_all)

    # 5 ── ferrimagnetic species splits
    sp_mag = sorted({str(species[i]) for i in mag_idx})
    if len(sp_mag) > 1:
        for j in range(1, 1 << (len(sp_mag) - 1)):      # up to a global flip
            s = np.zeros(n, dtype=np.int8)
            for i in mag_idx:
                bit = sp_mag.index(str(species[i]))
                # species 0 pins the gauge at +1; species b>=1 takes bit b-1 of j
                s[i] = -1 if (bit > 0 and ((j >> (bit - 1)) & 1)) else 1
            add('species_split_%d' % j, s)

    # 6 ── optional symmetry-orbit patterns (pymatgen, never required)
    if structure is not None:
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            sym = SpacegroupAnalyzer(structure, symprec=0.1)
            orbits = [list(g) for g in sym.get_symmetrized_structure().equivalent_indices]
            mag_orbits = [o for o in orbits if mag[o[0]]]
            if len(mag_orbits) > 1:
                for j in range(1, 1 << (len(mag_orbits) - 1)):
                    s = np.zeros(n, dtype=np.int8)
                    for oi, orb in enumerate(mag_orbits):
                        val = -1 if (oi > 0 and (j >> (oi - 1)) & 1) else 1
                        for i in orb:
                            s[i] = val
                    add('orbit_split_%d' % j, s)
        except Exception as exc:                        # never fatal
            sys.stderr.write(f'[propose] orbit candidates skipped: '
                             f'{type(exc).__name__}: {exc}\n')

    return out[:k_max]


def propose_from_data(data, *, k_max=DEFAULT_K, species_key='species'):
    """Convenience wrapper for a PyG Data object carrying a `species` list."""
    sp = getattr(data, species_key, None)
    if sp is None:
        raise AttributeError(f'propose_from_data: data has no {species_key!r}')
    ei = getattr(data, 'edge_index', None)
    if ei is not None and hasattr(ei, 'cpu'):
        ei = ei.cpu().numpy()
    return propose(sp, ei, k_max=k_max)


# ══════════════════════════════════════════════════════════════════════════════
#  dry run
# ══════════════════════════════════════════════════════════════════════════════
def _demo_cases():
    """Small hand-checkable structures: (name, species, edges, expectation)."""
    def chain(sp):
        e = [[], []]
        for i in range(len(sp) - 1):
            e[0] += [i, i + 1]
            e[1] += [i + 1, i]
        return np.array(e)

    def ring(sp):
        e = [[], []]
        n = len(sp)
        for i in range(n):
            j = (i + 1) % n
            e[0] += [i, j]
            e[1] += [j, i]
        return np.array(e)

    return [
        ('MnO chain, 4 Mn', ['Mn', 'O', 'Mn', 'O', 'Mn', 'O', 'Mn', 'O'],
         chain(['Mn', 'O'] * 4), 'bipartite AFM available'),
        ('Mn triangle (frustrated)', ['Mn', 'Mn', 'Mn'], ring(['Mn'] * 3),
         'odd cycle -> afm_bipartite approximate'),
        ('Mn4 square ring', ['Mn'] * 4, ring(['Mn'] * 4), 'bipartite'),
        ('GdFe ferrimagnet', ['Gd', 'Fe', 'Gd', 'Fe'], ring(['Gd', 'Fe'] * 2),
         'species split gives a large net'),
        ('NaCl (no magnetic sites)', ['Na', 'Cl'], chain(['Na', 'Cl']),
         'nm only'),
        ('two isolated Fe pairs', ['Fe', 'Fe', 'Fe', 'Fe'],
         np.array([[0, 1, 2, 3], [1, 0, 3, 2]]), 'component flip available'),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--k', type=int, default=DEFAULT_K, help='candidate cap')
    ap.add_argument('--json', action='store_true', help='emit JSON')
    a = ap.parse_args()

    print(f'formal-magnitude table: {len(FORMAL_MAGNITUDE)} magnetic species, '
          f'max {max(FORMAL_MAGNITUDE.values())} μB')
    print(f'candidate cap K = {a.k}\n')
    blob = []
    for name, sp, ei, note in _demo_cases():
        cands = propose(sp, ei, k_max=a.k)
        print(f'{name}  —  {note}')
        print(f'    sites: {sp}')
        for c in cands:
            flag = '' if c['exact'] else '   (approximate: frustrated)'
            print('    %-18s %-28s up/down %d/%d   formal net %+6.1f%s'
                  % (c['name'], np.array2string(c['sign'], separator=','),
                     c['n_up'], c['n_down'], c['net_formal'], flag))
        print()
        blob.append({'case': name, 'species': sp,
                     'candidates': [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                     for k, v in c.items()} for c in cands]})
    if a.json:
        print(json.dumps(blob, indent=1))


if __name__ == '__main__':
    main()
