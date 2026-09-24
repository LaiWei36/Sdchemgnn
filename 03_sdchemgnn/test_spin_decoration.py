"""Offline tests for spin_decoration: seed derivation (eps boundary, NM
seed, the gauge-conflict corner, unsup masking), the raw-jsonl tripwire,
attach + cache verify, EVERY decoration mode (channel exactness, corruption
rates, the co-flip's full target consistency, promotion, site masking, the
within-graph shuffle, fm_formal, the double-decoration guard), the
pinned-vs-min loss dispatch and the self-consistency readout."""
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
import json
import math
import types

import torch

from site_moment_labels import build_magmom_labels, node_gauge_loss
from spin_decoration import (
    FORMAL_MAGNITUDE, SEED_CHANNELS,
    build_seed_table, verify_seed_vs_sidetable, attach_seed_channels,
    get_seed_sign, get_seed_fmag, get_seed_elig, verify_seed_in_cache,
    decorate_batch, node_pinned_loss, combined_node_loss, self_consistency)

PASS = []


def check(name, cond, got=None):
    assert cond, f'FAIL: {name}' + (f' (got {got})' if got is not None else '')
    PASS.append(name)


# ── synthetic side-tables (+ species + gauge row) ─────────────────────────────
rows_mm, rows_ord = [], []
manifest, node_counts = {}, {}


def add(mid, formula, status, magmom, cell_total, n_graph, raw_fu,
        action='', new=None, residual=None, species=None):
    rows_mm.append({'material_id': mid, 'formula': formula, 'status': status,
                    'n_sites': len(magmom) if magmom else n_graph,
                    'species': species, 'magmom': magmom,
                    'cell_total': cell_total,
                    **({'residual': residual} if residual is not None else {})})
    rows_ord.append({'material_id': mid,
                     'jsonl_raw_fu': '' if raw_fu is None else f'{raw_fu:.6f}',
                     'formula': formula})
    if action:
        manifest[mid] = (action, new)
    node_counts[mid] = n_graph


# supervised FM row; site 1/3 sit exactly ON eps=0.1 (|m| > eps is FALSE -> 0)
add('mp-ok', 'FeO', 'aligned', [2.1, 0.1, 2.1, 0.1], 4.5, 4, 2.25,
    residual=0.1, species=['Fe', 'O', 'Fe', 'O'])
# compensated AFM row -> mixed signs
add('mp-afm', 'MnF2', 'aligned', [4.5, -4.5, 0.0, 0.0, 0.0, 0.0], 0.05, 6,
    0.025, residual=0.05, species=['Mn', 'Mn', 'F', 'F', 'F', 'F'])
# no_magmom, both totals ~0 -> NM seed, promotable V site
add('mp-zero', 'VO2', 'no_magmom', None, 0.0, 6, 0.0,
    species=['V', 'V', 'O', 'O', 'O', 'O'])
# zero-moment magnetic species (promotion candidate) among magnetic labels
# (formula FeF: 2 atoms/f.u., 4 sites -> Z=2 agrees with cell/raw = 4/2)
add('mp-promo', 'FeF', 'aligned', [4.0, 0.02, 0.0, 0.0], 4.0, 4, 2.0,
    residual=0.02, species=['Fe', 'Fe', 'F', 'F'])
# THE GAUGE CORNER: curated y = +3/f.u. but stored site gauge sums to -6
add('mp-gauge', 'CoO', 'aligned', [-3.0, -3.0, 0.0, 0.0], 6.0, 4, 3.0,
    residual=12.0, species=['Co', 'Co', 'O', 'O'])
# masked row (relabel0) -> never seed-eligible
add('mp-r0', 'MnO', 'aligned', [4.0, 0.0, 4.0, 0.0], 8.0, 4, 4.0,
    action='relabel0', new=0.0, species=['Mn', 'O', 'Mn', 'O'])
# big-y rows so the label builder's sign sweep has members
for i in range(6):
    add(f'mp-big{i}', 'Fe2O3', 'aligned', [3.0, 3.0, 0.0, 0.0, 0.0], 6.0, 5,
        6.0, residual=0.0, species=['Fe', 'Fe', 'O', 'O', 'O'])

with open('/tmp/bs_mm.jsonl', 'w') as fh:
    for r in rows_mm:
        fh.write(json.dumps(r) + '\n')
with open('/tmp/bs_ord.csv', 'w') as fh:
    fh.write('material_id,jsonl_raw_fu,formula\n')
    for r in rows_ord:
        fh.write(f'{r["material_id"]},{r["jsonl_raw_fu"]},{r["formula"]}\n')

labels, lmeta = build_magmom_labels('/tmp/bs_mm.jsonl', '/tmp/bs_ord.csv',
                                    manifest, node_counts,
                                    halt_nsite_frac=0.5, halt_zsuspect_frac=0.5)
check('label builder: gauge row is supervised (the site-decomposition model keeps it)',
      labels['mp-gauge']['sup'] == 1)

seeds, meta = build_seed_table('/tmp/bs_mm.jsonl', '/tmp/bs_ord.csv', manifest,
                               labels, node_counts, eps=0.1)

# derivation branches
check('ok row eligible with eps-boundary zeros',
      seeds['mp-ok']['elig'] == 1 and seeds['mp-ok']['sign'] == [1, 0, 1, 0])
check('eps-band sites counted (two 0.1 sites)', meta['n_eps_band_sites'] >= 2)
check('afm row mixed signs', seeds['mp-afm']['sign'] == [1, -1, 0, 0, 0, 0]
      and meta['n_mixed_sign'] == 1)
check('NM seed on zero row', seeds['mp-zero']['why'] == 'nm_seed'
      and seeds['mp-zero']['sign'] == [0] * 6)
check('formal magnitudes from the ceiling table',
      seeds['mp-ok']['fmag'] == [5.0, 0.0, 5.0, 0.0]
      and seeds['mp-zero']['fmag'][0] == 3.0)      # V = 3
check('gauge conflict seed-masked (precondition 0.iv)',
      seeds['mp-gauge']['elig'] == 0
      and seeds['mp-gauge']['why'] == 'gauge_conflict'
      and meta['gauge_conflict_ids'] == ['mp-gauge'])
check('relabel0 row never eligible', seeds['mp-r0']['elig'] == 0
      and seeds['mp-r0']['why'].startswith('unsup_'))
check('promotable census counts zero-sign magnetic species',
      meta['n_promotable_sites'] >= 3)   # mp-promo Fe(0.02) + mp-zero 2x V
check('every id covered', len(seeds) == len(node_counts))

# tripwire: consistent build passes; a corrupted seed table raises
n_chk = verify_seed_vs_sidetable(seeds, '/tmp/bs_mm.jsonl', 0.1)
check('tripwire re-derives eligible rows', n_chk >= 8)
bad_seeds = {k: dict(v) for k, v in seeds.items()}
bad_seeds['mp-ok'] = dict(bad_seeds['mp-ok'], sign=[1, 0, -1, 0])
try:
    verify_seed_vs_sidetable(bad_seeds, '/tmp/bs_mm.jsonl', 0.1)
    raise SystemExit('FAIL: corrupted seed table did not raise')
except RuntimeError as exc:
    check('tripwire raises on divergence', 'SEED TRIPWIRE FAILED' in str(exc))

# ── attach + cache verify on a fake master ─────────────────────────────────────
order = list(node_counts)
offsets = [0]
for m in order:
    offsets.append(offsets[-1] + node_counts[m])
store = types.SimpleNamespace(material_id=order)
ds = types.SimpleNamespace(_data=store, slices={'x': torch.tensor(offsets)})
ids, n_elig = attach_seed_channels(ds, seeds)
check('attach ids order', ids == order)
check('attach n_elig matches meta', n_elig == meta['n_elig'])
check('attach node-aligned slices',
      torch.equal(ds.slices['seed_sign'], ds.slices['x'])
      and torch.equal(ds.slices['seed_fmag'], ds.slices['x']))
j = order.index('mp-afm')
lo, hi = offsets[j], offsets[j + 1]
check('attach sign values placed',
      torch.equal(store.seed_sign[lo:hi],
                  torch.tensor([1.0, -1.0, 0.0, 0.0, 0.0, 0.0])))
jg = order.index('mp-gauge')
check('gauge row: zero signs, elig 0, fmag still attached',
      int(store.seed_elig[jg]) == 0
      and float(store.seed_sign[offsets[jg]:offsets[jg + 1]].abs().sum()) == 0.0
      and float(store.seed_fmag[offsets[jg]]) == 4.0)   # Co = 4


def fake_batch(id_slice):
    idx = [order.index(m) for m in id_slice]
    sgn, fmg, bv = [], [], []
    for k, j2 in enumerate(idx):
        sgn.append(store.seed_sign[offsets[j2]:offsets[j2 + 1]])
        fmg.append(store.seed_fmag[offsets[j2]:offsets[j2 + 1]])
        bv.extend([k] * node_counts[order[j2]])
    return types.SimpleNamespace(
        material_id=id_slice,
        seed_sign=torch.cat(sgn), seed_fmag=torch.cat(fmg),
        seed_elig=store.seed_elig[torch.tensor(idx)],
        batch=torch.tensor(bv))


cache = [fake_batch(order[:5]), fake_batch(order[5:])]
verify_seed_in_cache(cache, seeds, 'test')
PASS.append('cache verify full sweep OK')
cache[0].seed_sign[0] = -1.0
try:
    verify_seed_in_cache(cache, seeds, 'test')
    raise SystemExit('FAIL: corrupted cached seed did not raise')
except RuntimeError as exc:
    check('corrupted cache raises', 'SEED VERIFY FAILED' in str(exc))

# ── decoration: exact channels + guards ────────────────────────────────────────
BASE = 9


def dec_batch(ids2):
    b = fake_batch(ids2)
    N = b.batch.numel()
    G = len(ids2)
    b.x = torch.randn(N, BASE)
    b.num_graphs = G
    b.node_magmom = torch.randn(N)
    b.resid_target = torch.randn(G)
    b.z_fu = torch.full((G,), 2.0)
    b.y = torch.randn(G, 2)
    b.total_magnetization_normalized_formula_units = b.y[:, 1].clone().reshape(-1, 1)
    return b


DCFG = {'base_channels': BASE, 'p_undecorate': 0.0, 'p_site_mask': 0.0,
        'p_promote': 0.0, 'fmag_scale': 7.0, 'coflip': False}

b = dec_batch(['mp-ok', 'mp-gauge'])
x0 = b.x.clone()
b = decorate_batch(b, 'oracle', DCFG)
check('oracle widens x by 3', b.x.shape == (8, BASE + SEED_CHANNELS))
check('oracle keeps base features', torch.equal(b.x[:, :BASE], x0))
check('oracle sign channel (eligible graph)',
      torch.equal(b.x[:4, BASE], torch.tensor([1.0, 0.0, 1.0, 0.0])))
check('oracle fmag channel scaled by 7',
      torch.allclose(b.x[:4, BASE + 1], torch.tensor([5 / 7, 0.0, 5 / 7, 0.0])))
check('oracle flag = 1 on decorated graph only',
      torch.equal(b.x[:, BASE + 2],
                  torch.tensor([1.0] * 4 + [0.0] * 4)))
check('oracle: ineligible (gauge) graph fully zero',
      float(b.x[4:, BASE:].abs().sum()) == 0.0)
check('seed_dec / seed_flip set', b.seed_dec.tolist() == [True, False]
      and b.seed_flip.tolist() == [False, False])
try:
    decorate_batch(b, 'oracle', DCFG)
    raise SystemExit('FAIL: double decoration did not raise')
except RuntimeError as exc:
    check('double decoration raises', 'double' in str(exc))

b = decorate_batch(dec_batch(['mp-ok', 'mp-gauge']), 'undecorated', DCFG)
check('undecorated: three zero channels, no graph decorated',
      float(b.x[:, BASE:].abs().sum()) == 0.0
      and not bool(b.seed_dec.any()))

b = decorate_batch(dec_batch(['mp-ok', 'mp-gauge']), 'fm_formal', DCFG)
check('fm_formal: +1 exactly on magnetic species, flag 1 everywhere',
      torch.equal(b.x[:, BASE],
                  torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]))
      and torch.equal(b.x[:, BASE + 2], torch.ones(8))
      and bool(b.seed_dec.all()))

# shuffled: multiset preserved within graph, arrangement broken somewhere
g = torch.Generator(); g.manual_seed(11)
torch.manual_seed(42)
b_or = decorate_batch(dec_batch(['mp-afm', 'mp-ok']), 'oracle', DCFG)
torch.manual_seed(42)
b_sh = decorate_batch(dec_batch(['mp-afm', 'mp-ok']), 'shuffled', DCFG, g)
check('shuffled preserves the within-graph sign multiset',
      sorted(b_sh.x[:6, BASE].tolist()) == sorted(b_or.x[:6, BASE].tolist())
      and sorted(b_sh.x[6:, BASE].tolist()) == sorted(b_or.x[6:, BASE].tolist()))
check('shuffled changes an arrangement',
      not torch.equal(b_sh.x[:, BASE], b_or.x[:, BASE]))
check('shuffled keeps base features un-permuted',
      torch.equal(b_sh.x[:, :BASE], b_or.x[:, :BASE]))

# ── corruption rates (statistical, seeded) ─────────────────────────────────────
NG, NN = 400, 4
big = types.SimpleNamespace(
    x=torch.randn(NG * NN, BASE),
    batch=torch.arange(NG).repeat_interleave(NN),
    num_graphs=NG,
    seed_sign=torch.ones(NG * NN),
    seed_fmag=torch.full((NG * NN,), 5.0),
    seed_elig=torch.ones(NG, dtype=torch.int8),
    node_magmom=torch.randn(NG * NN),
    resid_target=torch.randn(NG),
    z_fu=torch.full((NG,), 2.0),
    y=torch.randn(NG, 2))
big.total_magnetization_normalized_formula_units = big.y[:, 1].clone().reshape(-1, 1)
g = torch.Generator(); g.manual_seed(5)
D2 = dict(DCFG, p_undecorate=0.25, p_site_mask=0.10, coflip=False)
b = decorate_batch(big, 'train', D2, g)
frac_dec = float(b.seed_dec.float().mean())
check('p_undecorate ~ 0.25 (got dec frac in [0.68, 0.82])',
      0.68 <= frac_dec <= 0.82, frac_dec)
dec_nodes = b.seed_dec[b.batch]
flag = b.x[:, BASE + 2]
masked = dec_nodes & (flag == 0)
frac_mask = float(masked.float().sum() / dec_nodes.float().sum())
check('p_site_mask ~ 0.10 (got [0.06, 0.14])', 0.06 <= frac_mask <= 0.14,
      frac_mask)
check('masked sites zero ALL three channels',
      float(b.x[masked][:, BASE:].abs().sum()) == 0.0)

# promotion: all-zero signs on magnetic species
big2 = types.SimpleNamespace(**{**big.__dict__})
big2.x = torch.randn(NG * NN, BASE)
big2.seed_sign = torch.zeros(NG * NN)
big2.y = torch.randn(NG, 2)
big2.total_magnetization_normalized_formula_units = big2.y[:, 1].clone().reshape(-1, 1)
g = torch.Generator(); g.manual_seed(7)
D3 = dict(DCFG, p_promote=0.30, coflip=False)
b2 = decorate_batch(big2, 'train', D3, g)
prom = b2.x[:, BASE].abs() > 0
frac_prom = float(prom.float().mean())
check('p_promote ~ 0.30 on promotable sites (got [0.25, 0.35])',
      0.25 <= frac_prom <= 0.35, frac_prom)
check('promoted signs are +/-1 with both signs drawn',
      set(b2.x[prom][:, BASE].tolist()) == {-1.0, 1.0})

# ── co-flip: EVERY signed target flips together ────────────────────────────────
big3 = types.SimpleNamespace(**{**big.__dict__})
big3.x = torch.randn(NG * NN, BASE)
big3.seed_sign = torch.ones(NG * NN)
big3.node_magmom = torch.randn(NG * NN)
big3.resid_target = torch.randn(NG)
big3.y = torch.randn(NG, 2)
big3.total_magnetization_normalized_formula_units = big3.y[:, 1].clone().reshape(-1, 1)
nm0 = big3.node_magmom.clone()
rt0 = big3.resid_target.clone()
y0 = big3.y.clone()
g = torch.Generator(); g.manual_seed(3)
D4 = dict(DCFG, coflip=True)
b3 = decorate_batch(big3, 'train', D4, g)
fl = b3.seed_flip
check('co-flip hits ~half the decorated graphs',
      0.35 <= float(fl.float().sum() / b3.seed_dec.float().sum()) <= 0.65)
fn = fl[b3.batch]
check('co-flip negates the seed channel',
      torch.allclose(b3.x[fn, BASE], -torch.ones(int(fn.sum()))))
check('co-flip negates node_magmom on flipped graphs',
      torch.allclose(b3.node_magmom[fn], -nm0[fn])
      and torch.allclose(b3.node_magmom[~fn], nm0[~fn]))
check('co-flip negates resid_target',
      torch.allclose(b3.resid_target[fl], -rt0[fl])
      and torch.allclose(b3.resid_target[~fl], rt0[~fl]))
check('co-flip negates the y magnetization column ONLY',
      torch.allclose(b3.y[fl, 1], -y0[fl, 1])
      and torch.allclose(b3.y[:, 0], y0[:, 0]))
check('co-flip negates the NAMED per-f.u. attribute (read-path lesson)',
      torch.allclose(
          b3.total_magnetization_normalized_formula_units.reshape(-1)[fl],
          -y0[fl, 1]))
check('undecorated graphs never flip', not bool((fl & ~b3.seed_dec).any()))

# ── loss dispatch: pinned on decorated, min-over-s on undecorated ──────────────
m_true = torch.tensor([3.0, -3.0, 2.0, 2.0])
m_hat = torch.tensor([-3.0, 3.0, 2.0, 2.0])       # graph 0 globally flipped
bv = torch.tensor([0, 0, 1, 1])
Lp = node_pinned_loss(m_hat, m_true, bv, 2)
check('pinned loss does NOT forgive the flip', abs(float(Lp[0]) - 6.0) < 1e-6)
L, flip = combined_node_loss(m_hat, m_true, bv, 2,
                             torch.tensor([True, False]), node_gauge_loss)
check('decorated graph scored PINNED (6.0)', abs(float(L[0]) - 6.0) < 1e-6)
check('flip flag forced False on decorated graphs', not bool(flip[0]))
L2, flip2 = combined_node_loss(m_hat, m_true, bv, 2,
                               torch.tensor([False, False]), node_gauge_loss)
check('undecorated graph keeps min-over-s (0.0)', abs(float(L2[0])) < 1e-6
      and bool(flip2[0]))
# gradient flows through the pinned branch
mh = m_hat.clone().requires_grad_(True)
L3, _ = combined_node_loss(mh, m_true, bv, 2, torch.tensor([True, False]),
                           node_gauge_loss)
L3.mean().backward()
check('pinned branch gradient flows', float(mh.grad.abs().sum()) > 0)

# ── self-consistency ───────────────────────────────────────────────────────────
sc_b = types.SimpleNamespace(
    x=torch.cat([torch.zeros(6, BASE),
                 torch.tensor([[1.0], [-1.0], [0.0], [1.0], [0.0], [0.0]]),
                 torch.zeros(6, 1),
                 torch.tensor([[1.0], [1.0], [1.0], [1.0], [0.0], [0.0]])],
                dim=1),
    batch=torch.tensor([0, 0, 0, 1, 1, 1]), num_graphs=2)
m_hat_sc = torch.tensor([2.0, -1.0, 0.5, -0.4, 0.0, 0.0])
frac, n_comm = self_consistency(m_hat_sc, sc_b, tiny=0.05)
check('self-consistency: g0 both committal seeded sites agree (frac 1.0)',
      abs(float(frac[0]) - 1.0) < 1e-6)
check('self-consistency: g1 single committal site disagrees (frac 0.0)',
      abs(float(frac[1])) < 1e-6)
check('self-consistency: committal count = 3', n_comm == 3)
frac2, _ = self_consistency(torch.zeros(6), sc_b, tiny=0.05)
check('self-consistency: no committal sites -> NaN', bool(torch.isnan(frac2).all()))

# formal-magnitude table sanity (matches the ceiling table import guard)
check('formal magnitudes: Fe=5 Gd=7 Eu=7, O absent',
      FORMAL_MAGNITUDE['Fe'] == 5 and FORMAL_MAGNITUDE['Gd'] == 7
      and FORMAL_MAGNITUDE['Eu'] == 7 and 'O' not in FORMAL_MAGNITUDE)

print(f'ALL {len(PASS)} SEED/DECORATION CHECKS PASSED')
