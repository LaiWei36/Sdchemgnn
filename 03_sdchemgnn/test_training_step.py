"""Extraction test for the PATCHED train_epoch / evaluate_nodes /
collect_sitedecomp / _dec_eval in train_sdchemgnn.py: the
functions are pulled from the REAL file and run against hand-computed batches
through the REAL decorate_batch/combined_node_loss. Verifies: the decoration
fires inside the loops, the PINNED-vs-min node-loss dispatch (a globally
flipped prediction on a DECORATED graph is punished, on an undecorated one
forgiven), the per-f.u. residual loss, the decoration census, the per-f.u.
r_hat stats, self-consistency, and the 5-column sitedecomp rows."""
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
import math
import re
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from site_moment_labels import (node_gauge_loss, get_node_magmom,
                                   get_node_sup, get_resid_target, get_z_fu)
from spin_decoration import (decorate_batch, combined_node_loss,
                                self_consistency)

PASS = []


def check(name, cond, got=None):
    assert cond, f'FAIL: {name}' + (f' (got {got})' if got is not None else '')
    PASS.append(name)


src = open('train_sdchemgnn.py').read()


def extract(name):
    m = re.search(rf'^(@torch\.no_grad\(\)\n)?def {name}\(.*?(?=\n\n\n)',
                  src, re.S | re.M)
    assert m, f'cannot extract {name}'
    return m.group(0)


class CfgStub:
    target_cap = 30.0
    bulk_cutoff = 2.0
    sc_tiny = 0.05


ns = {
    'torch': torch, 'F': F, 'nn': nn, 'time': __import__('time'),
    'sys': __import__('sys'), 'os': __import__('os'), 'math': math,
    'device': torch.device('cpu'), '_PROFILE_BATCHES': 0,
    '_YIELD': {'flag': False, 'why': ''}, 'cfg': CfgStub,
    '_cuda_sync': lambda: None,
    'get_magnetization': lambda d: d.M.reshape(-1),
    'get_ordering_mode': lambda d: d.km.reshape(-1),
    'cap_target': lambda t: t.clamp(-30.0, 30.0),
    'node_gauge_loss': node_gauge_loss,
    'get_node_magmom': get_node_magmom, 'get_node_sup': get_node_sup,
    'get_resid_target': get_resid_target, 'get_z_fu': get_z_fu,
    'decorate_batch': decorate_batch,
    'combined_node_loss': combined_node_loss,
    'self_consistency': self_consistency,
}
for fn in ('_huber_elem', '_dec_eval', 'evaluate_nodes', 'collect_sitedecomp',
           'train_epoch'):
    exec(extract(fn), ns)                     # noqa: S102 (offline test)

# ── the hand-computed batch ────────────────────────────────────────────────────
# g0: 4 nodes, SUPERVISED + SEED-ELIGIBLE, m=[2,-2,1,0], signs=[1,-1,1,0],
#     resid_target 0.3 uB/cell, Z=2, M=1, mode 0
# g1: 2 nodes, unsupervised, INELIGIBLE,               resid 0, Z=3, M=3, -1
BASE = 9


def make_batch():
    b = types.SimpleNamespace(
        M=torch.tensor([1.0, 3.0]), km=torch.tensor([0, -1]),
        x=torch.zeros(6, BASE),
        batch=torch.tensor([0, 0, 0, 0, 1, 1]), num_graphs=2,
        node_magmom=torch.tensor([2.0, -2.0, 1.0, 0.0, 0.0, 0.0]),
        node_sup=torch.tensor([1, 0], dtype=torch.int8),
        resid_target=torch.tensor([0.3, 0.0]),
        z_fu=torch.tensor([2.0, 3.0]),
        y=torch.tensor([[0.0, 1.0], [0.0, 3.0]]),
        seed_sign=torch.tensor([1.0, -1.0, 1.0, 0.0, 0.0, 0.0]),
        seed_fmag=torch.tensor([5.0, 5.0, 5.0, 0.0, 0.0, 0.0]),
        seed_elig=torch.tensor([1, 0], dtype=torch.int8))
    b.total_magnetization_normalized_formula_units = \
        b.M.clone().reshape(-1, 1)
    b.to = lambda *a, **k: b
    return b


DCFG = {'base_channels': BASE, 'p_undecorate': 0.0, 'p_site_mask': 0.0,
        'p_promote': 0.0, 'fmag_scale': 7.0, 'coflip': False}
DEC_TRAIN = {'dcfg': DCFG, 'gen': torch.Generator()}
DEC_ORACLE = {'mode': 'oracle', 'dcfg': DCFG}


class FakeModel(nn.Module):
    """SD-ChemGNN forward_b contract: y = m_sum/Z + r_hat (r_hat PER F.U.). Asserts
    the batch reached it DECORATED (the loops must decorate first)."""

    def __init__(self, m_hat):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))
        self._m = m_hat

    def forward_b(self, data):
        assert data.x.size(1) == BASE + 3, 'batch reached the model undecorated'
        z = self.p * 0
        m_hat = torch.tensor(self._m) + z
        r_hat = torch.tensor([0.2, -0.1]) + z
        m_sum = torch.zeros(2).index_add(0, data.batch, m_hat)
        y = m_sum / data.z_fu + r_hat
        y_aux = torch.tensor([0.5, 2.5]) + z
        logits = torch.tensor([[2.0, 0.5, 0.5], [0.2, 0.2, 0.2]]) + z
        vals = torch.tensor([[0.05, 6.0, 6.0], [0.05, 6.0, 6.0]]) + z
        return {'y': y, 'y_aux': y_aux, 'm_hat': m_hat, 'r_hat': r_hat,
                'm_sum': m_sum, 'cell': m_sum + r_hat * data.z_fu,
                'mode_logits': logits, 'mode_vals': vals}


class Loader:
    def __iter__(self):
        return iter([make_batch()])

    def __len__(self):
        return 1


M_HAT = [1.8, -2.1, 0.9, 0.1, 0.5, 0.4]
model = FakeModel(M_HAT)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
loss_fn = lambda p, t: (p - t).abs().mean()          # plain L1
aux = {'w_mode': 0.2, 'w_val': 0.1, 'ce_w': torch.ones(3), 'empty_batches': 0}
bopt = {'w_node': 0.5, 'w_resid': 0.1, 'w_shrink': 0.01, 'w_aux': 0.2,
        'huber_d': 1.0, 'empty_batches': 0}

train_loss, train_mae, aux_stats, bstats = ns['train_epoch'](
    model, optimizer, loss_fn, Loader(), mask_nan=False, aux=aux, bopt=bopt,
    dec=DEC_TRAIN)

# hand computation — SD-ChemGNN reconstruction
m_sum0 = 1.8 - 2.1 + 0.9 + 0.1                        # 0.7
m_sum1 = 0.5 + 0.4                                    # 0.9
y0 = m_sum0 / 2.0 + 0.2                               # 0.55
y1 = m_sum1 / 3.0 - 0.1                               # 0.2
reg = (abs(y0 - 1.0) + abs(y1 - 3.0)) / 2             # 1.625
check('train reg loss (per-f.u. reconstruction)',
      abs(train_loss - reg) < 1e-5, train_loss)
check('train MAE matches', abs(train_mae - reg) < 1e-5, train_mae)
# g0 decorated -> PINNED node loss = mean(|.2|,|.1|,|.1|,|.1|) = 0.125
check('bstats node = 0.125 (pinned, g0)', abs(bstats['node'] - 0.125) < 1e-6,
      bstats['node'])
# residual PER F.U.: Huber(0.2, 0.3/2=0.15, d=1) = 0.5*0.05^2 = 0.00125
check('bstats resid = Huber(0.2, 0.15) = 0.00125',
      abs(bstats['resid'] - 0.00125) < 1e-8, bstats['resid'])
check('bstats shrink = mean|r_hat| = 0.15', abs(bstats['shrink'] - 0.15) < 1e-6)
check('bstats sup_frac = 0.5', abs(bstats['sup_frac'] - 0.5) < 1e-6)
check('bstats dec_frac = 0.5 (only g0 eligible)',
      abs(bstats['dec_frac'] - 0.5) < 1e-6, bstats['dec_frac'])
check('bstats flip_frac = 0 (coflip off)', bstats['flip_frac'] == 0.0)
check('aux l1 = |mu_0 - M| = 0.95', abs(aux_stats['l1'] - 0.95) < 1e-6)

# ── the DISPATCH bites: a globally flipped prediction on the DECORATED graph
# is NOT forgiven (the site-decomposition model's min-over-s would have scored it 0.125) ──────────
model_flip = FakeModel([-1.8, 2.1, -0.9, -0.1, 0.5, 0.4])
_, _, _, bs_flip = ns['train_epoch'](
    model_flip, torch.optim.SGD(model_flip.parameters(), lr=0.0), loss_fn,
    Loader(), mask_nan=False, aux=aux, bopt=bopt, dec=DEC_TRAIN)
pin_expect = (3.8 + 4.1 + 1.9 + 0.1) / 4              # 2.475
check('pinned dispatch punishes the flipped gauge on decorated g0',
      abs(bs_flip['node'] - pin_expect) < 1e-6, bs_flip['node'])

# gradient assembles through the composed loss
model2 = FakeModel(M_HAT)
ns['train_epoch'](model2, torch.optim.SGD(model2.parameters(), lr=0.0),
                  loss_fn, Loader(), mask_nan=False, aux=aux, bopt=bopt,
                  dec=DEC_TRAIN)
check('composed loss backward reaches params', model2.p.grad is not None)

# _huber_elem unchanged
h = ns['_huber_elem'](torch.tensor([0.5, 3.0]), torch.tensor([0.0, 0.0]), 1.0)
check('_huber_elem = mean(0.125, 2.5)', abs(float(h) - (0.125 + 2.5) / 2) < 1e-6)

# ── evaluate_nodes (oracle mode) ───────────────────────────────────────────────
nv = ns['evaluate_nodes'](model, Loader(), DEC_ORACLE)
check('nodes site_mae 0.125 (pinned)', abs(nv['site_mae'] - 0.125) < 1e-6,
      nv['site_mae'])
check('nodes n_sites = 4', nv['n_sites'] == 4)
check('nodes r_abs (per f.u.) = 0.15', abs(nv['r_abs'] - 0.15) < 1e-6)
check('nodes r_corr nan for 1 point', math.isnan(nv['r_corr']))
# tail share on g1 (M=3>2): |m_sum/Z| = 0.3, |r| = 0.1
share = 0.1 / (0.3 + 0.1 + 1e-6)
check('nodes tail share uses per-f.u. m_sum', abs(nv['r_share'] - share) < 1e-5,
      nv['r_share'])
check('nodes mae_recon = reg', abs(nv['mae_recon'] - reg) < 1e-5)
check('nodes mae_aux = 0.5', abs(nv['mae_aux'] - 0.5) < 1e-6)
# self-consistency: g0 seeded committal sites (m_hat 1.8, -2.1, 0.9) all match the seed
check('nodes self-consistency mean/rate = 1.0 on g0', abs(nv['sc_mean'] - 1.0) < 1e-6
      and abs(nv['sc_rate'] - 1.0) < 1e-6 and nv['sc_n'] == 1)
check('nodes n_dec = 1', nv['n_dec'] == 1)
# a sign-violating prediction drops self-consistency
nv2 = ns['evaluate_nodes'](FakeModel([1.8, 2.1, 0.9, 0.1, 0.5, 0.4]),
                           Loader(), DEC_ORACLE)
check('nodes self-consistency drops on a seed-violating sign (2/3)',
      abs(nv2['sc_mean'] - 2.0 / 3.0) < 1e-6 and nv2['sc_rate'] == 0.0)

# ── collect_sitedecomp: 5 columns, per-f.u. residual, self-consistency ─────────
sd = ns['collect_sitedecomp'](model, Loader(), DEC_ORACLE)
check('sitedecomp shape [2,5]', tuple(sd.shape) == (2, 5))
check('sitedecomp g0 row [4.9, 0.7, 0.2, 2, 1.0]',
      torch.allclose(sd[0], torch.tensor([4.9, 0.7, 0.2, 2.0, 1.0]),
                     atol=1e-5), sd[0])
check('sitedecomp g1: undecorated -> sc NaN',
      torch.allclose(sd[1, :4], torch.tensor([0.9, 0.9, -0.1, 3.0]),
                     atol=1e-5) and math.isnan(float(sd[1, 4])))

# ── co-flip consistency through the REAL decoration (one flipped graph):
# an EQUIVARIANT model (predicts the flipped truth) must see zero node loss
# and an unchanged reg target relationship ────────────────────────────────────
def find_flip_gen():
    for s in range(200):
        gg = torch.Generator(); gg.manual_seed(s)
        b = decorate_batch(make_batch(), 'train',
                           dict(DCFG, coflip=True), gg)
        if bool(b.seed_flip[0]):
            gg2 = torch.Generator(); gg2.manual_seed(s)
            return gg2
    raise SystemExit('FAIL: no flipping seed found in 200 draws')


class EquivModel(FakeModel):
    """Reads the (possibly flipped) seed channel and mirrors the flipped
    truth exactly: m_hat = node_magmom (post-flip), r_hat = resid/Z."""

    def __init__(self):
        super().__init__(None)

    def forward_b(self, data):
        z = self.p * 0
        m_hat = data.node_magmom.reshape(-1).clone() + z
        r_hat = data.resid_target.reshape(-1) / data.z_fu + z
        m_sum = torch.zeros(2).index_add(0, data.batch, m_hat)
        return {'y': m_sum / data.z_fu + r_hat, 'y_aux': torch.zeros(2) + z,
                'm_hat': m_hat, 'r_hat': r_hat, 'm_sum': m_sum,
                'cell': m_sum + r_hat * data.z_fu,
                'mode_logits': torch.zeros(2, 3) + z,
                'mode_vals': torch.zeros(2, 3) + z}


_, _, _, bs_eq = ns['train_epoch'](
    EquivModel(), torch.optim.SGD(EquivModel().parameters(), lr=0.0), loss_fn,
    Loader(), mask_nan=False, aux=aux, bopt=bopt,
    dec={'dcfg': dict(DCFG, coflip=True), 'gen': find_flip_gen()})
check('equivariant model: zero pinned node loss UNDER the co-flip',
      abs(bs_eq['node']) < 1e-6, bs_eq['node'])
check('co-flip actually fired in that epoch', bs_eq['flip_frac'] > 0)
check('equivariant model: residual loss consistent under the flip',
      abs(bs_eq['resid']) < 1e-6, bs_eq['resid'])

# ── the job-20883498 regression: stale item-cache vs the read-back layers ──────
# PyG's InMemoryDataset memoizes separated items (_data_list); an item cached
# by an earlier read-back predates later attaches. _fresh_item_view must clear
# the cache, and main() must call it after EVERY attach, before its read-back.
exec(extract('_fresh_item_view'), ns)


class CachedDS:
    """Stub reproducing InMemoryDataset.get() memoization: attributes present
    on the STORE at first access are frozen into the cached item."""

    def __init__(self):
        self.store = {'node_magmom': torch.tensor([1.0])}
        self._data_list = [None]

    def __getitem__(self, i):
        if self._data_list is None:          # PyG re-creates the cache lazily
            self._data_list = [None]
        if self._data_list[i] is None:
            self._data_list[i] = types.SimpleNamespace(**dict(self.store))
        return self._data_list[i]


dsq = CachedDS()
_ = dsq[0]                                   # manifest-layer access (caches)
dsq.store['seed_sign'] = torch.tensor([1.0])  # ... seed attach happens later
check('stub reproduces the crash: stale item lacks seed_sign',
      not hasattr(dsq[0], 'seed_sign'))
ns['_fresh_item_view'](dsq)
check('_fresh_item_view clears the cache -> fresh item carries seed_sign',
      hasattr(dsq[0], 'seed_sign'))
check('_fresh_item_view tolerates objects without a cache',
      ns['_fresh_item_view'](types.SimpleNamespace()) is not None)

# source-level guard: every attach CALL SITE in main() is followed by a cache
# reset before its item-path read-back loop
for att in ('ids_m, n_lab_total = attach_ordering_mode_labels(dataset',
            'ids_b, n_sup_b = attach_magmom_labels(dataset',
            'ids_s, n_elig_s = attach_seed_channels(dataset'):
    pos = src.index(att)
    nxt = src.index('_fresh_item_view(dataset)', pos)
    check(f'source: _fresh_item_view follows {att.split(" =")[0]} call '
          f'(within 200 chars)', 0 < nxt - pos < 200, nxt - pos)

print(f'ALL {len(PASS)} TRAIN-STEP CHECKS PASSED')
