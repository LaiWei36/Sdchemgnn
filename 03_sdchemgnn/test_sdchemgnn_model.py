"""Offline tests for sdchemgnn_model with a mocked shared chemgnn_backbone.py
Covers, per model variant:
  SeedInputNet  — shapes, the SIZE-CONSISTENT per-f.u. reconstruction
                  y = m_sum/Z + r_hat, zero-init r_hat, parity, m_sum/cell
                  keys, replication invariance of the reconstruction algebra;
  XuConditionedNet — decorated-width guard, EXACT flip-invariance of the
                  symmetrized spin embedding, EXACT m_hat co-flip + y_aux /
                  r_hat invariance under a global seed flip, projection zeros
                  on unseeded sites, exact AFM compensation on a 2-site toy,
                  'spin_mlp.' optimizer-prefix contract;
  make_model_bs — variant dispatch + config guards."""
import sys
import types

import torch
import torch.nn as nn

PASS = []


def check(name, cond, got=None):
    assert cond, f'FAIL: {name}' + (f' (got {got})' if got is not None else '')
    PASS.append(name)


def make_fake_model_module():
    fake = types.ModuleType('chemgnn_backbone')

    class FakeConv(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.lin = nn.Linear(cin, cout)

        def forward(self, x, edge_index=None, edge_attr=None):
            return self.lin(x)

    class MyCEALNetwork(nn.Module):
        def __init__(self, in_channels, edge_dim, out_channels=64,
                     aggregators=None, scalers=None, numLayers=3, towers=1,
                     pre_layers=1, post_layers=1, divide_input=False,
                     aggMLP=True, aggMLP_factor=0.125, dropout=0.1,
                     norm='layer'):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.pre_mlp = nn.Sequential(
                nn.Linear(in_channels, 32), nn.ReLU(),
                nn.Linear(32, in_channels))
            self.ceal_1 = FakeConv(in_channels, out_channels)
            self.norm_1 = nn.LayerNorm(out_channels)
            self.cealConvs = nn.ModuleList(
                [FakeConv(out_channels, out_channels)
                 for _ in range(numLayers - 1)])
            self.norms = nn.ModuleList(
                [nn.LayerNorm(out_channels) for _ in range(numLayers - 1)])
            self.dropout = nn.Dropout(dropout)
            self.post_mlp = nn.Sequential(
                nn.Linear(out_channels, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, data):
            x = self.pre_mlp(data.x)
            x = torch.relu(self.norm_1(self.ceal_1(x, data.edge_index, None)))
            for i, conv in enumerate(self.cealConvs):
                x = self.norms[i](conv(x, data.edge_index, None))
                x = self.dropout(x)
            G = int(data.batch.max()) + 1
            cnt = torch.zeros(G).index_add_(0, data.batch,
                                            torch.ones(x.size(0)))
            pooled = torch.zeros(G, x.size(1)).index_add_(0, data.batch, x)
            pooled = pooled / cnt.clamp(min=1).unsqueeze(1)
            return self.post_mlp(pooled)

    fake.MyCEALNetwork = MyCEALNetwork
    return fake


sys.modules['chemgnn_backbone'] = make_fake_model_module()
for _m in ('site_decomposition_heads', 'sdchemgnn_model'):
    sys.modules.pop(_m, None)
import sdchemgnn_model as mbs                            # noqa: E402

BASE = 9
BK = dict(edge_dim=5, out_channels=64, aggregators=['sum'],
          scalers=['identity'], numLayers=3, towers=1, pre_layers=1,
          post_layers=1, divide_input=False, aggMLP=True, aggMLP_factor=0.125,
          dropout=0.1, norm='layer')


def decorated_batch(signs, n_nodes=(4, 6), z_fu=(2.0, 3.0), seed=7):
    torch.manual_seed(seed)
    N = sum(n_nodes)
    bv = []
    for gi, n in enumerate(n_nodes):
        bv.extend([gi] * n)
    x = torch.randn(N, BASE)
    sign = torch.tensor(signs, dtype=torch.float)
    fmag = torch.where(sign.abs() > 0, torch.full((N,), 5.0 / 7.0),
                       torch.zeros(N))
    flag = torch.ones(N)
    return types.SimpleNamespace(
        x=torch.cat([x, sign.unsqueeze(1), fmag.unsqueeze(1),
                     flag.unsqueeze(1)], dim=1),
        edge_index=None, batch=torch.tensor(bv), num_graphs=len(n_nodes),
        z_fu=torch.tensor(list(z_fu)))


SIGNS = [1, -1, 1, 0, 1, 1, 0, 0, -1, 0]

# ── SD-ChemGNN: SeedInputNet ────────────────────────────────────────────────────────
torch.manual_seed(7)
net = mbs.SeedInputNet(in_channels=BASE + 3, **BK)
net.eval()
data = decorated_batch(SIGNS)
y_plain = net(data)
check('SeedInputNet plain forward shape [G,1]', tuple(y_plain.shape) == (2, 1))
out = net.forward_b(data)
check('SeedInputNet m_hat [N]', out['m_hat'].shape == (10,))
check('SeedInputNet r_hat zero-init -> exactly 0',
      float(out['r_hat'].abs().max()) == 0.0)
check('SeedInputNet parity y_aux == plain forward',
      torch.allclose(out['y_aux'], y_plain.reshape(-1), atol=1e-6))
# SIZE-CONSISTENT reconstruction: y = m_sum / Z + r_hat  (r_hat per f.u.)
ms0 = float(out['m_hat'][:4].sum())
ms1 = float(out['m_hat'][4:].sum())
check('SeedInputNet m_sum key', abs(float(out['m_sum'][0]) - ms0) < 1e-5
      and abs(float(out['m_sum'][1]) - ms1) < 1e-5)
check('SeedInputNet reconstruction y = m_sum/Z + r_hat',
      abs(float(out['y'][0]) - (ms0 / 2.0 + float(out['r_hat'][0]))) < 1e-5
      and abs(float(out['y'][1]) - (ms1 / 3.0 + float(out['r_hat'][1]))) < 1e-5)
check('SeedInputNet cell key = m_sum + r_hat*Z',
      torch.allclose(out['cell'], out['m_sum'] + out['r_hat']
                     * data.z_fu, atol=1e-5))
# replication invariance of the ALGEBRA: with nonzero r_hat, doubling every
# site and Z leaves y unchanged (per-f.u. residual does not dilute)
with torch.no_grad():
    net.interstitial_residual_head[-1].weight.normal_(std=0.05)
    net.interstitial_residual_head[-1].bias.fill_(0.3)
o1 = net.forward_b(decorated_batch(SIGNS, seed=7))
m_sum_1, r_1, z_1 = float(o1['m_sum'][0]), float(o1['r_hat'][0]), 2.0
y_repl_1 = m_sum_1 / z_1 + r_1
y_repl_2 = (2 * m_sum_1) / (2 * z_1) + r_1     # identical pooled h -> same r
check('per-f.u. r_hat is replication-invariant (exact algebra)',
      abs(y_repl_1 - y_repl_2) < 1e-12)
check('old per-cell form would NOT be ((2m+r)/2Z shifts by r/2Z)',
      abs((m_sum_1 + r_1) / z_1 - (2 * m_sum_1 + r_1) / (2 * z_1))
      > 1e-9 or abs(r_1) < 1e-12)

# gradient flows below the pool; post_mlp untouched by the node loss
net.train()
out = net.forward_b(decorated_batch(SIGNS))
out['m_hat'].abs().mean().backward()
conv_g = [p.grad.abs().sum() for p in net.cealConvs.parameters()
          if p.grad is not None]
check('SeedInputNet node loss backprops into convs',
      len(conv_g) > 0 and float(sum(conv_g)) > 0)
post_g = [p.grad for p in net.post_mlp.parameters()]
check('SeedInputNet node loss does NOT touch post_mlp',
      all(g2 is None or float(g2.abs().sum()) == 0 for g2 in post_g))

# ── Xu-conditioned comparator: XuConditionedNet ────────────────────────────────────────────────────
torch.manual_seed(13)
xu = mbs.XuConditionedNet(base_in_channels=BASE, spin_embed_dim=8, **BK)
xu.eval()
check('XuConditionedNet trunk width = base + embed', xu.in_channels == BASE + 8)
check('XuConditionedNet spin_mlp prefix present (optimizer contract)',
      any(n.startswith('spin_mlp.') for n, _ in xu.named_parameters()))

bad = decorated_batch(SIGNS)
bad.x = bad.x[:, :BASE]
try:
    xu.forward_b(bad)
    raise SystemExit('FAIL: undecorated width did not raise')
except RuntimeError as exc:
    check('XuConditionedNet decorated-width guard raises', 'decorate_batch' in str(exc))

# exact e_s invariance under the sign flip
cols = torch.randn(16, 3)
cols[:, 0] = torch.sign(cols[:, 0])
flipped = cols.clone()
flipped[:, 0] = -flipped[:, 0]
check('XuConditionedNet spin embedding EXACTLY flip-invariant',
      torch.allclose(xu.spin_embed(cols), xu.spin_embed(flipped), atol=0.0))

# global flip: m_hat co-flips exactly; y_aux and r_hat invariant
with torch.no_grad():                       # make r_hat nonzero so the test bites
    xu.interstitial_residual_head[-1].weight.normal_(std=0.05)
    xu.interstitial_residual_head[-1].bias.fill_(0.2)
d1 = decorated_batch(SIGNS, seed=21)
d2 = decorated_batch(SIGNS, seed=21)
d2.x[:, BASE] = -d2.x[:, BASE]
o1, o2 = xu.forward_b(d1), xu.forward_b(d2)
check('XuConditionedNet global flip co-flips m_hat EXACTLY',
      torch.allclose(o2['m_hat'], -o1['m_hat'], atol=1e-6))
check('XuConditionedNet y_aux flip-invariant',
      torch.allclose(o2['y_aux'], o1['y_aux'], atol=1e-6))
check('XuConditionedNet r_hat flip-invariant (the quantified deviation: y flips only '
      'through m_sum)',
      torch.allclose(o2['r_hat'], o1['r_hat'], atol=1e-6))
check('XuConditionedNet m_sum co-flips', torch.allclose(o2['m_sum'], -o1['m_sum'],
                                            atol=1e-6))

# projection zeros unseeded sites
zero_sites = [i for i, s in enumerate(SIGNS) if s == 0]
check('XuConditionedNet unseeded sites emit exactly 0',
      float(o1['m_hat'][zero_sites].abs().max()) == 0.0)

# 2-site AFM toy: identical features, signs (+1,-1) -> exact compensation
torch.manual_seed(3)
x0 = torch.randn(BASE)
toy = types.SimpleNamespace(
    x=torch.cat([torch.stack([x0, x0]),
                 torch.tensor([[1.0], [-1.0]]),
                 torch.full((2, 1), 5.0 / 7.0),
                 torch.ones(2, 1)], dim=1),
    edge_index=None, batch=torch.tensor([0, 0]), num_graphs=1,
    z_fu=torch.tensor([1.0]))
ot = xu.forward_b(toy)
check('XuConditionedNet AFM toy: exact compensation (sum m_hat == 0)',
      abs(float(ot['m_hat'].sum())) < 1e-6)
check('XuConditionedNet AFM toy: symmetry broken (opposite nonzero moments)',
      float(ot['m_hat'].abs().min()) > 0
      and abs(float(ot['m_hat'][0] + ot['m_hat'][1])) < 1e-6)

# ── make_model_bs dispatch + guards ────────────────────────────────────────────
class CfgOK:
    magmom_sidetable = './x.jsonl'
    ordering_sidetable = './y.csv'
    bs_arm = 'seed_input'
    xu_spin_embed_dim = 8
    out_channels = 64; aggregators = ['sum']; scalers = ['identity']
    numLayers = 3; towers = 1; pre_layers = 1; post_layers = 1
    divide_input = False; aggMLP = True; aggMLP_factor = 0.125
    dropout = 0.1; norm = 'layer'


class CfgXu(CfgOK):
    bs_arm = 'xu_baseline'


class CfgNoMagmom(CfgOK):
    magmom_sidetable = None


class CfgBadArm(CfgOK):
    bs_arm = 'nope'


m1 = mbs.make_model_bs(BASE + 3, 5, CfgOK)
check('make_model_bs seed_input', isinstance(m1, mbs.SeedInputNet)
      and m1.in_channels == BASE + 3)
m2 = mbs.make_model_bs(BASE + 3, 5, CfgXu)
check('make_model_bs xu_baseline (trunk = base + embed)',
      isinstance(m2, mbs.XuConditionedNet) and m2.in_channels == BASE + 8
      and m2.base_in_channels == BASE)
try:
    mbs.make_model_bs(BASE + 3, 5, CfgNoMagmom)
    raise SystemExit('FAIL: missing magmom_sidetable did not raise')
except RuntimeError as exc:
    check('make_model_bs magmom guard', 'magmom_sidetable' in str(exc))
try:
    mbs.make_model_bs(BASE + 3, 5, CfgBadArm)
    raise SystemExit('FAIL: bad variant did not raise')
except ValueError as exc:
    check('make_model_bs variant guard', 'bs_arm' in str(exc))

print(f'ALL {len(PASS)} MODEL CHECKS PASSED')
