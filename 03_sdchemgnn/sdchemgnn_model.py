"""sdchemgnn_model.py — the SD-ChemGNN model and the Xu-conditioned comparator.

Builds on site_decomposition_heads.MagneticMomentNet (capture machinery,
auxiliary mode heads and site-decomposition heads all inherited;
chemgnn_backbone.py stays untouched — in_channels is constructor-driven, which
is why the decoration is a feature-builder change rather than a model change).

Two model variants, selected by config.bs_arm:

  'seed_input'   SeedInputNet — SD-ChemGNN, the minimal delta. Input widened
                 95 -> 98 by the three decoration channels (appended by
                 spin_decoration.decorate_batch); heads and trunk are
                 byte-identical to the site-decomposition model. Gauge is
                 handled by the co-flip augmentation in the trainer, not in
                 the model.

  'xu_baseline'  XuConditionedNet — the published conditioning scheme
                 (Xu et al., PNAS 122(27) e2422973122, 2025) grafted onto OUR
                 trunk and OUR magnetization target; the Xu-conditioned model
                 of the paper's comparison:
                   e_s   = ( mlp(s, m, f) + mlp(-s, m, f) ) / 2      per node
                           (shared weights; EXACTLY invariant under a spin
                           flip — their symmetrized spin embedding)
                   trunk input = concat(base 95ch, e_s)   [95 + spin_embed_dim]
                   m_hat_i = mlp(z_i) * s_i               (their sign-projected
                           moment head: the output sign FOLLOWS the proposal by
                           construction; a flip stays possible via a negative
                           scalar; s_i = 0 -> m_hat_i = 0 on unproposed sites)
                 Under a global sign flip the trunk input is exactly invariant,
                 so z/h/r_hat/y_aux are invariant and m_hat co-flips EXACTLY —
                 the architectural realisation of the gauge structure (no
                 co-flip augmentation needed or used for this variant).
                 KNOWN, QUANTIFIED DEVIATION: the interstitial r_hat reads the
                 (flip-invariant) pooled h and therefore does NOT co-flip, so
                 the reconstruction's flip-asymmetry is bounded by 2|r_hat|
                 (~1e-4 scale on the trained site-decomposition head). Xu et
                 al. carry no residual term; ours stays because every
                 site-decomposition loss and head is kept fixed across the two
                 variants. The flip canary asserts the m_hat co-flip exactly
                 and REPORTS the r_hat asymmetry.

BOTH variants use the SIZE-CONSISTENT reconstruction:

    y_hat = sum_i(m_hat_i) / Z  +  r_hat        [r_hat read as uB PER F.U.]

Every term is replication-invariant, so the prediction is exactly
size-consistent — this supersedes the earlier y = (sum m + r) / Z, in which
r_hat was per-cell and provably could not scale (intensive pooled input).
The residual TARGET is correspondingly read per f.u. in the trainer
(resid_target / z_fu).

forward_b returns the site-decomposition dict plus 'm_sum' [G]; 'cell' stays
the per-cell equivalent m_sum + r_hat * z_fu for downstream compatibility.

Loading note: save_model_CEAL records the TRUNK constructor params only. For
'xu_baseline' the recorded in_channels is base + spin_embed_dim (the trunk
width); rebuilding for scoring must go through make_model_bs with the run's
config (the hyper json carries bs_arm / spin_embed_dim / base channels), not
through training_utils.load_ceal_model.

Naming contract: 'spin_mlp.' joins the trainer's wd_mlp prefix list next to
the inherited head prefixes. Checkpoints of the two variants are not
interchangeable, and neither loads into the undecorated site-decomposition
model or vice versa.
"""

import torch
import torch.nn as nn

from site_decomposition_heads import MagneticMomentNet, _backbone_kwargs

SEED_CHANNELS = 3          # sign, formal magnitude, flag (spin_decoration)


class _XView:
    """Light view of a PyG batch with x replaced — the ONLY attributes the
    shared MyCEALNetwork.forward and the capture machinery read are exposed
    and delegate to the wrapped batch (edge_index/edge_attr/batch/num_graphs
    stay shared; x is the swapped tensor)."""

    __slots__ = ('x', '_d')

    def __init__(self, data, x_new):
        self.x = x_new
        self._d = data

    def __getattr__(self, name):
        return getattr(self._d, name)


class _BSMomentNetBase(MagneticMomentNet):
    """Shared forward: captured trunk pass -> site moments (variant-specific)
    -> SIZE-CONSISTENT reconstruction y = m_sum / Z + r_hat (r_hat per f.u.)."""

    def _model_input(self, data):
        """Variant hook: what the trunk actually consumes (SD-ChemGNN: the
        decorated batch as-is; Xu-conditioned: base features + learned spin
        embedding)."""
        return data

    def _site_moments(self, z, data):
        """Variant hook: per-node signed moments from the node embeddings."""
        return self.node_moment_head(z).reshape(-1)

    def forward(self, data, return_aux=False):
        """Plain/aux paths routed through _model_input so BOTH variants accept the
        SAME decorated batches (parity checks and the mode-head evaluators keep
        working unchanged)."""
        return MagneticMomentNet.forward(self, self._model_input(data),
                                         return_aux=return_aux)

    def forward_b(self, data):
        pred, z, h = self._captured_forward(self._model_input(data))
        m_hat = self._site_moments(z, data)
        r_hat = self.interstitial_residual_head(h).reshape(-1)   # uB PER F.U.
        n_graphs = h.size(0)
        if hasattr(data, 'batch') and data.batch is not None:
            batch_vec = data.batch
        else:
            batch_vec = torch.zeros(m_hat.numel(), dtype=torch.long,
                                    device=m_hat.device)
        m_sum = torch.zeros(n_graphs, dtype=m_hat.dtype,
                            device=m_hat.device).index_add(0, batch_vec, m_hat)
        z_fu = getattr(data, 'z_fu', None)
        if z_fu is None:
            raise RuntimeError('forward_b: data.z_fu is missing — '
                               'attach_magmom_labels must run before training.')
        z_fu = z_fu.reshape(-1)
        if z_fu.numel() != n_graphs or bool((z_fu <= 0).any()):
            raise RuntimeError(f'forward_b: z_fu has {z_fu.numel()} entries '
                               f'for {n_graphs} graphs (or non-positive '
                               f'values).')
        y = m_sum / z_fu + r_hat                 # ── the §4.7 fix: exact ──
        return {'y': y, 'y_aux': pred.reshape(-1), 'm_hat': m_hat,
                'r_hat': r_hat, 'm_sum': m_sum,
                'cell': m_sum + r_hat * z_fu,    # per-cell equivalent (compat)
                'mode_logits': self.mode_head(h),
                'mode_vals': self.mode_value_head(h)}


class SeedInputNet(_BSMomentNetBase):
    """SD-ChemGNN: site-decomposition heads + trunk, input widened by the three seed
    channels. Nothing else differs from MagneticMomentNet (the decoration is
    the single attributed variable)."""


class XuConditionedNet(_BSMomentNetBase):
    """The Xu-conditioned comparator: Xu-style symmetrized spin embedding + seed-projected moment
    head on OUR trunk (docstring at module top)."""

    def __init__(self, *args, base_in_channels=None, spin_embed_dim=16,
                 **kwargs):
        if base_in_channels is None:
            raise TypeError('XuConditionedNet requires base_in_channels '
                            '(the UNDECORATED node-feature width, e.g. 95).')
        d_s = int(spin_embed_dim)
        # the trunk consumes concat(base, e_s), never the raw seed channels
        kwargs['in_channels'] = int(base_in_channels) + d_s
        super().__init__(*args, **kwargs)
        self.base_in_channels = int(base_in_channels)
        self.spin_embed_dim = d_s
        self.spin_mlp = nn.Sequential(
            nn.Linear(SEED_CHANNELS, d_s), nn.ReLU(), nn.Linear(d_s, d_s))

    def spin_embed(self, seed_cols):
        """e_s = (mlp(s, m, f) + mlp(-s, m, f)) / 2 — exactly flip-invariant
        (the magnitude and flag columns are sign-free and left unflipped)."""
        flipped = seed_cols.clone()
        flipped[:, 0] = -flipped[:, 0]
        return 0.5 * (self.spin_mlp(seed_cols) + self.spin_mlp(flipped))

    def _model_input(self, data):
        x = data.x
        want = self.base_in_channels + SEED_CHANNELS
        if x.size(1) != want:
            raise RuntimeError(
                f'XuConditionedNet: x has {x.size(1)} channels, expected '
                f'{want} (= base {self.base_in_channels} + {SEED_CHANNELS} '
                f'seed channels) — run spin_decoration.decorate_batch '
                f'first.')
        base = x[:, :self.base_in_channels]
        e_s = self.spin_embed(x[:, self.base_in_channels:])
        return _XView(data, torch.cat([base, e_s], dim=1))

    def _site_moments(self, z, data):
        sign = data.x[:, self.base_in_channels]          # the seed-sign column
        return self.node_moment_head(z).reshape(-1) * sign


def make_model_bs(in_channels, edge_dim, cfg):
    """Build the configured model variant. `in_channels` is the DECORATED width
    (base + 3) — what the trainer measures on a decorated batch; both label
    channels are required exactly as in the site-decomposition model."""
    if not getattr(cfg, 'magmom_sidetable', None):
        raise RuntimeError('config.magmom_sidetable is unset — SD-ChemGNN has no '
                           'meaning without node labels (seeds derive from '
                           'them).')
    if not getattr(cfg, 'ordering_sidetable', None):
        raise RuntimeError('config.ordering_sidetable is unset — it supplies '
                           'raw per-f.u. values (Z derivation) and the '
                           'diagnostic mode labels.')
    arm = str(getattr(cfg, 'bs_arm', 'seed_input')).lower()
    base = int(in_channels) - SEED_CHANNELS
    if base <= 0:
        raise RuntimeError(f'make_model_bs: decorated in_channels='
                           f'{in_channels} leaves no base features.')
    if arm == 'seed_input':
        return SeedInputNet(in_channels=int(in_channels), edge_dim=edge_dim,
                            **_backbone_kwargs(cfg))
    if arm == 'xu_baseline':
        return XuConditionedNet(
            base_in_channels=base,
            spin_embed_dim=int(getattr(cfg, 'xu_spin_embed_dim', 16)),
            edge_dim=edge_dim, **_backbone_kwargs(cfg))
    raise ValueError(f"bs_arm must be 'seed_input' or 'xu_baseline', "
                     f'got {arm!r}')
