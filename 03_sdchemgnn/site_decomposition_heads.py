"""
site_decomposition_heads.py — per-site moment decomposition heads.

MagneticMomentNet subclasses MyCEALNetwork (chemgnn_backbone.py is SHARED with the
formation-energy pipeline and stays untouched) and adds:

    node_moment_head            Linear(64->32) -> ReLU -> Linear(32->1) applied
                                to EVERY node embedding z_i  ->  m_hat [N x 1]
                                signed uB/atom (weights shared across atoms)
    interstitial_residual_head  same shape on the POOLED embedding h ->
                                r_hat [G x 1] uB/cell; output layer ZERO-INIT
                                (the node path must earn the total first)
    mode_head / mode_value_head the auxiliary ordering-diagnostic heads,
                                RETAINED as diagnostics (out of score)

    RECONSTRUCTION (the prediction path):
        cell_total = scatter_add(m_hat, batch) + r_hat
        y_hat      = cell_total / z_fu          (z_fu rides on the batch,
                                                 attached by site_moment_labels)
    post_mlp's scalar stays alive as y_aux (deep supervision + a live
    single-scalar comparator); selection/early-stop/metrics use y_hat only
    (trainer side). SD-ChemGNN (sdchemgnn_model.py) replaces this
    reconstruction by the size-consistent y = sum(m)/Z + r_hat with r_hat
    read per formula unit.

CAPTURE DESIGN: rather than re-implementing the backbone's forward, we run the
PARENT forward unmodified and capture the two tensors we need with hooks:
    h  (pooled, [G x 64])   forward PRE-hook on post_mlp (its input IS the
                            pooled representation)
    z  (per-node, [N x 64]) forward hooks on cealConvs[-1] AND norms[-1]; the
                            LAST node-shaped capture that fires BEFORE the
                            post_mlp pre-hook is taken as z. The choice is made
                            once, validated (shape [num_nodes, hidden], firing
                            order), and printed. If chemgnn_backbone.py applies an
                            activation/dropout between that module and the
                            pool, z is that module's output (a well-defined
                            trunk node representation) rather than the exact
                            pool input.
Because the parent forward runs UNMODIFIED, y_aux == MyCEALNetwork's output by
construction; the trainer still runs a one-batch eval-mode parity check
(plain forward vs forward_b()['y_aux']) at startup as a belt-and-suspenders.

Naming contract: 'mode_head.' / 'mode_value_head.' / 'node_moment_head.' /
'interstitial_residual_head.' are matched by build_optimizer's mlp_prefixes in
train_sdchemgnn.py (wd_mlp group). Everything else keeps its backbone name, so
save_model_CEAL works unchanged. Checkpoints of this model are NOT loadable
into the plain single-scalar backbone or vice versa.
"""

import torch
import torch.nn as nn

from chemgnn_backbone import MyCEALNetwork

_MODE_VALUE_BIAS_INIT = (0.05, 6.0, 6.0)   # train medians per mode

# MyCEALNetwork constructor arguments supplied by sdchemgnn_config.py.
_BACKBONE_KEYS = (
    'out_channels', 'aggregators', 'scalers', 'numLayers', 'towers',
    'pre_layers', 'post_layers', 'divide_input', 'aggMLP', 'aggMLP_factor',
    'dropout', 'norm',
)


def _backbone_kwargs(cfg):
    kwargs = {}
    for key in _BACKBONE_KEYS:
        try:
            kwargs[key] = getattr(cfg, key)
        except AttributeError as exc:
            raise AttributeError(
                f"sdchemgnn_config is missing '{key}', required to build "
                f'the model.') from exc
    return kwargs


def _as_tensor(out):
    """Hook outputs may be a tensor or a tuple whose first element is one."""
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)) and out and torch.is_tensor(out[0]):
        return out[0]
    return None


class MagneticMomentNet(MyCEALNetwork):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden = int(self.out_channels)
        # auxiliary diagnostic heads (retained; out of score)
        self.mode_head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 3))
        self.mode_value_head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 3))
        with torch.no_grad():
            self.mode_value_head[-1].bias.copy_(
                torch.tensor(_MODE_VALUE_BIAS_INIT, dtype=torch.float))
        # OPTION-B heads
        self.node_moment_head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
        self.interstitial_residual_head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
        with torch.no_grad():
            self.node_moment_head[-1].bias.zero_()
            # zero-init the WHOLE residual output layer: r_hat starts at exactly
            # 0 so the node path must earn the total first (anti-bypass)
            self.interstitial_residual_head[-1].weight.zero_()
            self.interstitial_residual_head[-1].bias.zero_()

        # ── capture machinery (hook pattern, extended with the pre-pool tap) ──
        self._cap_on = False
        self._cap = {}            # tag -> tensor
        self._cap_seq = []        # firing order of tags within one forward
        self._tap_choice = None   # 'z_norm' or 'z_conv', decided on first use
        self._tap_announced = False
        self.post_mlp.register_forward_pre_hook(self._grab_h)
        self.cealConvs[-1].register_forward_hook(self._make_grab('z_conv'))
        norms = getattr(self, 'norms', None)
        if norms is not None and len(norms) > 0:
            norms[-1].register_forward_hook(self._make_grab('z_norm'))

    # hooks -- no-ops unless _cap_on (default forward stays byte-identical)
    def _grab_h(self, _module, inputs):
        if self._cap_on:
            self._cap['h'] = inputs[0]
            self._cap_seq.append('h')

    def _make_grab(self, tag):
        def hook(_module, _inputs, output):
            if self._cap_on:
                t = _as_tensor(output)
                if t is not None:
                    self._cap[tag] = t
                    self._cap_seq.append(tag)
        return hook

    def _captured_forward(self, data):
        """Run the PARENT forward unmodified with captures on; return
        (pred, z, h) after validating shapes and firing order."""
        self._cap_on = True
        self._cap = {}
        self._cap_seq = []
        try:
            pred = MyCEALNetwork.forward(self, data)
        finally:
            self._cap_on = False
        cap, seq = self._cap, self._cap_seq
        self._cap = {}
        h = cap.get('h')
        if h is None:
            raise RuntimeError('MagneticMomentNet: post_mlp pre-hook never fired '
                               '— chemgnn_backbone.py structure changed; adapt the capture.')
        n_graphs = int(data.num_graphs) if hasattr(data, 'num_graphs') \
            else pred.reshape(-1).numel()
        if h.dim() != 2 or h.size(0) != n_graphs:
            raise RuntimeError(
                f'MagneticMomentNet: captured h has shape {tuple(h.shape)} for '
                f'{n_graphs} graphs — post_mlp does not receive the pooled '
                f'embedding in this chemgnn_backbone.py; adapt the capture point.')
        n_nodes = int(data.x.size(0))
        hidden = int(self.out_channels)
        if self._tap_choice is None:
            h_pos = seq.index('h')
            best = None
            for tag in ('z_norm', 'z_conv'):     # prefer the later-firing norm
                t = cap.get(tag)
                if (t is not None and tag in seq and seq.index(tag) < h_pos
                        and t.dim() == 2 and t.size(0) == n_nodes
                        and t.size(1) == hidden):
                    if best is None or seq.index(tag) > seq.index(best):
                        best = tag
            if best is None:
                raise RuntimeError(
                    f'MagneticMomentNet: no node-shaped [N x {hidden}] capture '
                    f'fired before post_mlp (seq={seq}, N={n_nodes}) — cannot '
                    f'locate the node embeddings in this chemgnn_backbone.py.')
            self._tap_choice = best
        z = cap.get(self._tap_choice)
        if z is None or z.dim() != 2 or z.size(0) != n_nodes or z.size(1) != hidden:
            raise RuntimeError(
                f'MagneticMomentNet: node tap {self._tap_choice!r} returned '
                f'shape {None if z is None else tuple(z.shape)} for '
                f'{n_nodes} nodes x {hidden}ch.')
        if not self._tap_announced:
            self._tap_announced = True
            print(f'    node tap = {self._tap_choice} (forward firing order: '
                  f'{" -> ".join(seq)})', flush=True)
        return pred, z, h

    def forward(self, data, return_aux=False):
        """Plain path (byte-identical backbone) or the aux path — kept so the
        mode-head evaluators (evaluate_modes / collect_modes) run unchanged."""
        if not return_aux:
            return MyCEALNetwork.forward(self, data)
        pred, _z, h = self._captured_forward(data)
        return pred, self.mode_head(h), self.mode_value_head(h)

    def forward_b(self, data):
        """OPTION-B forward: one trunk pass, every head, the reconstruction.

        Returns a dict:
            y      [G] reconstruction prediction  (THE prediction path)
            y_aux  [G] post_mlp scalar            (backbone scalar, deep supervision)
            m_hat  [N] signed site moments        (uB/atom)
            r_hat  [G] interstitial residual      (uB/cell)
            cell   [G] scatter_add(m_hat) + r_hat (uB/cell)
            mode_logits [G x 3], mode_vals [G x 3] (diagnostics)
        """
        pred, z, h = self._captured_forward(data)
        m_hat = self.node_moment_head(z).reshape(-1)
        r_hat = self.interstitial_residual_head(h).reshape(-1)
        n_graphs = h.size(0)
        if hasattr(data, 'batch') and data.batch is not None:
            batch_vec = data.batch
        else:
            batch_vec = torch.zeros(m_hat.numel(), dtype=torch.long,
                                    device=m_hat.device)
        cell = torch.zeros(n_graphs, dtype=m_hat.dtype, device=m_hat.device)
        cell = cell.index_add(0, batch_vec, m_hat) + r_hat
        z_fu = getattr(data, 'z_fu', None)
        if z_fu is None:
            raise RuntimeError('forward_b: data.z_fu is missing — '
                               'attach_magmom_labels must run before training.')
        z_fu = z_fu.reshape(-1)
        if z_fu.numel() != n_graphs or bool((z_fu <= 0).any()):
            raise RuntimeError(f'forward_b: z_fu has {z_fu.numel()} entries for '
                               f'{n_graphs} graphs (or non-positive values).')
        y = cell / z_fu
        return {'y': y, 'y_aux': pred.reshape(-1), 'm_hat': m_hat,
                'r_hat': r_hat, 'cell': cell,
                'mode_logits': self.mode_head(h),
                'mode_vals': self.mode_value_head(h)}


def make_model(in_channels, edge_dim, cfg):
    """Build the site-decomposition model. Both label channels are REQUIRED —
    this file has no headless fallback."""
    if not getattr(cfg, 'magmom_sidetable', None):
        raise RuntimeError('sdchemgnn_config.magmom_sidetable is unset — '
                           'the site-decomposition model has no meaning without node labels.')
    if not getattr(cfg, 'ordering_sidetable', None):
        raise RuntimeError('sdchemgnn_config.ordering_sidetable is unset '
                           '— it supplies raw per-f.u. values (Z derivation) '
                           'and the diagnostic mode labels.')
    return MagneticMomentNet(
        in_channels=in_channels,
        edge_dim=edge_dim,
        **_backbone_kwargs(cfg),
    )
