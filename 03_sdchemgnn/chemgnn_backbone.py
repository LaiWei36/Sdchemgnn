import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential, ModuleList, LayerNorm
from torch_geometric.nn import GraphNorm, global_mean_pool, global_add_pool

from ceal_layer import CEALConv


class MyCEALNetwork(torch.nn.Module):
    """
    Multi-layer CEAL network for graph-level regression.

    Normalization
    -------------
    Norm has NO train/eval running-statistics distinction (behaves identically in
    train() and eval()), selected via `norm`:
        'layer' -> torch.nn.LayerNorm            (per-node over the feature dim)  [default]
        'graph' -> torch_geometric.nn.GraphNorm  (per-graph; centers each graph,
                   which collapses a mean-pool readout — avoid with global_mean_pool)

    Dropout  (IMPORTANT — placement matters)
    ----------------------------------------
    Dropout is applied INSIDE each residual branch, on the node features, BEFORE
    pooling. global_mean_pool then averages the dropout noise across all of a
    graph's atoms, so the pooled graph vector is stable and the network does not
    co-adapt to the mask.

    Do NOT move dropout to the readout (after pooling, on the single pooled
    vector). Doing so makes the model produce good predictions ONLY with dropout
    active and collapse to the predict-the-mean baseline at inference — train MAE
    looks fine while clean eval (and validation) sit at baseline. The residual
    stream (the skip path) and the stem are left dropout-free so the gradient
    highway stays clean.

    Recommended: set dropout ~0.1 in config for light regularization; 0.0 disables.

    Residual connections
    --------------------
    Each CEAL layer after the first is wrapped in an additive, parameter-free
    residual:  x <- x + Dropout(ReLU(Norm(CEALConv(x)))).  The first layer is a
    stem (in_channels -> out_channels, no residual). Residuals (and therefore the
    per-branch dropout) apply to layers 2 .. numLayers.

    Args:
        in_channels  (int):   Input node feature dimension.
        out_channels (int):   Hidden and output feature dimension for each CEAL layer.
        aggregators  (list):  Aggregation functions, e.g. ['sum', 'mean', 'max'].
        scalers      (list):  Scalers to apply after aggregation, e.g. ['identity'].
        numLayers    (int):   Total number of CEAL layers (default: 2). Residuals
                              (and per-branch dropout) require numLayers >= 2.
        edge_dim     (int):   Edge feature dimension, or None if no edge features.
        towers       (int):   Number of towers in each CEAL layer (default: 1).
        pre_layers   (int):   Pre-aggregation MLP layers in each CEAL layer (default: 1).
        post_layers  (int):   Post-aggregation MLP layers in each CEAL layer (default: 1).
        divide_input (bool):  Whether to split input features across towers (default: False).
        aggMLP       (bool):  If True, use per-aggregator MLP weights;
                              if False, use learnable scalar weights (default: False).
        aggMLP_factor (float): Hidden-layer size factor for weight MLPs (default: 0.5).
        dropout      (float): Dropout prob applied in each residual branch (default: 0.1).
        norm         (str):   Normalization type, 'layer' or 'graph' (default: 'layer').
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        aggregators,
        scalers,
        numLayers=2,
        edge_dim=None,
        towers=1,
        pre_layers=1,
        post_layers=1,
        divide_input=False,
        aggMLP=False,
        aggMLP_factor=0.5,
        dropout=0.1,
        norm='layer',
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregators = aggregators
        self.scalers = scalers
        self.numLayers = numLayers
        self.edge_dim = edge_dim
        self.towers = towers
        self.pre_layers = pre_layers
        self.post_layers = post_layers
        self.divide_input = divide_input
        self.aggMLP = aggMLP
        self.aggMLP_factor = aggMLP_factor
        self.dropout = dropout
        self.norm = norm.lower()
        if self.norm not in ('graph', 'layer'):
            raise ValueError(f"norm must be 'graph' or 'layer', got '{norm}'.")

        # First CEAL layer (stem): in_channels -> out_channels  (no residual).
        self.ceal_1 = CEALConv(
            in_channels=in_channels,
            out_channels=out_channels,
            aggregators=aggregators,
            scalers=scalers,
            edge_dim=edge_dim,
            towers=towers,
            pre_layers=pre_layers,
            post_layers=post_layers,
            divide_input=divide_input,
            aggMLP=aggMLP,
            aggMLP_factor=aggMLP_factor,
        )
        self.norm_1 = self._make_norm(out_channels)

        # Remaining CEAL layers: out_channels -> out_channels (each gets a residual).
        self.cealConvs = ModuleList()
        self.norms = ModuleList()
        for _ in range(1, numLayers):
            self.cealConvs.append(
                CEALConv(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    aggregators=aggregators,
                    scalers=scalers,
                    edge_dim=edge_dim,
                    towers=towers,
                    pre_layers=pre_layers,
                    post_layers=post_layers,
                    divide_input=divide_input,
                    aggMLP=aggMLP,
                    aggMLP_factor=aggMLP_factor,
                )
            )
            self.norms.append(self._make_norm(out_channels))

        # Pre-processing MLP (optional, disabled in forward by default).
        self.pre_mlp = Sequential(
            Linear(in_channels, out_channels // 2),
            ReLU(),
            Linear(out_channels // 2, in_channels),
        )

        # Output MLP: out_channels -> out_channels//2 -> 1
        self.post_mlp = Sequential(
            Linear(out_channels, out_channels // 2),
            ReLU(),
            Linear(out_channels // 2, 1),
        )

    def _make_norm(self, dim):
        """Create a normalization layer of the configured type."""
        if self.norm == 'graph':
            return GraphNorm(dim)
        return LayerNorm(dim)

    def _apply_norm(self, norm_layer, x, batch):
        """Apply a norm layer, passing the batch vector only for GraphNorm."""
        if self.norm == 'graph':
            return norm_layer(x, batch)
        return norm_layer(x)

    def forward(self, batch_data):
        """
        Args:
            batch_data: a PyG Data/Batch object with attributes:
                        x (node features), edge_index, edge_attr (optional), batch.
        Returns:
            out: graph-level predictions, shape (num_graphs, 1).
        """
        x, edge_index, edge_attr = batch_data.x, batch_data.edge_index, batch_data.edge_attr
        batch = batch_data.batch

        # Optionally transform node features before message passing.
        # x = self.pre_mlp(x)

        # ── Stem: first CEAL layer (no residual, no dropout) ─────────────────────
        # Kept clean so the residual stream starts from an un-perturbed state.
        x = self.ceal_1(x, edge_index, batch, edge_attr)
        x = self._apply_norm(self.norm_1, x, batch)
        x = F.relu(x)

        # ── Residual blocks: layers 2 .. numLayers ───────────────────────────────
        # Dropout is applied to the BRANCH (node features, pre-pooling). mean-pool
        # averages the noise across a graph's atoms, so the readout stays stable.
        # The skip path (x) is left clean.
        for cealconv, norm in zip(self.cealConvs, self.norms):
            h = cealconv(x, edge_index, batch, edge_attr)
            h = self._apply_norm(norm, h, batch)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h  # residual connection (skip path stays clean)

        # ── Readout: pool, then output MLP.  NO dropout here. ────────────────────
        # x = global_add_pool(x, batch)   # alternative: sum pooling
        x = global_mean_pool(x, batch)
        return self.post_mlp(x)

