# This is the implementation of CEAL layer

import torch
from torch.nn import ReLU, ModuleList, Sequential
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

from torch import Tensor

##########################################################################


class CEALConv(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        aggregators,
        scalers,
        edge_dim=None,
        towers=1,
        pre_layers=1,
        post_layers=1,
        divide_input=False,
        aggMLP=False,
        aggMLP_factor=0.5,
        **kwargs,
    ):
        """
        in_channels:  dimension of initial node embedding
        out_channels: dimension of node message (after linear transform)
        aggregators:  list of aggregation functions to apply (e.g. ['sum', 'mean', 'max'])
        scalers:      list of scalers to apply after aggregation (e.g. ['identity'])
        edge_dim:     edge feature dimensionality, or None if no edge features
        towers:       number of towers
        pre_layers:   number of MLP layers before aggregation
        post_layers:  number of MLP layers after aggregation
        divide_input: whether to split input features across towers
        aggMLP:       if True, use a per-aggregator MLP to compute adaptive weights;
                      if False, use a single learnable scalar weight per aggregator
        aggMLP_factor: hidden-layer size factor for the per-aggregator weight MLPs
        """

        # node_dim=0: node features are along dimension 0 of the x tensor,
        # i.e. x has shape (num_nodes, feature_dim).
        # aggr=None: we define our own aggregate() method below.
        super().__init__(node_dim=0, aggr=None)

        if divide_input:
            assert in_channels % towers == 0
        assert out_channels % towers == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregators = aggregators
        self.scalers = scalers
        self.edge_dim = edge_dim
        self.towers = towers
        self.divide_input = divide_input
        self.aggMLP = aggMLP
        self.aggMLP_factor = aggMLP_factor

        self.F_in = in_channels // towers if divide_input else in_channels
        self.F_out = self.out_channels // towers

        # Edge attribute encoder: projects edge features to node feature dimension
        # so they can be concatenated with node features in message().
        if self.edge_dim is not None:
            self.edge_encoder = Linear(edge_dim, self.F_in)

        # Pre-aggregation MLP per tower:
        #   input:  (3 * F_in) if edge features present, else (2 * F_in)
        #   output: F_in
        # Post-aggregation MLP per tower:
        #   input:  (num_aggregators * num_scalers + 1) * F_in
        #   output: F_out
        self.pre_nns = ModuleList()
        self.post_nns = ModuleList()
        for _ in range(towers):
            modules = [Linear((3 if edge_dim else 2) * self.F_in, self.F_in)]
            for _ in range(pre_layers - 1):
                modules += [ReLU()]
                modules += [Linear(self.F_in, self.F_in)]
            self.pre_nns.append(Sequential(*modules))

            in_channels = (len(aggregators) * len(scalers) + 1) * self.F_in
            modules = [Linear(in_channels, self.F_out)]
            for _ in range(post_layers - 1):
                modules += [ReLU()]
                modules += [Linear(self.F_out, self.F_out)]
            self.post_nns.append(Sequential(*modules))

        self.lin = Linear(out_channels, out_channels)

        if self.aggMLP:
            # One small MLP per aggregator to compute input-dependent weights.
            # Each MLP: F_in -> round(F_in * aggMLP_factor) -> F_in
            factor = aggMLP_factor
            self.mlp_w0 = Sequential(
                torch.nn.Linear(self.F_in, round(self.F_in * factor)),
                ReLU(),
                torch.nn.Linear(round(self.F_in * factor), self.F_in),
            )
            self.mlp_w1 = Sequential(
                torch.nn.Linear(self.F_in, round(self.F_in * factor)),
                ReLU(),
                torch.nn.Linear(round(self.F_in * factor), self.F_in),
            )
            self.mlp_w2 = Sequential(
                torch.nn.Linear(self.F_in, round(self.F_in * factor)),
                ReLU(),
                torch.nn.Linear(round(self.F_in * factor), self.F_in),
            )
            self.mlp_w3 = Sequential(
                torch.nn.Linear(self.F_in, round(self.F_in * factor)),
                ReLU(),
                torch.nn.Linear(round(self.F_in * factor), self.F_in),
            )
            self.mlp_w4 = Sequential(
                torch.nn.Linear(self.F_in, round(self.F_in * factor)),
                ReLU(),
                torch.nn.Linear(round(self.F_in * factor), self.F_in),
            )
        else:
            # One learnable scalar weight per aggregator.
            self.agg_weights = torch.nn.Parameter(torch.rand(len(aggregators)))

        self.reset_parameters()

    def reset_parameters(self):
        if self.edge_dim is not None:
            self.edge_encoder.reset_parameters()
        for layer in self.pre_nns:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        for layer in self.post_nns:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        self.lin.reset_parameters()
        if self.aggMLP:
            self.mlp_w0[0].reset_parameters()
            self.mlp_w0[2].reset_parameters()
            self.mlp_w1[0].reset_parameters()
            self.mlp_w1[2].reset_parameters()
            self.mlp_w2[0].reset_parameters()
            self.mlp_w2[2].reset_parameters()
            self.mlp_w3[0].reset_parameters()
            self.mlp_w3[2].reset_parameters()
            self.mlp_w4[0].reset_parameters()
            self.mlp_w4[2].reset_parameters()

    def forward(self, x, edge_index, batch, edge_attr=None):
        """
        x:          node feature tensor, shape (num_nodes, in_channels)
        edge_index: graph connectivity, shape (2, num_edges)
        batch:      batch vector, shape (num_nodes,)
        edge_attr:  edge feature tensor, shape (num_edges, edge_dim), or None
        """
        if self.divide_input:
            x = x.view(-1, self.towers, self.F_in)
        else:
            x = x.view(-1, 1, self.F_in).repeat(1, self.towers, 1)

        if self.aggMLP:
            out = self.propagate(edge_index, x=x, edge_attr=edge_attr, weights=None, batch=batch, size=None)
        else:
            myweights = self.agg_weights
            out = self.propagate(edge_index, x=x, edge_attr=edge_attr, weights=myweights, batch=batch, size=None)

        # Concatenate central node features with aggregated neighbourhood,
        # then pass through per-tower post-MLPs and final linear layer.
        out = torch.cat([x, out], dim=-1)
        outs = [nn(out[:, i]) for i, nn in enumerate(self.post_nns)]
        out = torch.cat(outs, dim=1)

        return self.lin(out)

    def message(self, x_i, x_j, edge_attr):
        """
        Constructs messages from central node i and neighbour node j.
        Concatenates [x_i, x_j] (and edge_attr if present), then passes
        through the per-tower pre-aggregation MLP.
        """
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
            edge_attr = edge_attr.view(-1, 1, self.F_in)
            edge_attr = edge_attr.repeat(1, self.towers, 1)
            h = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            h = torch.cat([x_i, x_j], dim=-1)

        hs = [nn(h[:, i]) for i, nn in enumerate(self.pre_nns)]
        return torch.stack(hs, dim=1)

    def aggregate(self, inputs, index, weights, batch, dim_size=None):
        """
        Applies multiple aggregators to incoming messages, each weighted
        either by a learned MLP (aggMLP=True) or a learned scalar (aggMLP=False).
        Results are then passed through the scaler pipeline.
        """
        outs = []

        if self.aggMLP:
            for aggregator in self.aggregators:
                if aggregator == "sum":
                    a = scatter(inputs, index, 0, None, dim_size, reduce="sum")
                    a_size = a.size()
                    w0 = self.mlp_w0(a[:, 0])
                    out = w0 * a[:, 0]
                    out = out.view(a_size[0], a_size[1], a_size[2])
                elif aggregator == "mean":
                    a = scatter(inputs, index, 0, None, dim_size, reduce="mean")
                    a_size = a.size()
                    w1 = self.mlp_w1(a[:, 0])
                    out = w1 * a[:, 0]
                    out = out.view(a_size[0], a_size[1], a_size[2])
                elif aggregator == "min":
                    a = scatter(inputs, index, 0, None, dim_size, reduce="min")
                    a_size = a.size()
                    w2 = self.mlp_w2(a[:, 0])
                    out = w2 * a[:, 0]
                    out = out.view(a_size[0], a_size[1], a_size[2])
                elif aggregator == "max":
                    a = scatter(inputs, index, 0, None, dim_size, reduce="max")
                    a_size = a.size()
                    w3 = self.mlp_w3(a[:, 0])
                    out = w3 * a[:, 0]
                    out = out.view(a_size[0], a_size[1], a_size[2])
                elif aggregator == "var" or aggregator == "std":
                    mean = scatter(inputs, index, 0, None, dim_size, reduce="mean")
                    mean_squares = scatter(inputs * inputs, index, 0, None, dim_size, reduce="mean")
                    a = mean_squares - mean * mean
                    a_size = a.size()
                    w4 = self.mlp_w4(a[:, 0])
                    out = w4 * a[:, 0]
                    out = out.view(a_size[0], a_size[1], a_size[2])
                    if aggregator == "std":
                        out = torch.sqrt(torch.relu(out) + 1e-5)
                else:
                    raise ValueError(f'Unknown aggregator "{aggregator}".')
                outs.append(out)
        else:
            for aggregator in self.aggregators:
                if aggregator == "sum":
                    out = weights[0] * scatter(inputs, index, 0, None, dim_size, reduce="sum")
                elif aggregator == "mean":
                    out = weights[1] * scatter(inputs, index, 0, None, dim_size, reduce="mean")
                elif aggregator == "min":
                    out = weights[2] * scatter(inputs, index, 0, None, dim_size, reduce="min")
                elif aggregator == "max":
                    out = weights[3] * scatter(inputs, index, 0, None, dim_size, reduce="max")
                elif aggregator == "var" or aggregator == "std":
                    mean = scatter(inputs, index, 0, None, dim_size, reduce="mean")
                    mean_squares = scatter(inputs * inputs, index, 0, None, dim_size, reduce="mean")
                    out = weights[4] * (mean_squares - mean * mean)
                    if aggregator == "std":
                        out = torch.sqrt(torch.relu(out) + 1e-5)
                else:
                    raise ValueError(f'Unknown aggregator "{aggregator}".')
                outs.append(out)

        out = torch.cat(outs, dim=-1)

        outs = []
        for scaler in self.scalers:
            if scaler == "identity":
                pass
            else:
                raise ValueError(f'Unknown scaler "{scaler}".')
            outs.append(out)

        return torch.cat(outs, dim=-1)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, towers={self.towers}, "
            f"edge_dim={self.edge_dim})"
        )

