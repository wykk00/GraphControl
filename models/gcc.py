from typing import Any, Mapping
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
import torch_geometric
from utils.register import register


def change_params_key(params):
    """
    Change GCC source parameters keys
    """
    for key in list(params.keys()):
        sp = key.split('.')
        if len(sp) > 3 and sp[3] == 'apply_func':
            sp[3] = 'nn'
            str = '.'.join(sp)
            params[str] = params[key]
            params.pop(key)
        if sp[0] == 'set2set':
            params.pop(key)


@register.model_register
class GCC(nn.Module):
    """
    MPNN from
    `Neural Message Passing for Quantum Chemistry <https://arxiv.org/abs/1704.01212>`__

    Parameters
    ----------
    node_input_dim : int
        Dimension of input node feature, default to be 15.
    edge_input_dim : int
        Dimension of input edge feature, default to be 15.
    output_dim : int
        Dimension of prediction, default to be 12.
    node_hidden_dim : int
        Dimension of node feature in hidden layers, default to be 64.
    edge_hidden_dim : int
        Dimension of edge feature in hidden layers, default to be 128.
    num_step_message_passing : int
        Number of message passing steps, default to be 6.
    num_step_set2set : int
        Number of set2set steps
    num_layer_set2set : int
        Number of set2set layers
    """

    def __init__(
        self,
        positional_embedding_size=32,
        max_node_freq=8,
        max_edge_freq=8,
        max_degree=128,
        freq_embedding_size=32,
        degree_embedding_size=32,
        output_dim=32,
        node_hidden_dim=32,
        edge_hidden_dim=32,
        num_layers=6,
        num_heads=4,
        num_step_set2set=6,
        num_layer_set2set=3,
        norm=False,
        gnn_model="mpnn",
        degree_input=False,
        lstm_as_gate=False,
        num_classes=10
    ):
        super(GCC, self).__init__()

        if degree_input:
            node_input_dim = positional_embedding_size + degree_embedding_size + 1
        else:
            node_input_dim = positional_embedding_size + 1
        # node_input_dim = (
        #     positional_embedding_size + freq_embedding_size + degree_embedding_size + 3
        # )
        edge_input_dim = freq_embedding_size + 1
        self.gnn = UnsupervisedGIN(
            num_layers=num_layers,
            num_mlp_layers=2,
            input_dim=node_input_dim,
            hidden_dim=node_hidden_dim,
            output_dim=output_dim,
            final_dropout=0.5,
            learn_eps=False,
            graph_pooling_type="sum",
            neighbor_pooling_type="sum",
            use_selayer=False,
        )
        self.gnn_model = gnn_model

        self.max_node_freq = max_node_freq
        self.max_edge_freq = max_edge_freq
        self.max_degree = max_degree
        self.degree_input = degree_input

       
        if degree_input:
            self.degree_embedding = nn.Embedding(
                num_embeddings=max_degree + 1, embedding_dim=degree_embedding_size
            )

        self.lin_readout = nn.Sequential(
            nn.Linear(2 * node_hidden_dim, node_hidden_dim),
            nn.ReLU(),
            nn.Linear(node_hidden_dim, output_dim),
        )
        self.norm = norm
    
    def forward(self, x, edge_index, edge_weight=None, frozen=False, **kwargs):
        raise NotImplementedError('Please use --subsampling')
    
    def forward_subgraph(self, x, edge_index, batch, root_n_id, edge_weight=None, frozen=False, **kwargs):
        """Predict molecule labels

        Parameters
        ----------
        g : DGLGraph
            Input DGLGraph for molecule(s)
        n_feat : tensor of dtype float32 and shape (B1, D1)
            Node features. B1 for number of nodes and D1 for
            the node feature size.
        e_feat : tensor of dtype float32 and shape (B2, D2)
            Edge features. B2 for number of edges and D2 for
            the edge feature size.

        Returns
        -------
        res : Predicted labels
        """
        # nfreq = g.ndata["nfreq"]
        if self.degree_input:
            # device = g.ndata["seed"].device
            device = x.device
            degrees = torch_geometric.utils.degree(edge_index[0]).long().to(device)
            ego_indicator = torch.zeros(x.shape[0]).bool().to(device)
            ego_indicator[root_n_id] = True

            n_feat = torch.cat(
                (
                    x,
                    self.degree_embedding(degrees.clamp(0, self.max_degree)),
                    ego_indicator.unsqueeze(1).float(),
                ),
                dim=-1,
            )
        else:
            n_feat = torch.cat(
                (
                    x,
                ),
                dim=-1,
            )

        e_feat = None

        x, all_outputs = self.gnn(n_feat, edge_index, batch)
       
        if self.norm:
            x = F.normalize(x, p=2, dim=-1, eps=1e-5)

        return x


class SELayer(nn.Module):
    """Squeeze-and-excitation networks"""

    def __init__(self, in_channels, se_channels):
        super(SELayer, self).__init__()

        self.in_channels = in_channels
        self.se_channels = se_channels

        self.encoder_decoder = nn.Sequential(
            nn.Linear(in_channels, se_channels),
            nn.ELU(),
            nn.Linear(se_channels, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """"""
        # Aggregate input representation
        x_global = torch.mean(x, dim=0)
        # Compute reweighting vector s
        s = self.encoder_decoder(x_global)

        return x * s


class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""

    def __init__(self, mlp, use_selayer):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = (
            SELayer(self.mlp.output_dim, int(np.sqrt(self.mlp.output_dim)))
            if use_selayer
            else nn.BatchNorm1d(self.mlp.output_dim)
        )

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, use_selayer):
        """MLP layers construction

        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction

        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(
                    SELayer(hidden_dim, int(np.sqrt(hidden_dim)))
                    if use_selayer
                    else nn.BatchNorm1d(hidden_dim)
                )

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class UnsupervisedGIN(nn.Module):
    """GIN model"""

    def __init__(
        self,
        num_layers,
        num_mlp_layers,
        input_dim,
        hidden_dim,
        output_dim,
        final_dropout,
        learn_eps,
        graph_pooling_type,
        neighbor_pooling_type,
        use_selayer,
    ):
        """model parameters setting

        Paramters
        ---------
        num_layers: int
            The number of linear layers in the neural network
        num_mlp_layers: int
            The number of linear layers in mlps
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        final_dropout: float
            dropout ratio on the final linear layer
        learn_eps: boolean
            If True, learn epsilon to distinguish center nodes from neighbors
            If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: str
            how to aggregate neighbors (sum, mean, or max)
        graph_pooling_type: str
            how to aggregate entire nodes in a graph (sum, mean or max)

        """
        super(UnsupervisedGIN, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = MLP(
                    num_mlp_layers, input_dim, hidden_dim, hidden_dim, use_selayer
                )
            else:
                mlp = MLP(
                    num_mlp_layers, hidden_dim, hidden_dim, hidden_dim, use_selayer
                )

            self.ginlayers.append(
                GINConv(
                    ApplyNodeFunc(mlp, use_selayer),
                    0,
                    self.learn_eps,
                )
            )
            self.batch_norms.append(
                SELayer(hidden_dim, int(np.sqrt(hidden_dim)))
                if use_selayer
                else nn.BatchNorm1d(hidden_dim)
            )

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(
                    nn.Linear(hidden_dim, output_dim))

        self.drop = nn.Dropout(final_dropout)

        if graph_pooling_type == "sum":
            self.pool = global_add_pool
        elif graph_pooling_type == "mean":
            self.pool = global_mean_pool
        elif graph_pooling_type == "max":
            self.pool = global_max_pool
        else:
            raise NotImplementedError

    def forward(self, x, edge_index, batch):
        # list of hidden representation at each layer (including input)
        hidden_rep = [x]
        h = x
        for i in range(self.num_layers - 1):
            h = self.ginlayers[i](h, edge_index)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)

        score_over_layer = 0

        # perform pooling over all nodes in each graph in every layer
        all_outputs = []
        for i, h in list(enumerate(hidden_rep)):
            pooled_h = self.pool(h, batch)
            all_outputs.append(pooled_h)
            score_over_layer += self.drop(self.linears_prediction[i](pooled_h))

        return score_over_layer, all_outputs[1:]

