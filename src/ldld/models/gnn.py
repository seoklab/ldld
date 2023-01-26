import math
from functools import partial
from dataclasses import dataclass
from typing import Type

import dgl
from dgl import function as fn
from dgl.nn import functional as GF
from torch import nn

from . import Params


class GraphConv(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 dropout: float = 0.2,
                 bias: bool = True,
                 Activation: Type[nn.Module] = nn.ReLU):
        super().__init__()

        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.act = Activation()
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph: dgl.DGLHeteroGraph):
        h0 = graph.ndata["h"]

        # What does the following line do?
        graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "u"))
        h = self.linear(graph.ndata["u"])
        h = self.act(h)
        # Q) Why do we add h0?
        h = self.norm(h + h0)
        h = self.dropout(h)

        graph.ndata["h"] = h
        return graph


class MLP(nn.Sequential):
    def __init__(self, input_dim: int, hidden_dim: int,
                 bias: bool = True,
                 Activation: Type[nn.Module] = nn.ReLU):
        if hidden_dim < input_dim:
            raise ValueError(
                "hidden_dim must be greater than or equal to input_dim.")

        super().__init__(
            nn.Linear(input_dim, hidden_dim, bias=bias),
            Activation(),
            nn.Linear(hidden_dim, input_dim, bias=bias),
        )


class GraphIsomorphism(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 mlp_hidden_dim: int = None,
                 dropout: float = 0.2,
                 bias: bool = True,
                 Activation: Type[nn.Module] = nn.ReLU):
        super().__init__()

        if mlp_hidden_dim is None:
            mlp_hidden_dim = hidden_dim * 4

        self.mlp = MLP(hidden_dim, mlp_hidden_dim,
                       bias=bias, Activation=Activation)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph: dgl.DGLHeteroGraph):
        h0 = graph.ndata["h"]

        graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "u"))
        h = self.mlp(graph.ndata["u"])
        h = self.norm(h + h0)
        h = self.dropout(h)

        graph.ndata["h"] = h
        return graph


class GraphIsomorphismEdge(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 mlp_hidden_dim: int = None,
                 dropout: float = 0.2,
                 bias: bool = True,
                 Activation: Type[nn.Module] = nn.ReLU):
        super().__init__()

        if mlp_hidden_dim is None:
            mlp_hidden_dim = hidden_dim * 4

        self.mlp = MLP(hidden_dim, mlp_hidden_dim,
                       bias=bias, Activation=Activation)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph: dgl.DGLHeteroGraph):
        h0 = graph.ndata["h"]

        graph.update_all(fn.copy_u("h", "m_n"), fn.sum("m_n", "u_n"))
        graph.update_all(fn.copy_e("e_ij", "m_e"), fn.sum("m_e", "u_e"))
        u = graph.ndata["u_n"] + graph.ndata["u_e"]

        h = self.mlp(u)
        h = self.norm(h + h0)
        h = self.dropout(h)

        graph.ndata["h"] = h
        return graph


class GraphMultiheadAttention(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 num_heads: int = 4,
                 bias: bool = True):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden dimension must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.splitted_dim = hidden_dim // num_heads
        self._inv_sqrt_dim = 1 / math.sqrt(self.splitted_dim)

        def linear():
            return nn.Linear(hidden_dim, hidden_dim, bias=bias)

        self.e = linear()
        self.n_q = linear()
        self.n_k = linear()
        self.n_v = linear()
        self.n_h0 = linear()

    def forward(self, graph: dgl.DGLHeteroGraph):
        h0 = graph.ndata["h"]
        e0 = graph.edata["e_ij"]

        # Notations taken from pytorch_geometric (https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.TransformerConv)
        # q = W_3 x_i
        graph.ndata["q"] = self.n_q(h0).view(
            -1, self.num_heads, self.splitted_dim)
        # k = W_4 x_j
        graph.ndata["k"] = self.n_k(h0).view(
            -1, self.num_heads, self.splitted_dim)
        # v = W_2 x_j
        graph.ndata["v"] = self.n_v(h0).view(
            -1, self.num_heads, self.splitted_dim)
        # x_ij = W_6 e_ij
        graph.edata["x_ij"] = self.e(e0).view(
            -1, self.num_heads, self.splitted_dim)

        # k = k + x_ij
        graph.apply_edges(fn.u_add_e("k", "x_ij", "k"))
        # a = q^T @ m
        graph.apply_edges(fn.v_dot_e("q", "k", "a"))
        # a = softmax(a / sqrt(d_k))
        graph.edata["a"] = GF.edge_softmax(
            graph, graph.edata["a"] * self._inv_sqrt_dim)

        # v = v + x_ij
        graph.apply_edges(fn.u_add_e("v", "x_ij", "v"))
        # m = a * v
        graph.edata["m"] = graph.edata["a"] * graph.edata["v"]
        # h = sum(m)
        graph.update_all(fn.copy_e("m", "m"), fn.sum("m", "h"))

        # h = W_1 x_i + h
        h = graph.ndata["h"].view(-1, self.hidden_dim)
        h = self.n_h0(h0) + h

        graph.ndata["h"] = h
        return graph


class GraphTransformer(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 mlp_hidden_dim: int = None,
                 num_heads: int = 4,
                 bias: bool = True,
                 Activation: Type[nn.Module] = nn.ReLU,
                 dropout: float = 0.2):
        super().__init__()

        if mlp_hidden_dim is None:
            mlp_hidden_dim = hidden_dim * 4

        self.attention = GraphMultiheadAttention(
            hidden_dim, num_heads, bias=bias)
        self.a_dropout = nn.Dropout(dropout)
        self.a_norm = nn.LayerNorm(hidden_dim)

        self.mlp = MLP(
            hidden_dim, mlp_hidden_dim, bias=bias, Activation=Activation)
        self.l_dropout = nn.Dropout(dropout)
        self.l_norm = nn.LayerNorm(hidden_dim)

    def forward(self, graph: dgl.DGLHeteroGraph):
        graph = self.attention(graph)

        h0 = graph.ndata["h"]
        h1 = self.a_dropout(h0)
        h1 = self.a_norm(h1)

        h2 = self.l_dropout(self.mlp(h1))
        h = self.l_norm(h1 + h2)

        graph.ndata["h"] = h
        return graph


class GNNClassifier(nn.Module):
    @dataclass
    class Params(Params):
        # Model hyperparams
        model_type: str = "gat"
        num_layers: int = 4
        hidden_dim: int = 64
        num_heads: int = 4  # Only used for GraphTransformer
        dropout: float = 0.2
        bias: bool = True
        readout: str = "sum"

    def __init__(self,
                 params: "GNNClassifier.Params",
                 atom_feature_dim: int,
                 bond_feature_dim: int):
        super().__init__()

        self.params = params
        self.embedding_node = nn.Linear(
            atom_feature_dim, params.hidden_dim, bias=False)
        self.embedding_edge = nn.Linear(
            bond_feature_dim, params.hidden_dim, bias=False)

        # What is the following code doing?
        if params.model_type == "gcn":
            Gnn = GraphConv
        elif params.model_type == "gin":
            Gnn = GraphIsomorphism
        elif params.model_type == "gin_e":
            Gnn = GraphIsomorphismEdge
        elif params.model_type == "gat":
            Gnn = partial(GraphTransformer, num_heads=params.num_heads)
        else:
            raise ValueError("Unknown model type")

        # Why do we need to use partial instead of calling Gnn directly?
        gnn_factory = partial(Gnn, params.hidden_dim,
                              dropout=params.dropout, bias=params.bias,
                              Activation=params.Activation)
        self.gnn = nn.Sequential(
            *(gnn_factory() for _ in range(params.num_layers)))

        self.projection = nn.Linear(params.hidden_dim, 2, bias=False)

    def forward(self, graph: dgl.DGLHeteroGraph):
        h = self.embedding_node(graph.ndata["h"].float())
        e_ij = self.embedding_edge(graph.edata["e_ij"].float())
        graph.ndata["h"] = h
        graph.edata["e_ij"] = e_ij

        graph = self.gnn(graph)
        h = dgl.readout_nodes(graph, "h", op=self.params.readout)

        out = self.projection(h)
        return out
