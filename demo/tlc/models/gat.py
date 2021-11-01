"""Graph Attention Network model
ref: Velickovic et al. 2018, Graph Attention Networks"""

from typing import Dict

import torch
from torch import nn

from ..device import get_torch_device
from ..structures.trees import DataTree
from .prototypes import TreeEmbedder


class GATBase(TreeEmbedder):
    """Base module of a graph-attention-network model"""

    def __init__(self) -> None:
        super().__init__()
        self.cache: Dict[DataTree, torch.Tensor] = dict()

    def _node_forward(self, h: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def forward(self, node: DataTree) -> torch.Tensor:
        if node in self.cache:
            return self.cache[node]

        neighbors = [node.feature_vector.float()]
        neighbors.extend(node.feature_vector.float() for node in node.neighbors)

        h = torch.stack(neighbors, dim=0)
        h_prime = self._node_forward(h)

        self.cache[node] = h_prime
        return h_prime

    def reset(self) -> None:
        self.cache = dict()


class GATModule(GATBase):
    """Base module of a graph-attention-network model"""

    def __init__(self, input_dimension: int, output_dimension: int, n_heads: int = 7, alpha: float = 0.2):
        super().__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.n_heads = n_heads
        self.W = nn.Parameter(torch.zeros(size=(n_heads, input_dimension, output_dimension)))
        self.a = nn.Parameter(torch.zeros(size=(n_heads, 2 * output_dimension, 1)))
        self.softmax = nn.Softmax(dim=2)
        self.leakyrelu = nn.LeakyReLU(negative_slope=alpha)

    def _node_forward(self, h: torch.Tensor) -> torch.Tensor:
        Wh = torch.bmm(h.expand(self.n_heads, -1, -1), self.W)
        a_input = torch.cat((Wh[:, 0, :].unsqueeze(dim=1).expand(Wh.shape), Wh), 2)

        attention = torch.transpose(torch.bmm(a_input, self.a), 1, 2)
        attention = self.softmax(self.leakyrelu(attention))

        return torch.mean(torch.bmm(attention, Wh), dim=0).squeeze()


class GATDotProductModule(GATBase):
    """Base module of a dot-product-attention-network model"""

    def __init__(self, input_dimension: int, output_dimension: int, n_heads: int = 7, alpha: float = 0.2):
        super().__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.n_heads = n_heads
        self.K = nn.Parameter(torch.zeros(size=(n_heads, input_dimension, output_dimension), device=get_torch_device()))
        self.Q = nn.Parameter(torch.zeros(size=(n_heads, input_dimension, output_dimension), device=get_torch_device()))
        self.V = nn.Parameter(torch.zeros(size=(n_heads, input_dimension, output_dimension), device=get_torch_device()))
        self.leakyrelu = nn.LeakyReLU(negative_slope=alpha)

    def _node_forward(self, h: torch.Tensor) -> torch.Tensor:

        query = torch.bmm(h[0].expand(self.n_heads, 1, -1), self.Q)
        key = torch.bmm(h.expand(self.n_heads, -1, -1), self.K)
        value = torch.bmm(h.expand(self.n_heads, -1, -1), self.V)

        attention = torch.bmm(query, torch.transpose(key, 1, 2))
        attention = torch.softmax(attention, dim=1)

        out = torch.bmm(attention, value)

        return self.leakyrelu(torch.mean(out, dim=0).squeeze())
