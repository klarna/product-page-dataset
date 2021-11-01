"""DOM Q Net model
ref: Jia, Kiron, Ba, 2019, DOM-Q-NET: Grounded RL on Structured Language"""

from typing import Dict

import torch
from torch import Tensor, nn
from torch.nn.functional import normalize

from ..device import get_torch_device
from ..structures.trees import DataTree
from ..utilities import stream_sample
from .prototypes import TreeEmbedder


class DOMQNETModule(TreeEmbedder):
    """Performs a one layer graph convolution."""

    def __init__(self, input_dimension: int, output_dimension: int, num_layers: int = 1):
        super().__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        self.GRU = nn.GRU(input_size=input_dimension, hidden_size=input_dimension, num_layers=num_layers)

        self.layer_neighborhood = nn.Linear(input_dimension, input_dimension, bias=True)

        self.output_layer = nn.Linear(input_dimension * 2, output_dimension, bias=True)
        self.relu = nn.ReLU()

        self.cache: Dict[DataTree, Tensor] = dict()

    def _representation(self, tree: DataTree) -> Tensor:
        """Creates a vector representation based on the local node and its one hop neighbors."""

        e_l = tree.feature_vector.float().unsqueeze(-1)

        if tree.size > 1:

            e_n = torch.stack([node.feature_vector.float() for node in tree.neighbors])

            m_n = torch.sum(self.layer_neighborhood(e_n), dim=0)

            e_n, _ = self.GRU(m_n.unsqueeze(0).unsqueeze(0), torch.transpose(e_l, 0, 1).unsqueeze(0))

            e_n = torch.transpose(e_n.squeeze(0), 0, 1)

        else:
            e_n = e_l

        return torch.cat((e_l, e_n), dim=0)

    def forward(self, node: DataTree) -> Tensor:
        if node in self.cache:
            return self.cache[node]

        e_ln = self._representation(node)
        e_ln = self.relu(self.output_layer(torch.transpose(e_ln, 0, 1))).squeeze(0)

        e_ln = normalize(e_ln, dim=0)

        self.cache[node] = e_ln

        return e_ln

    def reset(self) -> None:
        self.cache = dict()


class DOMQNETWithGlobalEmbeddingModule(DOMQNETModule):
    """DOMQNET module with global embeddings."""

    def __init__(self, input_dimension: int, output_dimension: int, num_layers: int = 1):
        super().__init__(input_dimension=input_dimension, output_dimension=output_dimension, num_layers=num_layers)
        self.output_layer = nn.Linear(input_dimension * 4, output_dimension, bias=True)
        self.cache_global: Dict[DataTree, Tensor] = dict()

    def _global_representation(self, tree: DataTree) -> Tensor:
        """Creates a single representation based on the entire DOM-tree."""

        e_g = torch.zeros(self.input_dimension * 2, 1, device=get_torch_device())

        sample_size = int(tree.root.size * 0.1)

        for node in stream_sample(tree.root, sample_size):

            e_ln = super()._representation(node)

            e_g = torch.maximum(e_g, e_ln)

        return e_g

    def _representation(self, tree: DataTree) -> Tensor:
        e_ln = super()._representation(tree)
        e_g = self._global_representation(tree)

        return torch.cat((e_ln, e_g), dim=0)

    def reset(self) -> None:
        super().reset()
        self.cache = dict()
