"""Graph Convolutional Neural Networks"""

from random import sample, shuffle
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn.functional import normalize

from ..structures.trees import DataTree
from .prototypes import TreeEmbedder


class GCNModule(TreeEmbedder):
    """Performs a one layer graph convolution."""

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        neighborhood_sample_percentage: float = 1.0,
        neighbor_dimension: Optional[int] = None,
    ):
        super().__init__()

        if neighbor_dimension is None:
            neighbor_dimension = input_dimension
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        # The weight matrix applied to the concatenation of the node and its neighbors at layer k = 1.
        self.layer_node = nn.Linear(input_dimension + neighbor_dimension, output_dimension, bias=True)

        # The layer applied to the node's neighbors at depth k=1
        self.layer_neighbors = nn.Linear(neighbor_dimension, neighbor_dimension, bias=True)

        self.cache: Dict[DataTree, Tensor] = dict()

        # This variable specifies the percentage of a node's neighbors that will be sampled in the random walk.
        self.neighborhood_sample_percentage = neighborhood_sample_percentage

    def layer_forward(self, x_l: Tensor, x_n: Tensor) -> Tensor:
        """Tensor forward pass"""

        # The weight matrix Q is applied to every neighbor's vector representation.
        h_n = torch.relu(self.layer_neighbors(x_n))

        # Apply a aggregation function to the neighbors.
        h_n = torch.mean(h_n, dim=0)

        # Concatenate the local nodes representation with its neighbors aggregated representation.
        h_ln = torch.cat((x_l, h_n)).unsqueeze(dim=0)

        h_lnew = torch.relu(self.layer_node(h_ln)).squeeze()

        return normalize(h_lnew, dim=0)

    def forward(self, node: DataTree) -> Tensor:
        if node in self.cache:
            return self.cache[node]

        # Randomly sample a neighborhood around the current node.
        neighbors = [n.feature_vector.float() for n in _random_walk(node, self.neighborhood_sample_percentage)]
        x_n = torch.stack(neighbors)

        # Retrieve feature vector of local node.
        x_l = node.feature_vector.float()

        h_lnew = self.layer_forward(x_l, x_n)

        self.cache[node] = h_lnew

        return h_lnew

    def reset(self) -> None:
        self.cache = dict()


class MultiLayerGCNModule(TreeEmbedder):
    """Multi layer graph convolution model"""

    def __init__(
        self,
        input_dimension: int,
        latent_dimension: List[int],
        neighborhood_sample_percentage: float = 1.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.cache: Dict[Tuple[DataTree, int], Tensor] = dict()
        self.neighborhood_sample_percentage = neighborhood_sample_percentage

        for i, o in zip([input_dimension] + latent_dimension[:-1], latent_dimension):
            self.layers.append(GCNModule(i, o, neighborhood_sample_percentage=neighborhood_sample_percentage))

    def _recurse(self, node: DataTree, i: int) -> Tensor:
        """Apply forward layer recursively backward through the graph"""
        if (node, i) in self.cache:
            return self.cache[(node, i)]

        if i == 0:  # Base case, apply the first layer to the raw nodes
            return self.layers[0](node)

        # Find embeddings of nodes from previous layer
        x_l = self._recurse(node, i - 1)
        x_n = torch.stack([self._recurse(n, i - 1) for n in _random_walk(node, self.neighborhood_sample_percentage)])

        # Compute output of current layer based on output of previous layers
        h = self.layers[i].layer_forward(x_l, x_n)

        self.cache[(node, i)] = h
        return h

    def forward(self, node: DataTree) -> Tensor:
        return self._recurse(node, len(self.layers) - 1)

    def reset(self) -> None:
        for mdl in self.layers:
            mdl.reset()

        self.cache = dict()


class FeedforwardMultiLayerGCNModule(TreeEmbedder):
    """Multi layer graph convolution model with local feature feedforward"""

    def __init__(
        self,
        input_dimension: int,
        latent_dimension: List[int],
        neighborhood_sample_percentage: float = 1.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.cache: Dict[Tuple[DataTree, int], Tensor] = dict()
        self.neighborhood_sample_percentage = neighborhood_sample_percentage

        for i, o in zip([input_dimension] + latent_dimension[:-1], latent_dimension):
            self.layers.append(
                GCNModule(
                    input_dimension,
                    o,
                    neighbor_dimension=i,
                    neighborhood_sample_percentage=neighborhood_sample_percentage,
                )
            )

    def _recurse(self, node: DataTree, i: int) -> Tensor:
        """Apply forward layer recursively backward through the graph"""
        if (node, i) in self.cache:
            return self.cache[(node, i)]

        if i == 0:  # Base case, apply the first layer to the raw nodes
            return self.layers[0](node)

        # Find embeddings of nodes from previous layer
        x_l = node.feature_vector.float()
        x_n = torch.stack([self._recurse(n, i - 1) for n in _random_walk(node, self.neighborhood_sample_percentage)])

        # Compute output of current layer based on output of previous layers
        h = self.layers[i].layer_forward(x_l, x_n)

        self.cache[(node, i)] = h
        return h

    def forward(self, node: DataTree) -> Tensor:
        return self._recurse(node, len(self.layers) - 1)

    def reset(self) -> None:
        for mdl in self.layers:
            mdl.reset()

        self.cache = dict()


def _random_walk(node: DataTree, neighborhood_sample_percentage: float = 1.0) -> List[DataTree]:
    """Performs a random walk of depth one around a node.
    The number of neighbors being considered in the random walk
    is specified by the neighborhood_sample_percentage variable.
    """
    # Extract the node's neighborhood exluding the node itself.
    neighbors = list(node.neighbors)
    shuffle(neighbors)

    # Calculate how many nodes the sampling percentage is equivalent to for the current node.
    no_neighbors = len(neighbors)
    sample_size = min(max(round(neighborhood_sample_percentage * no_neighbors), 1), no_neighbors)

    return sample(neighbors, sample_size)
