"""Tree Recurrent neural network model"""

from typing import Callable, Dict, Optional

import torch
from torch import Tensor, nn

from ..device import get_torch_device
from ..structures.trees import DataTree
from .prototypes import TreeEmbedder
from .utilities import SaveMixin


class BottomUpTreeRNN(TreeEmbedder, SaveMixin):
    """Base module of a bottom-up recurrent neural network model"""

    def __init__(self, input_dimension: int, output_dimension: int):
        super().__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        self.linear = nn.Linear(input_dimension + output_dimension, output_dimension)
        self.sigmoid = nn.LeakyReLU(negative_slope=0.1)

        self.cache: Dict[DataTree, Tensor] = dict()

    def _node_forward(self, x: Tensor, h: Tensor) -> Tensor:
        return self.linear(torch.cat((x, h)))

    def forward(self, node: DataTree) -> Tensor:
        if node in self.cache:
            return self.cache[node]

        if not node.children:
            h_child = torch.zeros(self.output_dimension, device=get_torch_device())
        else:
            h_child = torch.mean(torch.stack([self(child) for child in node.children]), dim=0)

        x = node.feature_vector.float()

        self.cache[node] = self._node_forward(x, h_child)

        return self.cache[node]

    def reset(self) -> None:
        self.cache = dict()


class TopDownTreeRNN(BottomUpTreeRNN):
    """Base module of a top-down recurrent neural network model"""

    def __init__(self, input_dimension: int, output_dimension: int):
        super().__init__(input_dimension, output_dimension)
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

    def forward(
        self,
        node: DataTree,
        tree_embedding: Optional[Callable[[DataTree], torch.Tensor]] = None,
    ) -> Tensor:
        # pylint: disable=arguments-differ
        if node in self.cache:
            return self.cache[node]

        if node.parent is None:
            h_parent = torch.zeros(self.output_dimension, device=get_torch_device())
        else:
            h_parent = self(node.parent, tree_embedding)

        x = tree_embedding(node) if tree_embedding else node.feature_vector.float()

        self.cache[node] = self._node_forward(x, h_parent)
        return self.cache[node]
