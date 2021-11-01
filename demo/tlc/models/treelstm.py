"""Tree LSTM model"""

from functools import partial
from typing import Callable, Dict, Tuple

import torch
from torch import Tensor, nn

from ..device import get_torch_device
from ..structures.trees import DataTree
from .prototypes import TreeEmbedder

EmbeddingType = Tuple[Tensor, Tensor]  # (cell state, hidden state)


class BottomUpTreeLSTM(TreeEmbedder):
    """Basic bottom-up LSTM model for tree-structured data"""

    def __init__(self, input_dimension: int, output_dimension: int):
        super().__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        self.ioux = nn.Linear(input_dimension, 3 * output_dimension)
        self.iouh = nn.Linear(output_dimension, 3 * output_dimension)

        self.fx = nn.Linear(input_dimension, output_dimension)
        self.fh = nn.Linear(output_dimension, output_dimension)
        self.cache: Dict[DataTree, EmbeddingType] = dict()

    def _node_forward(self, x: Tensor, previous_embedding: EmbeddingType) -> EmbeddingType:
        previous_c, previous_h = previous_embedding
        sum_h = torch.sum(previous_h, dim=0)
        iou = self.ioux(x.float()) + self.iouh(sum_h)
        i, o, u = torch.split(iou, self.output_dimension)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        fs = self.fx(x.float()).expand(len(previous_h), -1) + self.fh(previous_h)
        fs = torch.sigmoid(fs)
        c_out = torch.sum(torch.mul(fs, previous_c), dim=0) + torch.mul(i, u)

        h_out = torch.mul(o, torch.tanh(c_out))

        return (c_out, h_out)

    def reset(self) -> None:
        self.cache = dict()

    def _recurse(self, node: DataTree) -> EmbeddingType:
        if node in self.cache:
            return self.cache[node]

        if not node.children:
            child_h = torch.zeros(self.output_dimension, device=get_torch_device()).unsqueeze(0)
            child_c = torch.zeros(self.output_dimension, device=get_torch_device()).unsqueeze(0)
        else:
            seq_c, seq_h = zip(*map(partial(self._recurse), node.children))
            child_c, child_h = torch.stack(seq_c), torch.stack(seq_h)

        self.cache[node] = self._node_forward(node.feature_vector, (child_c, child_h))

        return self.cache[node]

    def forward(self, node: DataTree) -> Tensor:
        assert node is not None, "Tree is None!"
        return self._recurse(node)[1]


class TopDownTreeLSTM(BottomUpTreeLSTM):
    """Basic top-down LSTM model for tree-structured data"""

    # pylint: disable=arguments-differ
    def _recurse(
        self,
        node: DataTree,
        tree_embedding: Callable[[DataTree], EmbeddingType] = None,
    ) -> EmbeddingType:
        if node in self.cache:
            return self.cache[node]

        if node.parent is None:
            parent_h = torch.zeros(self.output_dimension, device=get_torch_device())
            parent_c = torch.zeros(self.output_dimension, device=get_torch_device())
        else:
            parent_c, parent_h = self._recurse(node.parent, tree_embedding=tree_embedding)  # type:ignore

        x = node.feature_vector if tree_embedding is None else tree_embedding(node)

        self.cache[node] = self._node_forward(x, (parent_c.unsqueeze(0), parent_h.unsqueeze(0)))

        return self.cache[node]

    def forward(
        self,
        node: DataTree,
        tree_embedding: Callable[[DataTree], EmbeddingType] = None,
    ) -> Tensor:
        assert node is not None, "Tree is None!"
        return self._recurse(node, tree_embedding)[1]
