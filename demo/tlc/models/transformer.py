"""Local transformer model"""

import torch
from torch import Tensor, nn
from torch.nn.functional import pad

from ..structures.trees import DataTree
from .prototypes import TreeEmbedder


class TransformerEncoderModule(TreeEmbedder):
    """Module for simple transformer encoder model"""

    def __init__(self, input_dimension: int, latent_dimension: int = 150, n_head: int = 5):
        super().__init__()
        self.input_dimension = input_dimension
        self.padded_dimension = n_head * (input_dimension // n_head + 1)
        self.pad_size = self.padded_dimension - input_dimension
        self.transformer = nn.TransformerEncoderLayer(
            self.padded_dimension, dim_feedforward=latent_dimension, nhead=n_head
        )

    def forward(self, node: DataTree) -> Tensor:

        # Compute neighbor features
        neighbors = [node.feature_vector.float()]
        neighbors.extend(n.feature_vector.float() for n in node.neighbors)
        h = pad(
            torch.stack(neighbors), (0, self.pad_size), value=0
        )  # Pad by one to account for feedforward discrepancy

        # Compute embedding as average pooled transformer decoding of neighborhood
        return self.transformer(h.unsqueeze(1))[0].squeeze()
