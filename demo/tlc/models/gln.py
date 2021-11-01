"""Gated Linear Network model
ref: Veness et al. 2020, Gated Linear Networks"""

from typing import List

import torch
from torch import nn

from tlc.device import get_torch_device


class GLNModule(nn.Module):
    """Gated linear network (initial implementation with backprop ON)."""

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        layer_dimensions: List[int],
        context_dimension: int,
        epsilon: float = 1e-3,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for prev_dim, curr_dim in zip([input_dimension] + layer_dimensions, layer_dimensions + [output_dimension]):
            self.layers.append(_GatedLinearLayer(prev_dim, curr_dim, context_dimension, input_dimension, epsilon))

    def forward(self, side_info: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model"""
        output_data = None
        for layer in self.layers:
            output_data = layer(side_info, output_data)
        return output_data


class _GatedLinearLayer(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        context_dimension: int,
        side_dimension: int,
        epsilon: float = 0.01,
    ):
        super().__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.side_dimension = side_dimension
        self.context_dimension = context_dimension
        self.offset = torch.randn(self.context_dimension, requires_grad=False, device=get_torch_device())
        self.normal = nn.functional.normalize(
            torch.randn(self.side_dimension, self.context_dimension, requires_grad=False, device=get_torch_device())
        )
        self.weights = nn.Parameter(
            1.0
            / (self.input_dimension + 1)
            * torch.ones(2 ** self.context_dimension, self.input_dimension + 1, self.output_dimension)
        )
        self.epsilon = epsilon
        bias = epsilon + torch.rand(1, requires_grad=False, device=get_torch_device()) * (1 - 2 * epsilon)
        while bias == 0.5:
            bias = epsilon + torch.rand(1, requires_grad=False, device=get_torch_device()) * (1 - 2 * epsilon)
        self.bias = bias

    def forward(self, side_info: torch.Tensor, input_data: torch.Tensor = None) -> torch.Tensor:
        """Forward pass of the model"""
        side_info = side_info.float()
        if input_data is None:
            input_data = torch.cat(
                (self.bias, torch.sigmoid(torch.clamp(side_info, self.epsilon, 1 - self.epsilon))), 0
            )
        else:
            input_data = torch.cat((self.bias, input_data), 0)
        context_func = torch.gt(torch.matmul(side_info, self.normal), self.offset).float()
        gating = self._bin2dec(context_func, self.context_dimension)
        output_data = torch.sigmoid(torch.matmul(torch.logit(input_data), self.weights[gating, :, :]))
        return torch.clamp(output_data, self.epsilon, 1 - self.epsilon)

    @staticmethod
    def _bin2dec(bit_vec: torch.Tensor, bit_num: int) -> torch.Tensor:
        powers2 = 2.0 ** torch.linspace(bit_num - 1, 0, bit_num)
        return torch.matmul(bit_vec, powers2).long()
