from typing import Union

import torch

from .base import BaseNetwork
from .layers import MaskedLayer

Batch = tuple[torch.Tensor, torch.Tensor]


class MaskedNetwork(BaseNetwork):
    """
    Base class for all networks with masked connections.

    Attributes:
        layers (list[MaskedLayer]): The layers of the network.
        mask (list[torch.Tensor]): The current mask of the network.

    Methods:
        prune(p_values): Prune the network by setting the weights of some connections to
            zero.
        as_ticket(): Return the winning ticket of the network.
    """

    layers: list[MaskedLayer]
    mask: list[torch.Tensor]

    def __init__(self, in_channels: int, in_features: int, out_features: int) -> None:
        """
        Initialize the network.

        Args:
            layers (list[MaskedLayer]): The layers of the network.
        """
        super().__init__()

        self.dimensions = (in_channels, in_features, out_features)

        self.register_full_backward_hook(self._backward_hook)

    def _backward_hook(self, _module, _grad_input, _grad_output):
        """Hook for the backward pass."""
        for layer in self.layers:
            assert isinstance(layer, MaskedLayer)
            if layer.weight.grad is not None:
                layer.weight.grad *= layer.mask

    def density(self) -> float:
        """
        Return the density of the network.

        Returns:
            float: The density of the network.
        """
        return sum([layer.density() for layer in self.layers]) / len(self.layers)
