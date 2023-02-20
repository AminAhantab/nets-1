from typing import Union

import torch

from .base import BaseNetwork
from .layers import MaskedLinear

Batch = tuple[torch.Tensor, torch.Tensor]


class MaskedNetwork(BaseNetwork):
    """
    Base class for all networks with masked connections.

    Attributes:
        layers (list[MaskedLinear]): The layers of the network.
        mask (list[torch.Tensor]): The current mask of the network.

    Methods:
        prune(p_values): Prune the network by setting the weights of some connections to zero.
        as_ticket(): Return the winning ticket of the network.
    """

    layers: list[MaskedLinear]
    mask: list[torch.Tensor]

    def __init__(self) -> None:
        """
        Initialize the network.

        Args:
            layers (list[MaskedLinear]): The layers of the network.
        """
        super().__init__()

        self.register_full_backward_hook(self._backward_hook)

    def _backward_hook(self, _module, _grad_input, _grad_output):
        """Hook for the backward pass."""
        for layer in self.layers:
            assert isinstance(layer, MaskedLinear)
            if layer.weight.grad is not None:
                layer.weight.grad *= layer.mask

    def density(self) -> float:
        """
        Return the density of the network.

        Returns:
            float: The density of the network.
        """
        return sum([layer.density() for layer in self.layers]) / len(self.layers)
