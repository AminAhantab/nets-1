"""Layers for neural networks."""

import math
import torch

from torch import nn, Tensor
from torch.nn import functional as F

from .prune import prune_oneshot


class MaskedLinear(nn.Module):
    """A linear layer with a pruning mask."""

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor
    mask: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        """
        Initialize the layer.

        Args:
            in_features: The number of input features.
            out_features: The number of output features.
            bias: Whether to use a bias.
            device: The device to use.
            dtype: The data type to use.
        """
        super(MaskedLinear, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.mask = nn.Parameter(
            torch.ones((out_features, in_features), device=device, dtype=torch.uint8),
            requires_grad=False,
        )

        self.reset_parameters()

    def reset(self, weights: Tensor) -> None:
        """
        Reset the parameters.

        Args:
            params: The parameters to use.
        """
        self.weight = nn.Parameter(weights, requires_grad=True)

    def reset_parameters(self, mask: bool = False) -> None:
        """
        Reset the parameters.

        Args:
            mask: Whether to reset the mask.
        """
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        if mask:
            self.reset_mask()

    def reset_mask(self) -> None:
        """Reset the mask to all ones."""
        self.mask[:, :] = 1

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass."""
        return F.linear(input, self.weight * self.mask, self.bias)

    def prune(self, p=0.2) -> int:
        """
        Prune the weights using the one-shot pruning method.

        Args:
            p: The percentile of weights to prune.

        Returns:
            The number of weights pruned.
        """
        before = self.mask.count_nonzero()
        self.mask = prune_oneshot(self.weight, self.mask, p)
        return (before - self.mask.count_nonzero()).int().item()

    def density(self) -> float:
        """Return the density of the mask."""
        return self.mask.float().mean().item()

    def num_parameters(self) -> int:
        """Return the number of parameters."""
        return self.weight.numel() + (self.bias.numel() if self.bias is not None else 0)

    def extra_repr(self) -> str:
        """Return the extra representation."""
        return "in_features={}, out_features={}, bias={}, density={}".format(
            self.in_features,
            self.out_features,
            self.bias is not None,
            f"{(self.density() * 100):.0f}%",
        )
