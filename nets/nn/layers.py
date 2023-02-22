"""Layers for neural networks."""

from abc import ABC
from collections.abc import Iterable
import math
from typing import Tuple, Union
import torch

from torch import nn, Tensor
from torch.nn import functional as F


class MaskedLayer(nn.Module, ABC):
    """A layer with a pruning mask."""

    weight: Tensor
    mask: Tensor

    def __init__(self) -> None:
        """Initialize the layer."""
        super(MaskedLayer, self).__init__()
        self.mask = None

    def reset(self, weights: Tensor) -> None:
        raise NotImplementedError

    def reset_mask(self) -> None:
        self.mask.fill_(1)

    def density(self) -> float:
        return self.mask.sum().item() / self.mask.numel()

    def num_parameters(self) -> int:
        raise NotImplementedError


# TODO: Inherit from Linear as well as MaskedLayer
class MaskedLinear(MaskedLayer):
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

    def reset_parameters(self) -> None:
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

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass."""
        return F.linear(input, self.weight * self.mask, self.bias)

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


def _pair(x):
    if isinstance(x, Iterable):
        return tuple(x)
    return (x, x)


class MaskedConv2d(nn.modules.Conv2d, MaskedLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[str, int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(MaskedConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

        mask = torch.ones_like(self.weight, **factory_kwargs)
        self.mask = nn.Parameter(mask, requires_grad=False)

    def reset(self, weights: Tensor) -> None:
        self.weight = nn.Parameter(weights, requires_grad=True)

    def reset_mask(self) -> None:
        self.mask.data.fill_(1)

    def density(self) -> float:
        return self.mask.sum().item() / self.mask.numel()

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight * self.mask, self.bias)

    def num_parameters(self) -> int:
        return self.weight.numel() + (self.bias.numel() if self.bias is not None else 0)

    def extra_repr(self) -> str:
        return "in_channels={}, out_channels={}, kernel_size={}, stride={}, padding={}, dilation={}, groups={}, bias={}, density={}".format(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.bias is not None,
            f"{(self.density() * 100):.0f}%",
        )
