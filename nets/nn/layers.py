"""Layers for neural networks."""

from abc import ABC
from collections.abc import Iterable
import math
from typing import List, Optional, Tuple, Union
import torch

from torch import nn, Tensor
from torch.nn import functional as F

PADDING_STRS = {"same", "valid"}
PADDING_MODES = {"zeros", "reflect", "replicate", "circular"}


class MaskedLayer(nn.Module, ABC):
    """A layer with a pruning mask."""

    mask: Tensor

    def __init__(self) -> None:
        """Initialize the layer."""
        super(MaskedLayer, self).__init__()
        self.mask = None

    def reset(self, weights: Tensor) -> None:
        raise NotImplementedError

    def reset_parameters(self, mask: bool = False) -> None:
        raise NotImplementedError

    def reset_mask(self) -> None:
        self.mask.fill_(1)

    def density(self) -> float:
        return self.mask.sum().item() / self.mask.numel()

    def num_parameters(self) -> int:
        raise NotImplementedError


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


class MaskedConvNd(MaskedLayer):
    __constants__ = [
        "stride",
        "padding",
        "dilation",
        "groups",
        "padding_mode",
        "output_padding",
        "in_channels",
        "out_channels",
        "kernel_size",
    ]
    __annotations__ = {"bias": Optional[torch.Tensor]}

    def _conv_forward(
        self,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        mask: Tensor,
    ) -> Tensor:
        ...

    in_channels: int
    _reversed_padding_repeated_twice: List[int]
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Union[str, Tuple[int, ...]]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: str
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...],
        padding: Tuple[int, ...],
        dilation: Tuple[int, ...],
        transposed: bool,
        output_padding: Tuple[int, ...],
        groups: int,
        bias: bool,
        padding_mode: str,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(MaskedConvNd, self).__init__()
        if groups <= 0:
            raise ValueError("groups must be a positive integer")

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")

        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        if isinstance(padding, str):
            if padding not in PADDING_STRS:
                msg = (
                    f"Invalid padding string {padding}, should be one of {PADDING_STRS}"
                )
                raise ValueError(msg)

            if padding == "same" and any(s != 1 for s in stride):
                msg = "padding='same' is not supported for strided convolutions"
                raise ValueError(msg)

        if padding_mode not in PADDING_MODES:
            msg = f"padding_mode must be one of {PADDING_MODES}, but got padding_mode='{padding_mode}'"
            raise ValueError(msg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode

        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == "same":
                for d, k, i in zip(
                    dilation, kernel_size, range(len(kernel_size) - 1, -1, -1)
                ):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad
                    )
        else:
            self._reversed_padding_repeated_twice = (
                nn.modules.utils._reverse_repeat_tuple(self.padding, 2)
            )

        if transposed:
            weight_dims = (in_channels, out_channels // groups, *kernel_size)
        else:
            weight_dims = (out_channels, in_channels // groups, *kernel_size)
        self.weight = nn.Parameter(torch.empty(weight_dims, **factory_kwargs))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        mask = torch.ones_like(self.weight, **factory_kwargs)
        self.mask = nn.Parameter(mask, requires_grad=False)

        self.reset_parameters()

    def reset_parameters(self, mask: bool = False) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

        if mask:
            self.reset_mask()

    def num_parameters(self) -> int:
        return self.weight.numel() + (self.bias.numel() if self.bias is not None else 0)

    def extra_repr(self):
        density = self.density() * 100
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}, "
        )
        s += f"density={density:.0f}%, "
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"

        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(MaskedConvNd, self).__setstate__(state)
        if not hasattr(self, "padding_mode"):
            self.padding_mode = "zeros"


def _pair(x):
    if isinstance(x, Iterable):
        return tuple(x)
    return (x, x)


class MaskedConv2d(MaskedConvNd):
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
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _conv_forward(
        self,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        mask: Tensor,
    ) -> Tensor:
        if self.padding_mode != "zeros":
            conv_input = F.pad(
                input,
                self._reversed_padding_repeated_twice,
                mode=self.padding_mode,
            )
        else:
            conv_input = input

        return F.conv2d(
            conv_input,
            weight * mask,
            bias,
            self.stride,
            _pair(0),
            self.dilation,
            self.groups,
        )

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias, self.mask)
