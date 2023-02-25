import torch
from torch.nn import Parameter

from nets.nn import MaskedNetwork, MaskedLayer


def uniform_mask(shape: tuple[int, ...], p: float) -> torch.Tensor:
    """
    Return a mask with a given density.

    Args:
        shape (tuple[int, ...]): The shape of the mask.
        p (float): The density of the mask.

    Returns:
        torch.Tensor: The mask.
    """
    return torch.bernoulli(torch.full(shape, p)).float()


def set_uniform_masks(model: MaskedNetwork, density: float):
    import torch

    if density == 0.0:
        for layer in model.layers:
            assert isinstance(layer, MaskedLayer)
            mask = torch.zeros(layer.mask.shape)
            layer.mask = Parameter(mask, requires_grad=False)

        return model

    if density == 1.0:
        for layer in model.layers:
            assert isinstance(layer, MaskedLayer)
            mask = torch.ones(layer.mask.shape)
            layer.mask = Parameter(mask, requires_grad=False)

        return model

    for layer in model.layers:
        assert isinstance(layer, MaskedLayer)
        mask = uniform_mask(layer.mask.shape, density)
        layer.mask = Parameter(mask, requires_grad=False)
