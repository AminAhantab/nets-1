"""
Pruning methods.
"""

from torch import Tensor
import torch
from nets.nn.layers import MaskedLayer

from nets.nn.masked import MaskedNetwork


def prune_magnitude(model: MaskedNetwork, fraction: float) -> int:
    """
    Prune the weights using the magnitude pruning method.

    Parameters
    ----------
    model : MaskedNetwork
        The model to prune.
    threshold : float
        The weight threshold τ at which to prune.

    Returns
    -------
    MaskedNetwork
        The pruned model.
    """
    removed_weights = 0
    for layer in model.layers:
        assert isinstance(layer, MaskedLayer)
        initial_weights = layer.mask.sum()
        new_mask = prune_magnitude_rate(layer.weight, layer.mask, fraction)
        removed_weights += initial_weights - new_mask.sum()

    return removed_weights


def prune_random(
    model: MaskedNetwork,
    count: int = None,
    fraction: float = None,
) -> MaskedNetwork:
    """
    Prune the weights using the random pruning method.

    Parameters
    ----------
    model : MaskedNetwork
        The model to prune.
    count : int, optional
        The number of weights to prune, by default None
    fraction : float, optional
        The fraction of weights to prune, by default None

    Returns
    -------
    MaskedNetwork
        The pruned model.
    """
    assert count is not None or fraction is not None

    removed_weights = 0
    for layer in model.layers:
        assert isinstance(layer, MaskedLayer)
        initial_weights = layer.mask.sum()
        new_mask = prune_random_rate(layer.weight, layer.mask, fraction)
        removed_weights += initial_weights - new_mask.sum()

    return removed_weights


def prune_threshold(weights: Tensor, p=0.2) -> float:
    """
    Determine the weight threshold τ at which to prune.

    Parameters
    ----------
    weights : Tensor
        The weights to prune.
    p : float, optional
        The percentile of weights to prune, by default 0.2

    Returns
    -------
    Tensor
        The weight threshold τ at which to prune.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError("p must be in the range [0, 1]")

    if weights is None or p == 0.0:
        return 0.0

    if p == 1.0:
        return weights.flatten().abs().max().item() + 1

    # Sort all the weights by absolute value into a one-dimensional vector
    s = weights[weights != 0].flatten().abs().sort()[0]

    # Determine the index of the bottom p-percentile value
    x = round(len(s) * p)

    # Return the threshold as the value at that index
    return s[x].item()


def prune_magnitude_rate(weights: Tensor, mask: Tensor, p=0.2) -> Tensor:
    """
    Prune the weights using the one-shot pruning method.

    Parameters
    ----------
    weights : Tensor
        The weights to prune.
    mask : Tensor
        The pruning mask.
    p : float, optional
        The percentile of weights to prune, by default 0.2

    Returns
    -------
    Tensor
        The updated pruning mask.
    """
    if weights is None or mask is None:
        return None

    if weights.shape != mask.shape:
        raise ValueError("weights and mask must have the same shape")

    if weights.device != mask.device:
        raise ValueError("weights and mask must be on the same device")

    # Apply the pruning mask (Hadamard product)
    hadamard_product = weights * mask

    # Determine the weight threshold τ at which to prune
    threshold = prune_threshold(hadamard_product, p)

    # Set the mask values to 0 for those below τ
    mask[weights.abs() < threshold] = 0

    # Return the pruning mask
    return mask

def prune_random_rate(weights: Tensor, mask: Tensor, p=0.2) -> Tensor:
    """
    Prune the weights using the random pruning method.

    Parameters
    ----------
    weights : Tensor
        The weights to prune.
    mask : Tensor
        The pruning mask.
    p : float, optional
        The percentile of weights to prune, by default 0.2

    Returns
    -------
    Tensor
        The updated pruning mask.
    """
    if weights is None or mask is None:
        return None

    if weights.shape != mask.shape:
        raise ValueError("weights and mask must have the same shape")

    if weights.device != mask.device:
        raise ValueError("weights and mask must be on the same device")

    # Number of non-zero weights
    count = int(p * mask.sum())

    # Zero out `count` number of non-zero mask values
    mask = mask.flatten()
    indices = torch.randperm(mask.sum())[:count]
    mask[indices] = 0
    mask = mask.reshape(weights.shape)


    # Return the pruning mask
    return mask
    