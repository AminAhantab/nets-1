"""
Pruning methods.
"""

from torch import Tensor


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


def prune_oneshot(weights: Tensor, mask: Tensor, p=0.2) -> Tensor:
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
