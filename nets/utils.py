import torch


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
