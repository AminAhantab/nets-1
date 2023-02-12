from typing import Tuple, Union

import torch
from torch.utils.data import TensorDataset

XORData = Union[TensorDataset, Tuple[TensorDataset, TensorDataset]]


def xor_data() -> TensorDataset:
    """
    Generate the XOR dataset.

    Returns:
        A tuple of the train and validation datasets.
    """
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y = torch.tensor([0, 1, 1, 0], dtype=torch.float32)
    return TensorDataset(X, y)


def generate_noisy_data(n: int = 100) -> TensorDataset:
    """
    Generate a noisy XOR dataset.

    Args:
        train_size: The size of the training set.
        val_size: The size of the validation set.

    Returns:
        A tuple of the train and validation datasets.
    """
    X = torch.rand(n, 2)
    y = torch.bitwise_xor(X[:, 0].round().int(), X[:, 1].round().int())
    return TensorDataset(X, y)


def load(noisy: bool = False, train_size: int = None, val_size: int = None) -> XORData:
    """
    Load the XOR dataset.

    Args:
        noisy: Whether to generate noisy data.
        train_size: The size of the training set.
        val_size: The size of the validation set.

    Returns:
        A tuple of the train and validation datasets if `val_size` is not None,
        otherwise just the training data.
    """
    # Generate either pure or noisy XOR data
    assert not noisy or train_size is None
    if noisy:
        train_data = generate_noisy_data(train_size)
    else:
        train_data = xor_data()

    # Decide whether to generate validation set
    if val_size is not None:
        return train_data, generate_noisy_data(val_size)

    return train_data
