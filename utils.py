import logging
import json
import os

import torch

from params import HyperParameters

logger = logging.getLogger("nets")


def set_seed(seed: int = None):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): Random seed to use. If None, a random seed is generated.

    Returns:
        seed (int): Random seed used."""
    if seed is None:
        seed = torch.seed()

    torch.manual_seed(seed)
    logger.info("Random seed set: %s", seed)

    return seed


def create_dirs(*dirs):
    """
    Create directories if they don't exist.

    Args:
        dirs (str): Directories to create.
    """
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            logger.info("Created directory: %s", d)


def write_params(
    params: HyperParameters,
    dir: str = "./results",
    name: str = "params.json",
):
    """
    Write hyperparameters to file.

    Args:
        params (HyperParameters): Hyperparameters to write.
        dir (str): Directory to write to.
        name (str): Name of file to write to.
    """
    os.makedirs(dir, exist_ok=True)
    path = f"{dir}/{name}"
    with open(path, "w") as f:
        json.dump(params.__dict__, f, indent=2)
