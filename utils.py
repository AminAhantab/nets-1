import logging
import json
import os

import torch

from hyperparameters import HyperParameters

logger = logging.getLogger("nets")


def set_seed(seed: int = None):
    if seed is None:
        seed = torch.seed()

    torch.manual_seed(seed)
    logger.info("Random seed set: %s", seed)

    return seed


def create_dirs(*dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            logger.info("Created directory: %s", d)


def write_params(
    params: HyperParameters,
    dir: str = "./results",
    name: str = "params.json",
):
    os.makedirs(dir, exist_ok=True)
    path = f"{dir}/{name}"
    with open(path, "w") as f:
        json.dump(params.__dict__, f, indent=2)
