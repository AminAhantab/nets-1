import logging
import os
from typing import Union

from nets_cli.args import BaseArgs

logger = logging.getLogger("nets_cli.common")


def configure_logger(args: BaseArgs):
    """
    Configures the `logging` module.

    Args:
        args: The parsed command-line arguments.
    """
    # Extract arguments
    log_level = args.log_level
    log_format = args.log_format
    log_file = args.log_file

    # Always log to console, optionally log to file.
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))

    # Configure logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers,
    )

    # Log whether we are logging to file.
    if log_file is not None:
        logger.info(f"Logging to {log_file}.")


def configure_seed(seed: Union[int, None]):
    """
    Configures the random seed.

    Args:
        seed: The random seed.

    Notes:
        If `seed` is `None`, the random seed is not set.
    """
    # If the seed is None, we are running in non-deterministic mode.
    if seed is None:
        logger.info("Running in non-deterministic mode.")
        return

    import random
    import numpy as np
    import torch

    # Make environment as reproducible as possible.
    random.seed(seed)  # python RNG
    np.random.seed(seed)  # numpy RNG
    torch.manual_seed(seed)  # pytorch RNG
    torch.cuda.manual_seed(seed)  # cuda RNG
    torch.backends.cudnn.deterministic = seed is not None  # use deterministic cuDNN
    torch.backends.cudnn.benchmark = False  # enable cuDNN benchmark

    logger.info(f"Running in deterministic mode with seed {seed}.")


def configure_torch(no_cuda: bool = False) -> str:
    """
    Configures PyTorch.

    Returns:
        The device type.
    """
    try:
        import torch

        import torch.multiprocessing as mp

        # PyTorch handles multiprocessing in a specific way.
        # NOTE: This may cause difficulties on Windows.
        mp.set_start_method("spawn", force=True)

        if no_cuda:
            logger.info("PyTorch cuda is disabled")
            torch.set_default_tensor_type("torch.FloatTensor")
            return "cpu"

        logger.info("Package torch version: %s", torch.__version__)
        if torch.cuda.is_available():
            logger.info("PyTorch cuda version: %s", torch.version.cuda)
            # torch.set_default_tensor_type("torch.cuda.FloatTensor")
            return "cuda"
        else:
            logger.info("PyTorch cuda is not available")
            # torch.set_default_tensor_type("torch.FloatTensor")
            return "cpu"

    except ImportError:
        raise ImportError(
            "Please install torch to use nets.\n"
            "You can install torch via `pip install torch`.\n"
            "If you have cloned the repository, you can create "
            "a conda environment with all the required dependencies "
            "by running `conda env create -f environment.yml`."
        )
