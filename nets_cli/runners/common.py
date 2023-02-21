import logging
import os
from typing import Union

import pandas as pd

from nets_cli.args import BaseArgs
from .utils import create_path, hydrate_class_name

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
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
            return "cuda"
        else:
            logger.info("PyTorch cuda is not available")
            torch.set_default_tensor_type("torch.FloatTensor")
            return "cpu"

    except ImportError:
        raise ImportError(
            "Please install torch to use nets.\n"
            "You can install torch via `pip install torch`.\n"
            "If you have cloned the repository, you can create "
            "a conda environment with all the required dependencies "
            "by running `conda env create -f environment.yml`."
        )


def write_model(
    model,
    out_path: str,
    file_name: str = "model.pt",
    overwrite: bool = False,
):
    """
    Writes a model to disk.

    Args:
        model: The model to write.
        path: The path to write the model to.
        file_name: The name of the file to write the model to.
        overwrite: Whether to overwrite the file if it already exists.

    Raises:
        FileExistsError: If the file already exists and `overwrite` is `False`.
    """
    import torch
    from nets.nn import MaskedNetwork

    # Assert that the model is a MaskedNetwork for type checking
    assert isinstance(model, MaskedNetwork)

    # Create the directory if it does not exist, and return the full path.
    file_path = create_path(out_path, file_name)

    # Check if the file already exists.
    if os.path.exists(file_path):
        logger.debug(f"File already exists: {file_path}")
        if not overwrite:
            raise FileExistsError(f"File already exists: {file_path}")

    # Serialise the model.
    model_state = model.state_dict()
    model_class = model.__class__.__name__
    dimensions = model.dimensions
    data = {"model": model_state, "model_class": model_class, "dimensions": dimensions}

    # Write the model to disk.
    logger.info(f"Saving model to {file_path}.")
    torch.save(data, file_path)


def write_df(
    df: pd.DataFrame,
    out_path: str,
    file_name: str = "results.csv",
    overwrite: bool = False,
):
    # Create the directory if it does not exist, and return the full path.
    file_path = create_path(out_path, file_name)

    # Check if the file already exists.
    if os.path.exists(file_path):
        logger.debug(f"File already exists: {file_path}")
        if not overwrite:
            raise FileExistsError(f"File already exists: {file_path}")

    # Write the model to disk.
    logger.info(f"Saving dataframe to {file_path}.")
    df.to_csv(file_path, index=False)


def load_model(file_path: str):
    """
    Loads a model from disk.

    Args:
        dimensions: The dimensions of the model.

    Returns:
        The loaded model.
    """
    import torch
    from nets.nn import MaskedNetwork

    # Load the model from disk.
    logger.info(f"Loading model from {file_path}.")
    saved_data = torch.load(file_path)

    # Extract the model state.
    state_dict = saved_data["model"]
    model_class = saved_data["model_class"]
    dimensions = saved_data["dimensions"]

    # Instantiate the model.
    model_init = hydrate_class_name(model_class)
    model = model_init(*dimensions)
    assert isinstance(model, MaskedNetwork)

    # Load the model state and return the model.
    model.load_state_dict(state_dict, strict=False)
    logger.debug("Loaded model: %s", model)
    return model
