import argparse
import json
import logging
import os

import pandas as pd

from .hydrator import hydrate_class_name

logger = logging.getLogger("nets_cli.io")


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


def write_df(
    df: pd.DataFrame,
    out_path: str,
    file_name: str = "results.csv",
    overwrite: bool = False,
):
    """
    Writes a dataframe to disk.

    Args:
        df: The dataframe to write.
        path: The path to write the dataframe to.
        file_name: The name of the file to write the dataframe to.
        overwrite: Whether to overwrite the file if it already exists.

    Raises:
        FileExistsError: If the file already exists and `overwrite` is `False`.
    """
    # Create the directory if it does not exist, and return the full path.
    file_path = create_path(out_path, file_name)

    # Check if the file already exists.
    if os.path.exists(file_path):
        logger.debug(f"File already exists: {file_path}")
        if not overwrite:
            raise FileExistsError(f"File already exists: {file_path}")

    # Write the model to disk.
    logger.info(f"Saving dataframe to {file_path}.")
    df.to_csv(file_path, index=True)


def is_dir(path: str) -> bool:
    """
    Checks whether a path is a directory.

    Args:
        path: The path to check.

    Returns:
        Whether the path is a directory.
    """
    if os.path.exists(path):
        return os.path.isdir(path)
    else:
        if path.endswith(os.path.sep):
            return True
        else:
            return False


def create_path(path: str, file_name: str = "model.pt") -> str:
    """
    Creates a path to a file iff the parent directory exists (or can be
    created).

    Args:
        path: The path to the file.
        file_name: The name of the file.

    Returns:
        The full path to the file.
    """
    if is_dir(path):
        os.makedirs(path, exist_ok=True)
        if not path.endswith(os.path.sep):
            path = path + os.path.sep

        path = os.path.join(path, file_name)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    return path


def write_params(
    args: argparse.Namespace,
    dir: str = "./results",
    name: str = "params.json",
):
    """
    Write hyperparameters to file.

    Args:
        args (argparse.Namespace): Namespace of hyperparameters.
        dir (str): Directory to write to.
        name (str): Name of file to write to.
    """
    os.makedirs(dir, exist_ok=True)
    path = f"{dir}/{name}"
    with open(path, "w") as f:
        json.dump(args.__dict__, f, indent=2)
