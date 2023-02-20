import logging
import os
from typing import Union

from nets.nn.masked import MaskedNetwork
from nets_cli.args import BaseArgs

logger = logging.getLogger("nets_cli.common")


def configure_logger(args: BaseArgs):
    log_level = args.log_level
    log_format = args.log_format
    log_file = args.log_file

    handlers = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers,
    )

    if log_file is not None:
        logger.info(f"Logging to {log_file}.")


def configure_seed(seed: Union[int, None]):
    if seed is None:
        logger.info("Running in non-deterministic mode.")
        return

    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = seed
    torch.backends.cudnn.benchmark = False

    logger.info(f"Running in deterministic mode with seed {seed}.")


def configure_torch():
    try:
        import torch

        logger.info("Package torch version: %s", torch.__version__)

        if torch.cuda.is_available():
            logger.info("PyTorch cuda version: %s", torch.version.cuda)
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            logger.info("PyTorch cuda is not available")
            torch.set_default_tensor_type("torch.FloatTensor")

    except ImportError:
        raise ImportError(
            "Please install torch to use nets. "
            "You can install torch via `pip install torch`."
            "If you have cloned the repository, you can create "
            "a conda environment with all the required dependencies "
            "by running `conda env create -f environment.yml`."
        )


def write_model(
    model: MaskedNetwork,
    path: str,
    file_name: str = "model.pt",
    overwrite: bool = False,
):
    import torch

    if is_dir(path):
        os.makedirs(path, exist_ok=True)
        if not path.endswith(os.path.sep):
            path = path + os.path.sep

        path = os.path.join(path, file_name)

    if os.path.exists(path):
        logger.debug(f"File already exists: {path}")
        if not overwrite:
            raise FileExistsError(f"File already exists: {path}")

    logger.info(f"Saving model to {path}.")
    torch.save(model.state_dict(), path)


def is_dir(path: str) -> bool:
    return (
        path.endswith("/")
        or path.endswith("\\")
        or path.endswith(os.path.sep)
        or os.path.isdir(path)
    )
