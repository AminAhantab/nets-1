import logging
import os
from typing import Tuple

import torch

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


def configure_seed(seed: int):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_model(
    model: MaskedNetwork,
    path: str,
    file_name: str = "model.pt",
    overwrite: bool = False,
):
    if is_dir(path):
        os.makedirs(path, exist_ok=True)
        if not path.endswith(os.path.sep):
            path = path + os.path.sep

        path = os.path.join(path, file_name)

    if os.path.exists(path):
        logger.warning(f"File already exists: {path}")
        if not overwrite:
            raise FileExistsError(f"File already exists: {path}")

    torch.save(model.state_dict(), path)


def is_dir(path: str) -> Tuple[bool, bool]:
    return (
        path.endswith("/")
        or path.endswith("\\")
        or path.endswith(os.path.sep)
        or os.path.isdir(path)
    )
