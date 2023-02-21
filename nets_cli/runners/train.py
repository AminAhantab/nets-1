import logging

import pandas as pd

from ..args import TrainArgs
from .common import (
    configure_logger,
    configure_seed,
    configure_torch,
    load_model,
    write_df,
    write_model,
)

logger = logging.getLogger("nets_cli.train")


def run_train(args: TrainArgs) -> None:
    # Configure environment
    configure_logger(args)
    device = configure_torch(args.no_cuda)
    configure_seed(args.seed)

    from nets.nn.train import train_model

    # Get relevant arguments
    model_path = args.model_path
    dataset = args.dataset
    optimiser = args.optimiser
    learning_rate = args.learning_rate
    max_epochs = args.max_epochs
    batch_size = args.batch_size

    # Get dataset
    train_data, val_data, test_data = get_datasets(
        dataset,
        val_size=5_000,
        device=device,
        seed=args.seed,
    )

    # Load model
    model = load_model(model_path)
    logger.info("Moving model to device %s.", device)
    model.to(device)

    # Initialise optimiser
    opt = get_optimiser(model, optimiser, learning_rate)

    # Initialise the data loaders
    train_loader = make_data_loader(
        train_data,
        batch_size=batch_size,
        device=device,
        seed=args.seed,
        shuffle=True,
        # num_workers=4,
    )

    val_loader = make_data_loader(
        val_data,
        batch_size=99_999,
        device=device,
        seed=args.seed,
        shuffle=False,
    )

    test_loader = make_data_loader(
        test_data,
        batch_size=99_999,
        device=device,
        seed=args.seed,
        shuffle=False,
    )

    # Initialise the callbacks
    df = pd.DataFrame()
    callbacks = {
        "iteration": [
            train_loss_cb(df, every=100),
            val_loss_cb(df, val_loader, every=100),
            test_loss_cb(df, test_loader, every=100),
        ],
        "epoch": [val_loss_cb(df, val_loader), test_loss_cb(df, test_loader)],
    }

    # Train model
    train_model(
        model,
        train_loader,
        opt,
        epochs=max_epochs,
        device=device,
        callbacks=callbacks,
    )

    # Save model and results
    write_model(model, args.out_path, file_name="trained.pt", overwrite=True)
    write_df(df, args.csv_path, file_name="results.csv", overwrite=True)


def get_datasets(
    dataset: str,
    val_size: int = None,
    device: str = "cpu",
    seed: int = None,
):
    if dataset == "mnist":
        return _get_mnist(val_size, device, seed)
    elif dataset == "cifar10":
        return _get_cifar10(val_size, device, seed)
    else:
        raise ValueError(f"Unknown dataset {dataset}.")


def _get_mnist(val_size: int = None, device: str = "cpu", seed: int = None):
    import torch
    from torch.utils.data import random_split
    from torchvision.datasets import MNIST
    from torchvision.transforms import Compose, ToTensor, Normalize

    logger.info("Loading MNIST dataset.")
    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    train_data = MNIST("data", train=True, download=True, transform=transform)
    test_data = MNIST("data", train=False, download=True, transform=transform)

    # Split training data into training and validation if required
    if val_size is not None and val_size > 0:
        logger.info("Sampling a validation set of size %d.", val_size)
        sizes = [len(train_data) - val_size, val_size]
        generator = torch.Generator(device).manual_seed(seed)
        train_data, val_data = random_split(train_data, sizes, generator=generator)
        return train_data, val_data, test_data

    # Otherwise, return just the training and test data
    return train_data, test_data


def _get_cifar10(val_size: int = None, device: str = "cpu", seed: int = None):
    import torch
    from torch.utils.data import random_split
    from torchvision.transforms import (
        Compose,
        ToTensor,
        Normalize,
        RandomCrop,
        RandomHorizontalFlip,
    )
    from torchvision.datasets import CIFAR10

    logger.info("Loading CIFAR10 dataset.")
    transform_fns = [
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    transform = Compose(transform_fns)
    train_data = CIFAR10("data", train=True, download=True, transform=transform)
    test_data = CIFAR10("data", train=False, download=True, transform=transform)

    # Split training data into training and validation if required
    if val_size is not None and val_size > 0:
        logger.info("Sampling a validation set of size %d.", val_size)
        sizes = [len(train_data) - val_size, val_size]
        generator = torch.Generator(device).manual_seed(seed)
        train_data, val_data = random_split(train_data, sizes, generator=generator)
        return train_data, val_data, test_data

    # Otherwise, return just the training and test data
    return train_data, test_data


def get_dimensions(dataset: str):
    if dataset == "mnist":
        return 1, 28, 28
    elif dataset == "cifar10":
        return 3, 32, 32
    else:
        raise ValueError(f"Unknown dataset {dataset}.")


def make_data_loader(dataset, batch_size: int, device: str, seed: int = None, **kwargs):
    import torch
    from torch.utils.data import DataLoader

    logger.info("Creating data loader with batch size %d.", batch_size)
    device = torch.device(device)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        generator=torch.Generator(device).manual_seed(seed),
        pin_memory=True,
        **kwargs,
    )


def get_optimiser(model, optimiser, learning_rate: float):
    import torch

    if optimiser == "sgd":
        logger.info("Using SGD optimiser with learning rate %.3f.", learning_rate)
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimiser == "adam":
        logger.info("Using Adam optimiser with learning rate %.3f.", learning_rate)
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimiser {optimiser}.")


def train_loss_cb(df: pd.DataFrame, every: int = 100) -> None:
    assert every is None or every > 0

    def _train_loss_cb(model, iteration: int, loss: float):
        if every is not None and iteration % every != 0:
            return

        logger.debug(f"Train loss: {loss:.4f}")
        df.loc[iteration, "train_loss"] = loss

    return _train_loss_cb


def val_loss_cb(df: pd.DataFrame, val_loader, every: int = None) -> None:
    from torch.utils.data import DataLoader
    from nets import MaskedNetwork
    from nets.nn import evaluate_model

    assert isinstance(val_loader, DataLoader)
    assert every is None or every > 0

    def _val_loss_cb(model: MaskedNetwork, iteration: int, _loss: float):
        if every is not None and iteration % every != 0:
            return

        val_loss, val_acc = evaluate_model(model, val_loader)
        logger.info(f"Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.4f}")
        df.loc[iteration, "val_loss"] = val_loss
        df.loc[iteration, "val_acc"] = val_acc

    return _val_loss_cb


def test_loss_cb(df: pd.DataFrame, test_loader, every: int = None) -> None:
    from torch.utils.data import DataLoader
    from nets import MaskedNetwork
    from nets.nn import evaluate_model

    assert isinstance(test_loader, DataLoader)
    assert every is None or every > 0

    def _test_loss_cb(model: MaskedNetwork, iteration: int, _loss: float):
        if every is not None and iteration % every != 0:
            return

        test_loss, test_acc = evaluate_model(model, test_loader)
        logger.info(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")
        df.loc[iteration, "test_loss"] = test_loss
        df.loc[iteration, "test_acc"] = test_acc

    return _test_loss_cb
