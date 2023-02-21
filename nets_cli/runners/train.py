import logging

from nets_cli.args import TrainArgs
from nets_cli.runners.common import (
    configure_logger,
    configure_seed,
    configure_torch,
    load_model,
    write_model,
)

logger = logging.getLogger("nets_cli.train")


def run_train(args: TrainArgs) -> None:
    # Configure environment
    configure_logger(args)
    device = configure_torch()
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
    train_data, _ = get_dataset(dataset)

    # Sample validation
    # TODO: Implement

    # Load model
    model = load_model(model_path)
    logger.info("Moving model to device %s.", device)
    model.to(device)

    # Initialise optimiser
    opt = get_optimiser(model, optimiser, learning_rate)

    # Initialise data loader
    train_loader = make_data_loader(
        train_data,
        batch_size=batch_size,
        device=device,
        seed=args.seed,
    )

    # Train model
    # TODO: Implement callbacks (early stopping, checkpointing, validation, etc.)
    # NOTE: See code in alexjackson1/stronglth:stronglth/nn/train.py
    train_model(model, train_loader, opt, epochs=max_epochs, device=device)

    # Evaluate model
    # TODO: Implement

    # Save model
    write_model(model, args.out_path, file_name="trained.pt", overwrite=True)


def get_dataset(dataset: str):
    from torchvision import datasets, transforms

    if dataset == "mnist":
        transform_fns = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
        transform = transforms.Compose(transform_fns)
        logger.info("Using MNIST dataset.")
        return datasets.MNIST(
            "data", train=True, download=True, transform=transform
        ), datasets.MNIST("data", train=False, download=True, transform=transform)
    elif dataset == "cifar10":
        transform_fns = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
        transform = transforms.Compose(transform_fns)
        logger.info("Using CIFAR10 dataset.")
        return datasets.CIFAR10(
            "data",
            train=True,
            download=True,
            transform=transform,
        ), datasets.CIFAR10(
            "data",
            train=False,
            download=True,
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown dataset {dataset}.")


def get_dimensions(dataset: str):
    if dataset == "mnist":
        return 1, 28, 28
    elif dataset == "cifar10":
        return 3, 32, 32
    else:
        raise ValueError(f"Unknown dataset {dataset}.")


def make_data_loader(dataset, batch_size: int, device: str, seed: int = None):
    import torch
    from torch.utils.data import DataLoader

    logger.info("Creating data loader with batch size %d.", batch_size)
    device = torch.device(device)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        generator=torch.Generator(device).manual_seed(seed),
        pin_memory=True,
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
