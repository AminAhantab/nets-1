import logging
from typing import Callable, Dict, List

import pandas as pd

from ..args import TrainArgs
from ..config import configure_logger, configure_seed, configure_torch
from ..io import load_model, write_df, write_model
from ..mapper import hydrate_dataset, hydrate_optimiser

logger = logging.getLogger("nets_cli.train")


def run_train(args: TrainArgs) -> None:
    # Configure environment
    configure_logger(args)
    device = configure_torch(args.no_cuda)
    configure_seed(args.seed)

    from torch.utils.data import DataLoader
    from nets.nn.train import train_model

    # Get relevant arguments
    model_path = args.model_path
    dataset = args.dataset
    optimiser = args.optimiser
    learning_rate = args.learning_rate
    max_epochs = args.max_epochs
    batch_size = args.batch_size
    val_size = args.val_size

    # Load model from disk
    model = load_model(model_path)
    model.to(device)

    # Initialise data
    train_data, val_data, test_data = hydrate_dataset(dataset, val_size=val_size)

    # Initialise optimiser
    optimiser_class = hydrate_optimiser(optimiser)
    opt = optimiser_class(model.parameters(), lr=learning_rate)

    # Initialise the data loaders
    kwargs = {"pin_memory": True, "num_workers": 0}
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_data, batch_size=99_999, **kwargs)
    test_loader = DataLoader(test_data, batch_size=99_999, **kwargs)

    # Initialise the callbacks
    results = pd.DataFrame()
    callbacks = init_callbacks(
        results,
        args,
        model,
        train_loader,
        val_loader,
        test_loader,
        device=device,
    )

    # Train model
    train_model(
        model,
        train_loader,
        opt,
        device=device,
        callbacks=callbacks,
    )

    # Save model and results
    write_model(model, args.out_path, file_name="trained.pt", overwrite=True)
    write_df(results, args.csv_path, file_name="results.csv", overwrite=True)


def init_callbacks(
    df: pd.DataFrame,
    args: TrainArgs,
    model,
    train_loader,
    val_loader,
    test_loader,
    device: str = None,
) -> Dict[str, List[Callable]]:
    from nets import callbacks as cb

    # Add initial stats to dataframe
    add_initial_stats(df, model, train_loader, val_loader, test_loader, device=device)

    # Get relevant arguments
    log_every = args.log_every
    log_val_every = args.log_val_every
    log_test_every = args.log_test_every
    max_epochs = args.max_epochs
    max_iterations = args.max_iterations

    iteration_callbacks = [
        cb.log_train_loss(df, every=log_every),
        cb.log_val_loss(df, val_loader, every=log_val_every, device=device),
        cb.log_test_loss(df, test_loader, every=log_test_every, device=device),
    ]

    epoch_callbacks = [
        cb.log_val_loss(df, val_loader, device=device, every=1),
        cb.log_test_loss(df, test_loader, device=device, every=1),
    ]

    early_stopping_criteria = [
        cb.max_epochs(df, max_epochs),
        cb.max_iterations(df, max_iterations),
    ]

    if device == "cuda":
        epoch_callbacks.append(cb.log_gpu_memory())

    callbacks = {
        "iteration": iteration_callbacks,
        "epoch": epoch_callbacks,
        "early_stopping": early_stopping_criteria,
    }

    return callbacks


def add_initial_stats(
    df: pd.DataFrame,
    model,
    train_loader,
    val_loader,
    test_loader,
    device=None,
):
    from nets.nn import evaluate_model

    logger.info("Evaluating initial model")
    train_loss, train_acc = evaluate_model(model, train_loader, device=device)
    val_loss, val_acc = evaluate_model(model, val_loader, device=device)
    test_loss, test_acc = evaluate_model(model, test_loader, device=device)
    logger.debug("Train loss: %f, Train accuracy: %f", train_loss, train_acc)
    logger.debug("Val loss: %f, Val accuracy: %f", val_loss, val_acc)
    logger.debug("Test loss: %f, Test accuracy: %f", test_loss, test_acc)

    df.loc[0, "train_loss"] = train_loss
    df.loc[0, "train_acc"] = train_acc
    df.loc[0, "val_loss"] = val_loss
    df.loc[0, "val_acc"] = val_acc
    df.loc[0, "test_loss"] = test_loss
    df.loc[0, "test_acc"] = test_acc
    df.loc[0, "epoch"] = 0

    logger.info("Initial val loss: %f", val_loss)
    logger.info("Initial val accuracy: %f", val_acc)
    logger.info("Initial test loss: %f", test_loss)
    logger.info("Initial test accuracy: %f", test_acc)
