import logging
import os
from typing import Callable, Dict, List

import pandas as pd

from ..args import IterativeMagnitudePruningArgs, PruneArgs, TrainArgs
from ..config import configure_logger, configure_seed, configure_torch
from ..io import load_model, write_df, write_model
from ..hydrator import hydrate_dataset, hydrate_optimiser, hydrate_prune_method

logger = logging.getLogger("nets_cli.imp")


def run_imp(args: IterativeMagnitudePruningArgs) -> None:
    # Configure environment
    configure_logger(args)
    device = configure_torch(args.no_cuda)
    configure_seed(args.seed)

    from torch.utils.data import DataLoader

    # Get relevant arguments
    model_path = args.model_path

    # Training hyperparameters
    dataset = args.dataset
    optimiser = args.optimiser
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    val_size = args.val_size

    # Pruning hyperparameters
    cycles = args.cycles
    reinit = args.reinit

    # Load model from disk
    model = load_model(model_path)

    # Store intiial weights
    initial_weights = []
    for layer in model.layers:
        initial_weights.append(layer.weight.data.clone())

    # Move model to device
    model.to(device)

    # Initialise data
    train_data, val_data, test_data = hydrate_dataset(dataset, val_size=val_size)

    # Initialise the data loaders
    kwargs = {"pin_memory": True, "num_workers": 0}
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_data, batch_size=batch_size, **kwargs)
    test_loader = DataLoader(test_data, batch_size=batch_size, **kwargs)
    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}

    # E.g. three cycles:
    # 0. Train model, prune model, save model
    # 1. Train model, prune model, save model
    # 2. Train model, prune model, save model
    # 3. Train final model, save model
    for cycle in range(cycles + 1):
        if cycle == cycles:
            logger.info("Training final model.")
        else:
            logger.info(f"Starting cycle {cycle} of {cycles}")

        # Initialise optimiser
        optimiser_class = hydrate_optimiser(optimiser)
        opt = optimiser_class(model.parameters(), lr=learning_rate)

        out_path = os.path.join(args.out_path, f"trained-{cycle}.pt")
        csv_path = os.path.join(args.out_path, f"training-{cycle}.csv")
        train(args, model, loaders, opt, device, out_path, csv_path)

        if cycle == cycles:
            break

        # Prune model
        logger.info(f"Pruning model (cycle {cycle} of {cycles})")
        out_path = os.path.join(args.out_path, f"pruned-{cycle}.pt")
        prune(args, model, out_path)

        # Reinitialise weights
        if reinit:
            reinitialise_weights(model, initial_weights, device)

    # Save model and results
    out_path = os.path.join(args.out_path, f"trained-{cycles}.pt")
    write_model(model, out_path, file_name="trained.pt", overwrite=True)


def train(args: TrainArgs, model, loaders: Dict, opt, device, out_path, csv_path):
    from nets.nn.train import train_model

    # Initialise the callbacks
    results = pd.DataFrame()

    callbacks = init_callbacks(
        results,
        args,
        model,
        loaders["train"],
        loaders["val"],
        loaders["test"],
        device=device,
    )

    # Train model
    train_model(
        model,
        loaders["train"],
        opt,
        device=device,
        callbacks=callbacks,
    )

    results["density"] = model.density()

    # Write training results to disk
    write_model(model, out_path, file_name=f"trained.pt", overwrite=True)
    write_df(results, csv_path, file_name=f"results.csv", overwrite=True)


def prune(args: PruneArgs, model, out_path: str):
    # Get relevant arguments
    criterion = args.criterion
    threshold = args.threshold
    count = args.count
    fraction = args.fraction

    # Get pruning method
    prune_method = hydrate_prune_method(criterion, threshold, count, fraction)

    # Prune model
    logger.info("Initial density: {:.2f}%".format(100 * model.density()))
    removed_weights = prune_method(model)

    # Log results
    logger.debug("Removed {} weights".format(removed_weights))
    logger.info("Final density: {:.2f}%".format(100 * model.density()))

    # Save model to disk
    write_model(model, out_path, file_name="pruned.pt", overwrite=True)


def reinitialise_weights(model, initial_weights, device):
    from nets.nn import MaskedNetwork

    assert isinstance(model, MaskedNetwork)

    for layer, initial_weight in zip(model.layers, initial_weights):
        layer.reset(initial_weight.clone().to(device))


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
        cb.max_epochs(max_epochs),
        cb.max_iterations(max_iterations),
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
