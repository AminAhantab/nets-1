import logging
import os

import torch

from nets_cli.runners.methods import iterative_magnitude_prune

from ..args import IterativeMagnitudePruningArgs
from ..io import load_model, write_df, write_model

logger = logging.getLogger("nets_cli.imp")


def run_imp(args: IterativeMagnitudePruningArgs, device: torch.device) -> None:
    # Get model hyperparameters
    model_path = args.model_path
    dataset = args.dataset

    # Get gradient descent hyperparameters
    optimiser = args.optimiser
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    val_size = args.val_size
    max_epochs = args.max_epochs
    max_iterations = args.max_iterations
    log_every = args.log_every
    log_val_every = args.log_val_every
    log_test_every = args.log_test_every

    # Get pruning hyperparameters
    cycles = args.cycles
    reinit = args.reinit
    criterion = args.criterion
    threshold = args.threshold
    count = args.count
    fraction = args.fraction

    # Load model from disk
    model = load_model(model_path)

    model, results = iterative_magnitude_prune(
        model,
        dataset=dataset,
        val_size=val_size,
        optimiser=optimiser,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_epochs=max_epochs,
        max_iterations=max_iterations,
        log_every=log_every,
        log_val_every=log_val_every,
        log_test_every=log_test_every,
        cycles=cycles,
        criterion=criterion,
        threshold=threshold,
        count=count,
        fraction=fraction,
        reinit=reinit,
        device=device,
    )

    # Save model and results
    out_path = os.path.join(args.out_path, f"result.pt")
    write_model(model, out_path, file_name="trained.pt", overwrite=True)
    write_df(results, args.csv_path, file_name="results.csv", overwrite=True)
