import logging

import torch

from ..args import TrainArgs
from ..io import load_model, write_df, write_model

from .methods import train

logger = logging.getLogger("nets_cli.train")


def run_train(args: TrainArgs, device: torch.device) -> None:
    # Get model path and hyperparameters
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

    # Load model from disk
    model = load_model(model_path)

    # Train model
    model, results = train(
        model,
        dataset,
        optimiser,
        learning_rate,
        batch_size,
        val_size,
        max_epochs,
        max_iterations,
        log_every,
        log_val_every,
        log_test_every,
        device,
    )

    # Save model and results
    write_model(model, args.out_path, file_name="trained.pt", overwrite=True)
    write_df(results, args.csv_path, file_name="results.csv", overwrite=True)
