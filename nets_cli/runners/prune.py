import logging

from ..args import PruneArgs
from ..config import configure_logger, configure_seed, configure_torch
from ..hydrator import hydrate_prune_method
from ..io import load_model, write_model

logger = logging.getLogger("nets_cli.prune")


def run_prune(args: PruneArgs) -> None:
    # Configure environment
    configure_logger(args)
    configure_torch()
    configure_seed(args.seed)

    # Get relevant arguments
    model_path = args.model_path
    criterion = args.criterion
    threshold = args.threshold
    count = args.count
    fraction = args.fraction

    # Load model from disk
    model = load_model(model_path)

    # Prune model
    logger.info("Initial density: {:.2f}%".format(100 * model.density()))
    prune_method = hydrate_prune_method(criterion, threshold, count, fraction)
    removed_weights = prune_method(model)
    logger.info("Removed {} weights".format(removed_weights))
    logger.info("Final density: {:.2f}%".format(100 * model.density()))

    # Save model to disk
    write_model(model, args.out_path, file_name="pruned.pt", overwrite=True)
