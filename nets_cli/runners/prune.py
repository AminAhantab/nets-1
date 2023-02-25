import logging

from ..args import PruneArgs
from ..io import load_model, write_model

from .methods import prune

logger = logging.getLogger("nets_cli.prune")


def run_prune(args: PruneArgs) -> None:
    # Get relevant arguments
    model_path = args.model_path
    criterion = args.criterion
    threshold = args.threshold
    count = args.count
    fraction = args.fraction

    # Load model from disk
    model = load_model(model_path)

    # Prune model
    model = prune(model, criterion, threshold, count, fraction)

    # Save model to disk
    write_model(model, args.out_path, file_name="pruned.pt", overwrite=True)
