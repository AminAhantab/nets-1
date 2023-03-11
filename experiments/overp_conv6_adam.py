import logging
import argparse
import os

from nets_cli.runners import methods
from nets_cli.config import configure_seed, configure_torch

logger = logging.getLogger("nets_experiments")

NETS_SEARCH_FILE = "nets_search.csv"
NETS_TRAIN_FILE = "nets_train.csv"
RAND_TRAIN_FILE = "rand_train.csv"

ARCHITECTURE = "conv-6"
DATASET = "cifar10"
VALIDATION_SIZE = 5_000
OPTIMISER = "adam"
LEARNING_RATE = 3e-4
BATCH_SIZE = 64

MAX_ITERATIONS = 50_000
LOG_EVERY = 500
LOG_VAL_EVERY = 500
LOG_TEST_EVERY = 500


def run(trial: int, our_dir: str):
    logging.basicConfig(level=logging.DEBUG)
    device = configure_torch()
    configure_seed(235234 + trial)

    # get filename without extension
    filename = os.path.basename(__file__)
    output_path = os.path.join(our_dir, filename)
    search_results_path = os.path.join(output_path, f"{trial}_{NETS_SEARCH_FILE}")
    nets_train_results_path = os.path.join(output_path, f"{trial}_{NETS_TRAIN_FILE}")
    rand_train_results_path = os.path.join(output_path, f"{trial}_{RAND_TRAIN_FILE}")

    os.makedirs(output_path, exist_ok=True)

    initialisation = methods.init(
        architecture=ARCHITECTURE,
        dataset=DATASET,
        density=1.0,
        bias=False,
    )

    # Train the best initialisation
    model, train_results = methods.train(
        model=initialisation,
        dataset=DATASET,
        val_size=VALIDATION_SIZE,
        optimiser=OPTIMISER,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        max_iterations=MAX_ITERATIONS,
        max_epochs=None,
        log_every=LOG_EVERY,
        log_val_every=LOG_VAL_EVERY,
        log_test_every=LOG_TEST_EVERY,
        device=device,
    )

    # Save results
    train_results.to_csv(nets_train_results_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="amin-results")
    args = parser.parse_args()
    run(args.trial, args.out_dir)
