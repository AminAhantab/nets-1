import logging
import os

from nets_cli.runners import methods
from nets_cli.config import configure_seed, configure_torch

logger = logging.getLogger("nets_experiments")

OUTPUT_PATH = "results/b_test"
TRAIN_FILE = "random_train.csv"

ARCHITECTURE = "lenet"
DATASET = "mnist"
VALIDATION_SIZE = 5_000
OPTIMISER = "sgd"
LEARNING_RATE = 0.001
BATCH_SIZE = 64
MAX_ITERATIONS = 5_000
MAX_EPOCHS = None
LOG_EVERY = 100
LOG_VAL_EVERY = 100
LOG_TEST_EVERY = 100


def run(trial: int):
    logging.basicConfig(level=logging.DEBUG)
    device = configure_torch()
    configure_seed(42 + trial)

    output_path = "results/b_test"
    os.makedirs(output_path, exist_ok=True)

    # Search for an initialisation using NeTS
    initialisation = methods.init(
        architecture=ARCHITECTURE, dataset=DATASET, density=0.46, bias=False
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
        max_epochs=MAX_EPOCHS,
        log_every=LOG_EVERY,
        log_val_every=LOG_VAL_EVERY,
        log_test_every=LOG_TEST_EVERY,
        device=device,
    )

    # Save results
    train_results_path = os.path.join(OUTPUT_PATH, TRAIN_FILE)
    train_results.to_csv(train_results_path)


if __name__ == "__main__":
    for trial in range(1, 2):
        run(trial)
