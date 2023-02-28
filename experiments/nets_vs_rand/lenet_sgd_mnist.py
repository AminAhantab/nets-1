import logging
import argparse
import os

from nets_cli.runners import methods
from nets_cli.config import configure_seed, configure_torch

logger = logging.getLogger("nets_experiments")

NETS_SEARCH_FILE = "nets_search.csv"
NETS_TRAIN_FILE = "nets_train.csv"
RAND_TRAIN_FILE = "rand_train.csv"

ARCHITECTURE = "lenet"
DATASET = "mnist"
VALIDATION_SIZE = 5_000
OPTIMISER = "sgd"
LEARNING_RATE = 1e-3
BATCH_SIZE = 64

POP_SIZE = 5
INITIAL_DENSITY = 1.0
TARGET_DENSITY = 0.1
ELITISM = 2
P_CROSSOVER = 0.5
MR_NOISE = 0.1
MR_RANDOM = 0.1
MR_DISABLE = 0.2
MR_NOISE_SCALE = 0.1
MAX_GENERATIONS = 200
MIN_FITNESS = 0.0

MAX_ITERATIONS = 50_000
MAX_EPOCHS = None
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

    # Search for an initialisation using NeTS
    initialisation, search_results = methods.search(
        arch=ARCHITECTURE,
        dataset=DATASET,
        val_size=VALIDATION_SIZE,
        optimiser=OPTIMISER,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        pop_size=POP_SIZE,
        initial_density=INITIAL_DENSITY,
        target_density=TARGET_DENSITY,
        elitism=ELITISM,
        p_crossover=P_CROSSOVER,
        mr_noise=MR_NOISE,
        mr_random=MR_RANDOM,
        mr_disable=MR_DISABLE,
        mr_noise_scale=MR_NOISE_SCALE,
        max_generations=MAX_GENERATIONS,
        min_fitness=MIN_FITNESS,
        device=device,
    )

    # Write the search results to a file
    search_results.to_csv(search_results_path)

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
    train_results.to_csv(nets_train_results_path)

    rand_model = methods.init(
        architecture=ARCHITECTURE,
        dataset=DATASET,
        density=model.density(),
        bias=False,
    )
    rand_model, rand_train_results = methods.train(
        model=rand_model,
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

    rand_train_results.to_csv(rand_train_results_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()
    run(args.trial, args.out_dir)
