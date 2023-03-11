import os
import logging
import argparse

from nets_cli.runners import methods
from nets_cli.config import configure_seed, configure_torch

logger = logging.getLogger("nets_experiments")

NETS_SEARCH_FILE = "nets_search.csv"
NETS_TRAIN_FILE = "nets_train.csv"
OVERP_TRAIN_FILE = "overp_train.csv"

ARCHITECTURE = "lenet"
DATASET = "mnist"
VALIDATION_SIZE = 5_000
OPTIMISER = "sgd"
LEARNING_RATE = 1e-3
BATCH_SIZE = 64

POP_SIZE = 15
INITIAL_DENSITY = 1.0
TARGET_DENSITY = 0.5
ELITISM = 3
P_CROSSOVER = 0.5
MR_NOISE = 0.1
MR_RANDOM = 0.1
MR_DISABLE = 0.01
MR_ENABLE = 0
MR_NOISE_SCALE = 0.1
MAX_GENERATIONS = 50
MAX_NO_CHANGE = 10
MIN_FITNESS = 0.0

MAX_ITERATIONS = 5_000
MAX_EPOCHS = None
LOG_EVERY = 100
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
    overp_train_results_path = os.path.join(output_path, f"{trial}_{OVERP_TRAIN_FILE}")

    os.makedirs(output_path, exist_ok=True)

    # overparameterised = methods.init(
    #     architecture=ARCHITECTURE, dataset=DATASET, density=1.0, bias=False
    # )

    # overparameterised, overp_train_results = methods.train(
    #     model=overparameterised,
    #     dataset=DATASET,
    #     val_size=VALIDATION_SIZE,
    #     optimiser=OPTIMISER,
    #     learning_rate=LEARNING_RATE,
    #     batch_size=BATCH_SIZE,
    #     max_iterations=MAX_ITERATIONS,
    #     max_epochs=MAX_EPOCHS,
    #     log_every=LOG_EVERY,
    #     log_val_every=LOG_VAL_EVERY,
    #     log_test_every=LOG_TEST_EVERY,
    #     device=device,
    # )

    # overp_train_results.to_csv(overp_train_results_path)

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
        mr_enable=MR_ENABLE,
        mr_noise_scale=MR_NOISE_SCALE,
        max_generations=MAX_GENERATIONS,
        max_no_change=MAX_NO_CHANGE,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="results_3")
    args = parser.parse_args()
    run(args.trials, args.out_dir)
