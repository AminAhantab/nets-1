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
MAX_GENERATIONS = 15
MIN_FITNESS = 0.0

MAX_ITERATIONS = 10_000
MAX_EPOCHS = None
LOG_EVERY = 500
LOG_VAL_EVERY = 500
LOG_TEST_EVERY = 500


def run(trial: int, our_dir: str):
    logging.basicConfig(level=logging.DEBUG)
    device = configure_torch()
    configure_seed(235234 + trial)

    import torch

    # get filename without extension
    filename = os.path.basename(__file__)
    output_path = os.path.join(our_dir, filename)

    def mkpath(*args):
        return os.path.join(output_path, *args)

    # Make output directories
    os.makedirs(output_path, exist_ok=True)

    INIT_SUFFIX = "init.pt"
    TRAIN_SUFFIX = "train.csv"
    TRAINED_SUFFIX = "trained.pt"
    SEARCH_SUFFIX = "search.csv"

    # Overparametrised initialisation =============================================================
    FILE_PREFIX = "overp"

    # Initialise a random model
    overp_init = methods.init(
        architecture=ARCHITECTURE,
        dataset=DATASET,
        density=INITIAL_DENSITY,
        bias=False,
    )

    # Save the initialisation
    overp_init_path = mkpath(f"{trial}_{FILE_PREFIX}_{INIT_SUFFIX}")
    torch.save(overp_init, overp_init_path)

    # Train the random model
    overp_init, overp_results = methods.train(
        model=overp_init,
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
    torch.save(overp_init, mkpath(f"{trial}_{FILE_PREFIX}_{TRAINED_SUFFIX}"))
    overp_results.to_csv(mkpath(f"{trial}_{FILE_PREFIX}_{TRAIN_SUFFIX}"))

    # NeTS initialisation =========================================================================
    FILE_PREFIX = "nets"

    # Search for an initialisation using NeTS
    nets_init, nets_search_results = methods.search(
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
        mr_enable=0,
        max_no_change=None,
        mr_noise_scale=MR_NOISE_SCALE,
        max_generations=MAX_GENERATIONS,
        min_fitness=MIN_FITNESS,
        device=device,
    )

    # Save results
    nets_search_results.to_csv(mkpath(f"{trial}_{FILE_PREFIX}_{SEARCH_SUFFIX}"))
    torch.save(nets_init, mkpath(f"{trial}_{FILE_PREFIX}_{INIT_SUFFIX}"))

    # Train the best initialisation
    nets_init, nets_train_results = methods.train(
        model=nets_init,
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
    nets_train_results.to_csv(mkpath(f"{trial}_{FILE_PREFIX}_{TRAIN_SUFFIX}"))
    torch.save(nets_init, mkpath(f"{trial}_{FILE_PREFIX}_{TRAINED_SUFFIX}"))

    # Random sparse initialisation ================================================================
    FILE_PREFIX = "sparse"

    # Initialise a random sparse model
    sparse_init = methods.init(
        architecture=ARCHITECTURE,
        dataset=DATASET,
        density=sparse_init.density(),
        bias=False,
    )

    # Save the initialisation
    torch.save(sparse_init, mkpath(f"{trial}_{FILE_PREFIX}_{INIT_SUFFIX}"))

    # Train the random sparse model
    sparse_init, sparse_results = methods.train(
        model=sparse_init,
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
    sparse_results.to_csv(mkpath(f"{trial}_{FILE_PREFIX}_{TRAIN_SUFFIX}"))
    torch.save(sparse_init, mkpath(f"{trial}_{FILE_PREFIX}_{TRAINED_SUFFIX}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()
    run(args.trial, args.out_dir)
