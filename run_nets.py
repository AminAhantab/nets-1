import logging
import os

from nets_cli.runners import methods
from nets_cli.config import configure_seed, configure_torch

logger = logging.getLogger("nets_experiments")

OUTPUT_PATH = "results/conv_2_adam"
NETS_SEARCH_FILE = "nets_search.csv"
NETS_TRAIN_FILE = "nets_train.csv"
RAND_TRAIN_FILE = "rand_train_.csv"

ARCHITECTURE = "conv-2"
DATASET = "cifar10"
VALIDATION_SIZE = 5_000
OPTIMISER = "adam"
LEARNING_RATE = 2e-4
BATCH_SIZE = 64

POP_SIZE = 5
INITIAL_DENSITY = 1.0
TARGET_DENSITY = 0.3
ELITISM = 2
P_CROSSOVER = 0.5
MR_NOISE = 0.1
MR_RANDOM = 0.1
MR_DISABLE = 0.2
MR_NOISE_SCALE = 0.1
MAX_GENERATIONS = 15
MIN_FITNESS = 0.0

MAX_ITERATIONS = 5_000
MAX_EPOCHS = None
LOG_EVERY = 100
LOG_VAL_EVERY = 100
LOG_TEST_EVERY = 100


def run(trial: int, target: float):
    logging.basicConfig(level=logging.DEBUG)
    device = configure_torch()
    configure_seed(235234 + trial)

    output_path = os.path.join(OUTPUT_PATH, f"t{target}")
    search_results_path = os.path.join(output_path, NETS_SEARCH_FILE)
    nets_train_results_path = os.path.join(output_path, NETS_TRAIN_FILE)
    rand_train_results_path = os.path.join(output_path, RAND_TRAIN_FILE)

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
        target_density=target,
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
        write_every=None,
        device=device,
    )

    # Save results
    train_results.to_csv(nets_train_results_path)

    last_test_acc = train_results["test_acc"].iloc[-1]

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
        write_every=None,
        device=device,
    )

    rand_train_results.to_csv(rand_train_results_path)

    print(f"NeTS Best Density: {model.density()}")
    print(f"NeTS Best Test Accuracy: {last_test_acc}")
    print(f"Random model Test Accuracy: {rand_train_results['test_acc'].iloc[-1]}")


if __name__ == "__main__":
    for i, target_density in enumerate([0.05, 0.1, 0.2, 0.3]):
        run(i, target_density)
