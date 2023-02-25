import logging
import os

from nets_cli.runners import methods
from nets_cli.config import configure_seed, configure_torch

logger = logging.getLogger("nets_experiments")


def run(trial: int):
    logging.basicConfig(level=logging.DEBUG)
    device = configure_torch()
    configure_seed(42 + trial)

    output_path = "/scratch/users/k1502897/nets_2/results/nets_experiments"
    os.makedirs(output_path, exist_ok=True)

    # Oneshot magnitude pruning====

    # Initialise lenet model
    model = methods.init("lenet", "mnist", density=1.0, bias=False)
    model, results = methods.iterative_magnitude_prune(
        model,
        dataset="mnist",
        val_size=5_000,
        optimiser="adam",
        learning_rate=1.2e-3,
        batch_size=60,
        max_iterations=50_000,
        max_epochs=None,
        log_every=100,
        log_val_every=500,
        log_test_every=500,
        cycles=1,
        criterion="magnitude",
        threshold=None,
        count=None,
        fraction=0.7,
        reinit=True,
        device=device,
    )

    # Save results
    results.to_csv(os.path.join(output_path, f"oneshot_mp_{trial}.csv"))

    # Iterative magnitude pruning ====

    # Initialise lenet model
    model = methods.init("lenet", "mnist", density=1.0, bias=False)
    model, results = methods.iterative_magnitude_prune(
        model,
        dataset="mnist",
        val_size=5_000,
        optimiser="adam",
        learning_rate=1.2e-3,
        batch_size=60,
        max_iterations=50_000,
        max_epochs=None,
        log_every=100,
        log_val_every=500,
        log_test_every=500,
        cycles=10,
        criterion="magnitude",
        threshold=None,
        count=None,
        fraction=0.2,
        reinit=True,
        device=device,
    )

    # # Save results
    results.to_csv(os.path.join(output_path, f"iterative_mp_{trial}.csv"))

    # NeTS and Train ====
    model, search_results = methods.search(
        "lenet",
        "mnist",
        val_size=5_000,
        optimiser="adam",
        learning_rate=1.2e-3,
        batch_size=60,
        pop_size=10,
        initial_density=0.6,
        target_density=0.3,
        elitism=2,
        p_crossover=0.5,
        mr_noise=0.1,
        mr_random=0.1,
        mr_disable=0.2,
        mr_noise_scale=0.1,
        max_generations=10,
        min_fitness=0.0,
        device=device,
    )

    # Save results
    search_results.to_csv(os.path.join(output_path, f"nets_search_{trial}.csv"))

    model, train_results = methods.train(
        model,
        dataset="mnist",
        val_size=5_000,
        optimiser="adam",
        learning_rate=1.2e-3,
        batch_size=60,
        max_iterations=50_000,
        max_epochs=None,
        log_every=100,
        log_val_every=500,
        log_test_every=500,
        device=device,
    )

    # Save results
    train_results.to_csv(os.path.join(output_path, f"nets_train_{trial}.csv"))


if __name__ == "__main__":
    for trial in range(1, 6):
        run(trial)