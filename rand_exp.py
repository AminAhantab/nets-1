import logging
import os

from nets_cli.runners import methods
from nets_cli.config import configure_seed, configure_torch

logger = logging.getLogger("nets_experiments")


def run(output_path: str, trial: int):
    logging.basicConfig(level=logging.DEBUG)
    device = configure_torch()
    configure_seed(42 + trial)

    os.makedirs(output_path, exist_ok=True)

    # Initialise lenet model
    model = methods.init("lenet", "mnist", density=0.3, bias=False)
    model, train_results = methods.train(
        model,
        dataset="mnist",
        val_size=5_000,
        optimiser="sgd",
        learning_rate=1e-3,
        batch_size=60,
        max_iterations=20_000,
        max_epochs=None,
        log_every=100,
        log_val_every=500,
        log_test_every=500,
        device=device,
    )

    # Save results
    train_results.to_csv(os.path.join(output_path, f"random_20k_{trial}.csv"))


if __name__ == "__main__":
    output_path = "results/nets_experiments"

    for trial in range(1, 2):
        run(output_path, trial)
