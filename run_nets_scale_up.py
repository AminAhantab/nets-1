import os
import logging

import pandas as pd

from data import mnist
from models import LeNetFeedForwardNetwork
from nets import neuroevolution_ts, callbacks as nets_callbacks
from params import HyperParameters
import utils

EXPERIMENT_NAME = f"nets-mnist-scale-up"

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("nets")
logger.info("Starting experiment %s", EXPERIMENT_NAME)


def mnist_experiment(params: HyperParameters) -> pd.DataFrame:
    # Set seed if provided, otherwise use a random seed
    params.seed = utils.set_seed(params.seed)

    # Create directories if they don't exist
    utils.create_dirs(params.data_dir, params.results_dir)

    # Download and load MNIST dataset
    mnist_data = mnist.load(params.data_dir, download=True, val_size=params.val_size)
    train_data, val_data, test_data = mnist_data
    train_size, val_size, test_size = len(train_data), len(val_data), len(test_data)
    logger.info("MNIST: train=%d, val=%d, test=%d", train_size, val_size, test_size)

    # Write hyperparameters to file
    file_name = "hyperparams.json"
    logger.info("Writing hyperparameters to file %s", f"results_dir/{file_name}")
    utils.write_params(params, dir=params.results_dir, name=file_name)

    # Create callbacks
    test_df = nets_callbacks.init_test_df()
    cbks = [
        nets_callbacks.test_callback_parallel(test_df, train_data, test_data, epochs=1),
        nets_callbacks.log_callback(),
    ]

    # Run neuroevolution
    model = LeNetFeedForwardNetwork(bias=False)
    results = neuroevolution_ts(
        model,
        train_data,
        val_data,
        batch_size=params.batch_size,
        pop_size=params.pop_size,
        elitism=params.elitism,
        p_crossover=params.p_crossover,
        mr_weight_noise=params.mr_weight_noise,
        mr_weight_rand=params.mr_weight_rand,
        mr_weight_zero=params.mr_weight_zero,
        max_generations=params.max_generations,
        min_fitness=params.min_fitness,
        min_val_loss=params.min_val_loss,
        callbacks=cbks,
        parallel=params.parallel,
    )

    # Return results as dataframe
    return pd.merge(results, test_df, on=["generation", "chromosome"])


if __name__ == "__main__":
    # Run experiment
    params = HyperParameters(
        seed=42,
        data_dir="data",
        results_dir=f"/scratch/users/k1502897/nets/results/{EXPERIMENT_NAME}",
        val_size=5_000,
        batch_size=60,
        pop_size=25,
        elitism=5,
        p_crossover=0.5,
        mr_weight_noise=0.1,
        mr_weight_rand=0.1,
        mr_weight_zero=0.2,
        mr_weight_stddev=0.1,
        max_generations=100,
        min_fitness=0,
        min_val_loss=0,
        parallel=True,
    )
    results = mnist_experiment(params)
    results.to_csv(os.path.join(params.results_dir, "results.csv"), index=False)
