import logging
import argparse
import os
from nets import genetic

from nets_cli.runners import methods
from nets_cli.config import configure_seed, configure_torch

logger = logging.getLogger("nets_experiments")

VALIDATION_SIZE = 5_000
BATCH_SIZE = 64

INITIAL_DENSITY = 1.0
TARGET_DENSITY = 0.1
ELITISM = 2
P_CROSSOVER = 0.5
MR_NOISE_SCALE = 0.1
MIN_FITNESS = 0.0

MAX_EPOCHS = None
LOG_EVERY = 500
LOG_VAL_EVERY = 500
LOG_TEST_EVERY = 500


def run(args: argparse.Namespace):
    logging.basicConfig(level=logging.DEBUG)
    device = configure_torch(args.cpu_only)

    trial = args.trial
    seed = args.seed
    out_dir = args.out_dir
    arch = args.arch
    dataset = args.dataset
    fitness = args.fitness
    optimiser = args.optimiser
    lr = args.lr
    max_iter = args.max_iter
    max_gen = args.max_gen
    mr_noise = args.mr_noise
    mr_disable = args.mr_disable
    mr_enable = args.mr_enable
    mr_random = args.mr_random
    pop_size = args.pop_size

    configure_seed(seed)

    import torch

    if fitness == "1epoch":
        fitness_path = "sgd"
    elif fitness == "fwpass":
        fitness_path = "fwp"
    elif fitness == "fwpass_train":
        fitness_path = "fwp_sgd"
    else:
        raise ValueError(f"Unknown fitness function {fitness}")

    output_path = os.path.join(out_dir, arch, dataset, optimiser, fitness_path)

    def mkpath(*args):
        return os.path.join(output_path, *args)

    # Make output directories
    os.makedirs(output_path, exist_ok=True)

    INIT_SUFFIX = "init.pt"
    TRAIN_SUFFIX = "train.csv"
    TRAINED_SUFFIX = "trained.pt"
    SEARCH_SUFFIX = "search.csv"

    # Overparametrised initialisation =============================================================
    # FILE_PREFIX = "overp"

    # # Initialise a random model
    # overp_init = methods.init(
    #     architecture=arch,
    #     dataset=dataset,
    #     density=INITIAL_DENSITY,
    #     bias=False,
    # )

    # # Save the initialisation
    # overp_init_path = mkpath(f"{trial}_{FILE_PREFIX}_{INIT_SUFFIX}")
    # torch.save(overp_init, overp_init_path)

    # # Train the random model
    # overp_init, overp_results = methods.train(
    #     model=overp_init,
    #     dataset=dataset,
    #     val_size=VALIDATION_SIZE,
    #     optimiser=optimiser,
    #     learning_rate=lr,
    #     batch_size=BATCH_SIZE,
    #     max_iterations=max_iter,
    #     max_epochs=MAX_EPOCHS,
    #     log_every=LOG_EVERY,
    #     log_val_every=LOG_VAL_EVERY,
    #     log_test_every=LOG_TEST_EVERY,
    #     device=device,
    # )

    # # Save results
    # torch.save(overp_init, mkpath(f"{trial}_{FILE_PREFIX}_{TRAINED_SUFFIX}"))
    # overp_results.to_csv(mkpath(f"{trial}_{FILE_PREFIX}_{TRAIN_SUFFIX}"))

    # NeTS initialisation =========================================================================
    FILE_PREFIX = "nets"

    # fitness functio
    if fitness == "fwpass":
        fitness_fn = genetic.nets_fitness
    elif fitness == "1epoch":
        fitness_fn = genetic.nets_fitness_1epoch
    elif fitness == "fwpass_train":
        fitness_fn = genetic.nets_fitness_train
    else:
        raise ValueError(f"Unknown fitness function: {fitness}")

    # Search for an initialisation using NeTS
    nets_init, nets_search_results = methods.search(
        arch=arch,
        dataset=dataset,
        val_size=VALIDATION_SIZE,
        optimiser=optimiser,
        learning_rate=lr,
        batch_size=BATCH_SIZE,
        pop_size=pop_size,
        initial_density=INITIAL_DENSITY,
        target_density=TARGET_DENSITY,
        elitism=ELITISM,
        p_crossover=P_CROSSOVER,
        mr_noise=mr_noise,
        mr_random=mr_random,
        mr_disable=mr_disable,
        mr_enable=mr_enable,
        max_no_change=None,
        mr_noise_scale=MR_NOISE_SCALE,
        max_generations=max_gen,
        min_fitness=MIN_FITNESS,
        fitness_fn=fitness_fn,
        device=device,
    )

    # Save results
    nets_search_results.to_csv(mkpath(f"{trial}_{FILE_PREFIX}_{SEARCH_SUFFIX}"))
    torch.save(nets_init, mkpath(f"{trial}_{FILE_PREFIX}_{INIT_SUFFIX}"))

    # Train the best initialisation
    nets_init, nets_train_results = methods.train(
        model=nets_init,
        dataset=dataset,
        val_size=VALIDATION_SIZE,
        optimiser=optimiser,
        learning_rate=lr,
        batch_size=BATCH_SIZE,
        max_iterations=max_iter,
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
        architecture=arch,
        dataset=dataset,
        density=nets_init.density(),
        bias=False,
    )

    # Save the initialisation
    torch.save(sparse_init, mkpath(f"{trial}_{FILE_PREFIX}_{INIT_SUFFIX}"))

    # Train the random sparse model
    sparse_init, sparse_results = methods.train(
        model=sparse_init,
        dataset=dataset,
        val_size=VALIDATION_SIZE,
        optimiser=optimiser,
        learning_rate=lr,
        batch_size=BATCH_SIZE,
        max_iterations=max_iter,
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
    parser.add_argument("--seed", type=int, default=235234)
    parser.add_argument("--arch", type=str, default="lenet")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--fitness", type=str, default="fwpass")
    parser.add_argument("--optimiser", type=str, default="sgd")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--max_iter", type=int, default=10_000)
    parser.add_argument("--cpu_only", action="store_true")
    parser.add_argument("--pop_size", type=int, default=5)
    parser.add_argument("--max_gen", type=int, default=15)
    parser.add_argument("--mr_random", type=float, default=0.1)
    parser.add_argument("--mr_noise", type=float, default=0.1)
    parser.add_argument("--mr_disable", type=float, default=0.2)
    parser.add_argument("--mr_enable", type=float, default=0.0)
    args = parser.parse_args()
    run(args)
