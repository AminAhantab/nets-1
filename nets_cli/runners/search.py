import torch

from nets_cli.runners.methods import search

from ..args import SearchArgs
from ..io import write_df, write_model


def run_search(args: SearchArgs, device: torch.device) -> None:
    # Get model hyperparameters
    arch = args.arch
    dataset = args.dataset
    initial_density = args.initial_density
    target_density = args.target_density

    # Get gradient descent hyperparameters
    optimiser = args.optimiser
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    val_size = args.val_size

    # Genetic algorithm hyperparameters
    pop_size = args.population_size
    mr_noise = args.mr_noise
    mr_noise_scale = args.mr_noise_scale
    mr_random = args.mr_random
    mr_disable = args.mr_disable
    p_crossover = args.p_crossover
    elitism = args.num_elites
    max_generations = args.max_generations
    min_fitness = args.min_fitness

    model, results = search(
        arch,
        dataset,
        val_size,
        optimiser,
        learning_rate,
        batch_size,
        pop_size,
        initial_density,
        target_density,
        elitism,
        p_crossover,
        mr_noise,
        mr_random,
        mr_disable,
        mr_noise_scale,
        max_generations,
        min_fitness,
        device,
    )

    # Return results as dataframe
    write_model(model, args.out_path, "model.pt", overwrite=True)
    write_df(results, args.csv_path, "results.csv")
