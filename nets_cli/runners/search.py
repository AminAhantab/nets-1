import pandas as pd

from nets.nets import neuroevolution_ts
from nets_cli.io import write_df, write_model

from ..args import SearchArgs
from ..config import configure_logger, configure_seed, configure_torch
from ..hydrator import hydrate_dataset, hydrate_new_model, hydrate_optimiser


def run_search(args: SearchArgs) -> None:
    # Configure environment
    configure_logger(args)
    device = configure_torch(args.no_cuda)
    configure_seed(args.seed)

    from torch.utils.data import DataLoader
    import nets.callbacks as cb

    # Get relevant arguments
    arch = args.arch
    initial_density = args.initial_density
    target_density = args.target_density
    dataset = args.dataset
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

    # Initialise data
    train_data, val_data, test_data = hydrate_dataset(dataset, val_size=val_size)

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Create callbacks
    test_df = pd.DataFrame(
        columns=["generation", "chromosome", "test_loss", "test_acc"]
    )
    callbacks = [
        cb.nets_log_test_loss(test_df, test_loader, every=1, device=device),
    ]

    # Initialise the model
    model = hydrate_new_model(arch, dataset, bias=False)

    # Initialise the optimiser
    optimiser_class = hydrate_optimiser(optimiser)
    init_opt = lambda params: optimiser_class(params, lr=learning_rate)

    # Run neuroevolution
    results = neuroevolution_ts(
        model,
        train_data,
        val_data,
        init_opt=init_opt,
        batch_size=batch_size,
        pop_size=pop_size,
        init_density=initial_density,
        target_density=target_density,
        elitism=elitism,
        p_crossover=p_crossover,
        mr_weight_noise=mr_noise,
        mr_weight_rand=mr_random,
        mr_weight_zero=mr_disable,
        mr_weight_stddev=mr_noise_scale,
        max_generations=max_generations,
        min_fitness=min_fitness,
        callbacks=callbacks,
        device=device,
    )

    # Return results as dataframe
    merged_df = pd.merge(results, test_df, on=["generation", "chromosome"])
    write_model(model, args.out_path, "model.pt", overwrite=True)
    write_df(merged_df, args.csv_path, "results.csv")
