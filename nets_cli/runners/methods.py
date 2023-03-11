import logging
from typing import Callable, Dict, List, Tuple

import pandas as pd

import torch
from torch.utils.data import DataLoader

from nets import callbacks as cb
from nets.nets import neuroevolution_ts
from nets.nn import MaskedNetwork, train_model, evaluate_model

from ..hydrator import (
    hydrate_dataset,
    hydrate_new_model,
    hydrate_optimiser,
    hydrate_prune_method,
)

logger = logging.getLogger("nets_cli.methods")


def init(architecture: str, dataset: str, density: float, bias: bool) -> MaskedNetwork:
    # Initialise model
    model = hydrate_new_model(architecture, dataset, density, bias)

    # Log model creation
    logger.info(
        "Initialised a %s (density %.0f%%) for learning %s classifications.",
        architecture,
        model.density() * 100,
        dataset,
    )
    logger.debug("Initialised model: %s", model)

    return model


def prune(
    model: MaskedNetwork,
    criterion: str,
    threshold: float,
    count: int,
    fraction: float,
) -> MaskedNetwork:
    # Prune model
    logger.info("Initial density: {:.2f}%".format(100 * model.density()))
    prune_method = hydrate_prune_method(criterion, threshold, count, fraction)
    removed_weights = prune_method(model)
    logger.info("Removed {} weights".format(removed_weights))
    logger.info("Final density: {:.2f}%".format(100 * model.density()))
    return model


def train(
    model: MaskedNetwork,
    dataset: str,
    val_size: int,
    optimiser: str,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    max_iterations: int,
    log_every: int,
    log_val_every: int,
    log_test_every: int,
    device: torch.device,
) -> Tuple[MaskedNetwork, pd.DataFrame]:
    model.to(device)

    # Initialise data
    train_data, val_data, test_data = hydrate_dataset(dataset, val_size=val_size)

    # Initialise optimiser
    optimiser_class = hydrate_optimiser(optimiser)
    opt = optimiser_class(model.parameters(), lr=learning_rate)

    # Initialise the data loaders
    kwargs = {"pin_memory": True, "num_workers": 0}
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_data, batch_size=batch_size, **kwargs)
    test_loader = DataLoader(test_data, batch_size=batch_size, **kwargs)

    # Initialise the callbacks
    results = pd.DataFrame()
    callbacks = _init_train_callbacks(
        results,
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        max_epochs=max_epochs,
        max_iterations=max_iterations,
        log_every=log_every,
        log_val_every=log_val_every,
        log_test_every=log_test_every,
        device=device,
    )

    # Train model
    train_model(
        model,
        train_loader,
        opt,
        device=device,
        callbacks=callbacks,
    )

    return model, results


def iterative_magnitude_prune(
    model: MaskedNetwork,
    dataset: str,
    val_size: int,
    optimiser: str,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    max_iterations: int,
    log_every: int,
    log_val_every: int,
    log_test_every: int,
    cycles: int,
    criterion: str,
    threshold: float,
    count: int,
    fraction: float,
    reinit: bool,
    device: torch.device,
):
    # Store intiial weights
    initial_weights = []
    for layer in model.layers:
        initial_weights.append(layer.state_dict["weight"].clone())

    # Move model to device
    model.to(device)

    combined_results = pd.DataFrame()
    for cycle in range(cycles + 1):
        if cycle == cycles:
            logger.info("Training final model.")
        else:
            logger.info(f"Starting cycle {cycle} of {cycles}")

        model, train_results = train(
            model,
            dataset,
            val_size,
            optimiser,
            learning_rate,
            batch_size,
            max_epochs,
            max_iterations,
            log_every,
            log_val_every,
            log_test_every,
            device,
        )

        train_results["cycle"] = cycle
        train_results["density"] = model.density()

        if cycle == cycles:
            break

        # Prune model
        logger.info(f"Pruning model (cycle {cycle} of {cycles})")
        prune(model, criterion, threshold, count, fraction)

        # Reinitialise weights
        if reinit:
            _reinitialise_weights(model, initial_weights, device)

        # Add results to combined dataframe
        combined_results = pd.concat([combined_results, train_results])

    # Add final results to combined dataframe
    combined_results = pd.concat([combined_results, train_results])

    return model, combined_results


def search(
    arch: str,
    dataset: str,
    val_size: int,
    optimiser: str,
    learning_rate: float,
    batch_size: int,
    pop_size: int,
    initial_density: float,
    target_density: float,
    elitism: float,
    p_crossover: float,
    mr_noise: float,
    mr_random: float,
    mr_disable: float,
    mr_enable: float,
    mr_noise_scale: float,
    max_generations: int,
    max_no_change: int,
    min_fitness: float,
    device: torch.device,
):
    # Initialise data
    train_data, val_data, _ = hydrate_dataset(dataset, val_size=val_size)

    # Create data loaders
    # test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Create nets callbacks
    columns = ["generation", "chromosome", "test_loss", "test_acc"]
    test_df = pd.DataFrame(columns=columns)
    # test_cb = cb.nets_log_test_loss(test_df, test_loader, every=1, device=device)
    callbacks = []

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
        shuffle=False,
        p_crossover=p_crossover,
        mr_weight_noise=mr_noise,
        mr_weight_rand=mr_random,
        mr_weight_zero=mr_disable,
        mr_weight_enable=mr_enable,
        mr_weight_stddev=mr_noise_scale,
        max_generations=max_generations,
        max_no_change=max_no_change,
        min_fitness=min_fitness,
        callbacks=callbacks,
        device=device,
    )

    merged_df = pd.merge(results, test_df, on=["generation", "chromosome"])
    return model, merged_df


def _reinitialise_weights(
    model: MaskedNetwork, initial_weights: List[torch.Tensor], device: torch.device
):
    for layer, initial_weight in zip(model.layers, initial_weights):
        layer.reset(initial_weight.clone().to(device))


def _init_train_callbacks(
    df: pd.DataFrame,
    model: MaskedNetwork,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    max_epochs: int,
    max_iterations: int,
    log_every: int,
    log_val_every: int,
    log_test_every: int,
    device: torch.device,
) -> Dict[str, List[Callable]]:

    # Add initial stats to dataframe
    _add_initial_stats(df, model, train_loader, val_loader, test_loader, device=device)

    iteration_callbacks = [
        cb.log_train_loss(df, every=log_every),
        cb.log_val_loss(df, val_loader, every=log_val_every, device=device),
        cb.log_test_loss(df, test_loader, every=log_test_every, device=device),
    ]

    epoch_callbacks = [
        cb.log_val_loss(df, val_loader, device=device, every=1),
        cb.log_test_loss(df, test_loader, device=device, every=1),
    ]

    early_stopping_criteria = [
        cb.max_epochs(max_epochs),
        cb.max_iterations(max_iterations),
    ]

    if device == "cuda":
        epoch_callbacks.append(cb.log_gpu_memory())

    callbacks = {
        "iteration": iteration_callbacks,
        "epoch": epoch_callbacks,
        "early_stopping": early_stopping_criteria,
    }

    return callbacks


def _add_initial_stats(
    df: pd.DataFrame,
    model,
    train_loader,
    val_loader,
    test_loader,
    device=None,
):
    logger.info("Evaluating initial model")
    train_loss, train_acc = evaluate_model(model, train_loader, device=device)
    val_loss, val_acc = evaluate_model(model, val_loader, device=device)
    test_loss, test_acc = evaluate_model(model, test_loader, device=device)
    logger.debug("Train loss: %f, Train accuracy: %f", train_loss, train_acc)
    logger.debug("Val loss: %f, Val accuracy: %f", val_loss, val_acc)
    logger.debug("Test loss: %f, Test accuracy: %f", test_loss, test_acc)

    df.loc[0, "train_loss"] = train_loss
    df.loc[0, "train_acc"] = train_acc
    df.loc[0, "val_loss"] = val_loss
    df.loc[0, "val_acc"] = val_acc
    df.loc[0, "test_loss"] = test_loss
    df.loc[0, "test_acc"] = test_acc
    df.loc[0, "epoch"] = 0

    logger.info("Initial val loss: %f", val_loss)
    logger.info("Initial val accuracy: %f", val_acc)
    logger.info("Initial test loss: %f", test_loss)
    logger.info("Initial test accuracy: %f", test_acc)
