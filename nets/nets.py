from datetime import datetime
import logging
from typing import Any, Callable, Dict, List, Union

import pandas as pd

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from .nn import MaskedNetwork
from . import genetic

logger = logging.getLogger("evolth")

Callback = Callable[[MaskedNetwork, Tensor, Dict[str, Tensor], int], Any]


def neuroevolution_ts(
    model: MaskedNetwork,
    data: Dataset,
    val_data: Dataset,
    batch_size: int = 64,
    pop_size: int = 10,
    elitism: int = 3,
    p_crossover: float = 0.5,
    mr_weight_noise: float = 0.1,
    mr_weight_rand: float = 0.1,
    mr_weight_zero: float = 0,
    mr_weight_stddev: float = 0.1,
    max_generations: int = None,
    min_fitness: float = None,
    min_val_loss: float = None,
    callbacks: List[Callback] = None,
):
    """
    Run Neuroevolution Ticket Search (NeTS) to find a winning network
    initialisation.

    Args:
        model: The prototype model to use for gradient descent.
        data: The training data.
        val_data: The validation data.
        batch_size: The batch size to use for stochastic gradient descent.
        pop_size: The size of the population.
        elitism: The number of individuals to keep in the population.
        p_crossover: The probability of crossover.
        mr_weight_noise: The probability of mutating a weight by adding noise.
        mr_weight_rand: The probability of mutating a weight by setting it to a
                random value.
        mr_weight_zero: The probability of mutating a weight by setting it to
                zero.
        mr_weight_stddev: The standard deviation of the noise to add to weights.
        max_generations: The maximum number of generations to run for.
        min_fitness: The minimum fitness to terminate evolution.
        min_val_loss: The minimum validation loss to terminate evolution.
        callbacks: A list of callback functions to run after each fitness
                evaluation.

    Returns:
        Results dataframe.
    """
    # Check arguments
    assert pop_size >= 2
    assert elitism <= pop_size and elitism >= 0
    assert p_crossover >= 0 and p_crossover <= 1
    assert mr_weight_noise >= 0 and mr_weight_noise <= 1
    assert mr_weight_rand >= 0 and mr_weight_rand <= 1
    assert mr_weight_zero >= 0 and mr_weight_zero <= 1
    assert mr_weight_stddev >= 0
    assert max_generations >= 0

    # Initialise population
    population = genetic.init_population(model, pop_size)

    # Initialise data loaders
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=None)

    # Initialise fitness function
    fitness_fn = genetic.nets_fitness(model, train_loader, val_loader)

    # Initialise results
    solution = None
    results = init_results()

    # Run evolution loop
    for gen in range(max_generations + 1):
        logger.info("Beginning generation %d...", gen)

        # Calculate fitness and validation stats
        fitness_result = fitness_fn(population)
        fitness = fitness_result["fitness"]
        best_idx = torch.argmin(fitness)

        # Log best individual and other results
        results = append_results(results, gen, fitness_result)

        # Run callbacks
        if callbacks is not None:
            for callback in callbacks:
                callback(model, population, fitness_result, gen)

        # Check termination conditions
        conditions = (min_fitness, min_val_loss, max_generations)
        reason = should_terminate(gen, fitness_result, best_idx, *conditions)
        if reason is not None:
            logger.info("Terminating evolution: %s", reason)
            break

        # Update best solution
        if solution is None or fitness[best_idx] < solution[1]:
            logger.debug("New best solution found: fitness=%.4f", fitness[best_idx])
            solution_chromosome = population[best_idx, :, :].squeeze().clone()
            solution = (solution_chromosome, fitness[best_idx].item())

        # Evolve population to next generation
        population = evolve(
            model,
            population,
            fitness,
            elitism,
            p_crossover,
            mr_weight_noise,
            mr_weight_rand,
            mr_weight_zero,
            mr_weight_stddev,
        )

    # Load best solution into model before returning
    genetic.load_weights(model, solution[0], requires_grad=True)
    logger.info("Best individual: %s", solution[1])

    # Return results
    return results


def evolve(
    model: MaskedNetwork,
    population: Tensor,
    fitness: Tensor,
    elitism: int,
    p_crossover: float,
    mr_weight_noise: float,
    mr_weight_rand: float,
    mr_weight_zero: float,
    mr_weight_stddev: float,
) -> Tensor:
    """Evolve population to next generation."""
    # Select parents and perform crossover
    num_parents = population.shape[0] - elitism
    parents = genetic.select_parents(fitness, num_parents)
    children = genetic.uniform_crossover(population, parents, p_crossover)

    # Mutate children
    genetic.noise_mutation(children, mr_weight_noise, mr_weight_stddev)
    genetic.random_weight_mutation(children, mr_weight_rand)
    genetic.disable_weight_mutation(children, mr_weight_zero)

    # Update population to next generation
    return genetic.next_generation(model, population, fitness, children, elitism)


def should_terminate(
    gen: int,
    fitness_results: Dict[str, Tensor],
    best_idx: int,
    min_fitness: float,
    min_val_loss: float,
    max_generations: int,
) -> Union[str, None]:
    """Check if evolution should terminate."""

    # Extract fitness and validation losses
    fitness = fitness_results["fitness"]
    val_losses = fitness_results["val_loss"]

    # Check min fitness termination condition
    if min_fitness is not None and fitness.min() < min_fitness:
        return "Terminating early: minimum fitness reached."

    # Check min validation loss termination condition
    if min_val_loss is not None and val_losses[best_idx] < min_val_loss:
        return "Terminating early: minimum validation loss reached."

    # Check max generations termination condition
    if gen == max_generations:
        return "Terminating: maximum generations reached."

    return None


def init_results() -> pd.DataFrame:
    """Initialise results dataframe."""

    return pd.DataFrame(
        columns=[
            "generation",
            "chromosome",
            "density",
            "fitness",
            "penalty",
            "train_loss",
            "val_loss",
        ]
    )


def append_results(
    results: pd.DataFrame, gen: int, result_dict: Dict[str, Tensor]
) -> pd.DataFrame:
    """Process and append results to dataframe."""

    # Extract results
    fitness = result_dict["fitness"]
    train_losses = result_dict["train_loss"]
    val_losses = result_dict["val_loss"]
    val_accs = result_dict["val_acc"]
    densities = result_dict["density"]
    penalties = result_dict["penalty"]

    # Other attributes
    generation_col = [gen] * len(val_losses)
    chromosome_index = [i for i in range(len(val_losses))]
    record_timestamp = [datetime.now().timestamp()] * len(val_losses)

    data = {
        "generation": generation_col,
        "chromosome": chromosome_index,
        "fitness": fitness.detach(),
        "train_loss": train_losses.detach(),
        "val_loss": val_losses.detach(),
        "val_acc": val_accs.detach(),
        "density": densities.detach(),
        "penalty": penalties.detach(),
        "timestamp": record_timestamp,
    }

    return pd.concat([results, pd.DataFrame(data)], ignore_index=True)
