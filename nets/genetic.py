import math
import logging
import sys
from typing import Callable, Dict, Iterator

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader


from .nn.masked import MaskedNetwork, MaskedLayer
from .nn.train import train_model, evaluate_model
from .utils import uniform_mask

logger = logging.getLogger("nets")


WEIGHT_INDEX = 0
MASK_INDEX = 1

FitnessFn = Callable[[Tensor], Dict[str, Tensor]]


def init_population(num_params: int, pop_size: int, density: float = 1.0) -> Tensor:
    """
    Initialise a population of individuals with random weights.

    Args:
        num_params: The number of parameters in the model.
        pop_size: The size of the population.
        density: The density of the network.

    Returns:
        A tensor of shape (pop_size, 2, num_parameters) containing the
        initial population.
    """
    assert pop_size > 0
    logger.info("Initialising population of size %d...", pop_size)
    population = torch.empty((pop_size, 2, num_params), dtype=None)
    # HACK: This is just for LeNet to test if the initialisation is the issue...
    for i in range(pop_size):
        fc1 = torch.empty((300, 28 * 28))
        fc2 = torch.empty((100, 300))
        fc3 = torch.empty((10, 100))
        nn.init.kaiming_uniform_(fc1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(fc2, a=math.sqrt(5))
        nn.init.kaiming_uniform_(fc3, a=math.sqrt(5))
        population[i, 0, : 300 * 28 * 28] = fc1.reshape(-1)
        population[i, 0, 300 * 28 * 28 : 300 * 28 * 28 + 300 * 100] = fc2.reshape(-1)
        population[i, 0, 300 * 28 * 28 + 300 * 100 :] = fc3.reshape(-1)

    population[:, 1, :] = uniform_mask((num_params,), density)

    return population


def load_weights(
    model: MaskedNetwork,
    individual: Tensor,
    device: torch.device = None,
):
    """
    Load weights from a single individual into a model.

    Args:
        model: The model to load the weights into.
        individual: The individual to load the weights from.
        device: The device to move the weights to.

    Returns:
        None
    """
    logger.debug("Loading weights from individual into model")
    for layer in model.layers:
        assert isinstance(layer, MaskedLayer)
        num_weights = layer.num_parameters()

        weight_vals = (
            individual[0, :num_weights]
            .reshape(layer.weight.shape)
            .detach()
            .clone()
            .to(device)
        )
        mask_vals = (
            individual[1, :num_weights]
            .reshape(layer.mask.shape)
            .detach()
            .clone()
            .to(device)
        )

        layer.weight.data = weight_vals
        layer.mask.data = mask_vals
        layer.to(device)
        individual = individual[:, num_weights:]


def select_parents(fitness: Tensor, num_parents: int) -> Tensor:
    """
    Select parents for the next generation using roulette wheel selection.

    Args:
        fitness: A tensor of shape (pop_size,) containing the fitness values
            of each individual.
        num_parents: The number of parents to select.

    Returns:
        A tensor of shape (num_parents,) containing the indices of the selected
        parents.
    """
    # Create a tensor to store the parents
    parents = torch.zeros(num_parents, dtype=torch.long)

    # Normalize the fitness values to get the probability of selection for each individual
    fitness = (fitness - fitness.min()) / (fitness.max() - fitness.min() + 1e-6)
    fitness = fitness / fitness.sum()

    # Calculate the cumulative probability of selection
    cumulative_probs = torch.cumsum(fitness, dim=0)

    # Generate random numbers between 0 and 1
    random_numbers = torch.rand(num_parents)

    # Roulette wheel selection
    parents = torch.searchsorted(cumulative_probs, random_numbers)
    return parents


def nets_fitness(
    model: MaskedNetwork,
    train_loader: DataLoader,
    val_loader: DataLoader,
    init_opt: Callable[[Iterator[nn.Parameter]], torch.optim.Optimizer],
    target: float = 0.2,
    train_callbacks: Dict[str, list] = None,
    device: torch.device = None,
) -> FitnessFn:
    """
    Create a fitness function for the Nets algorithm.

    This is a higher-order function that returns a function that can be used to
    calculate the fitness of a population.
    It will use the given model, data and target density to calculate the
    losses, accuracies, and fitness.

    Args:
        model: The model to train.
        train_data: The data to train on.
        val_data: The data to validate on.
        target: The target density of the model.

    Returns:
        A function that can be used to calculate the fitness of a population.
    """

    def fitness(population: Tensor) -> Dict[str, Tensor]:
        fitnesses = torch.zeros_like(population[:, 0, 0])
        train_losses = torch.zeros_like(fitnesses)
        val_losses = torch.zeros_like(fitnesses)
        val_accs = torch.zeros_like(fitnesses)
        densities = torch.zeros_like(fitnesses)
        penalties = torch.zeros_like(fitnesses)

        from models.lenet import LeNetFeedForwardNetwork

        for i, individual in enumerate(population):
            logger.debug(f"Training chromosome {i}...")

            # Load the weights into the model
            load_weights(model, individual, device=device)

            # Train the model
            # opt = init_opt(model.parameters())
            # train_loss = train_model(model, train_loader, opt, device, train_callbacks)

            # Evaluate the mode
            val_loss, val_acc = evaluate_model(model, val_loader, device)
            logger.info("Validation loss: %.4f", val_loss)
            logger.info("Validation accuracy: %.4f", val_acc)
            density = model.density()
            penalty = ((density - target) / (1 - target)) ** 2
            logger.info("Density: %.4f", density)

            # Calculate the fitness
            fitnesses[i] = val_loss + penalty
            logger.info("Fitness: %.4f", fitnesses[i])
            # train_losses[i] = train_loss
            val_losses[i] = val_loss
            val_accs[i] = val_acc
            densities[i] = density
            penalties[i] = penalty

        results = {
            "fitness": fitnesses,
            "train_loss": train_losses,
            "val_loss": val_losses,
            "val_acc": val_accs,
            "density": densities,
            "penalty": penalties,
        }

        return results

    return fitness


def uniform_crossover(
    population: Tensor,
    parents: Tensor,
    p: float = 0.5,
) -> Tensor:
    """
    Perform uniform crossover on a population.

    Args:
        population: The population to perform crossover on.
        parents: The indices of the parents to use.
        num_offspring: The number of offspring to create.
        p: The probability of inheriting a gene from the first parent.

    Returns:
        A tensor of shape (num_offspring, genome_depth, genome_length)
        containing the offspring.
    """

    # Get the shape of the population
    _, genome_depth, genome_length = population.shape
    num_parents = parents.shape[0]

    # Create a tensor to store the children
    children_shape = (num_parents, genome_depth, genome_length)
    children = torch.zeros(
        children_shape,
        dtype=population.dtype,
        device=population.device,
    )

    # Select the parents
    parent1 = population[parents[torch.arange(0, num_parents) % num_parents]]
    parent2 = population[parents[(torch.arange(0, num_parents) + 1) % num_parents]]

    # Create a mask of random values between 0 and 1
    random_mask = torch.rand(children_shape) < p
    children = torch.where(random_mask, parent1, parent2)

    return children


def noise_mutation(individuals: Tensor, p: float, scale: float = 0.1):
    """
    Add noise to the weights of the individuals.

    Args:
        individuals: A tensor of shape (pop_size, genome_depth, genome_length)
            containing the individuals.
        p: The probability of mutating a weight.
        scale: The scale of the noise to add.

    Returns:
        None
    """
    # Extract the shape of the population
    num_individuals, _, genome_length = individuals.shape

    # Create a mask of random values between 0 and 1
    random_values = torch.rand(num_individuals, genome_length)
    application_mask = (random_values < p).float()

    # Create a tensor of random noise
    noise_to_apply = torch.empty(num_individuals, genome_length).normal_(0, scale)

    # Add the noise to the weights
    individuals[:, WEIGHT_INDEX, :] += application_mask * noise_to_apply * scale


def random_weight_mutation(individuals: Tensor, p: float):
    """
    Replace the weights of the individuals with random values.

    Args:
        individuals: A tensor of shape (pop_size, genome_depth, genome_length)
            containing the individuals.
        p: The probability of mutating a weight.

    Returns:
        None
    """
    # Extract the shape of the population
    num_individuals, _, num_weights = individuals.shape

    # Create a mask of random values between 0 and 1
    random_values = torch.rand(num_individuals, num_weights)
    application_mask = (random_values < p).float()

    # Zero out the weights that are being mutated
    individuals[:, WEIGHT_INDEX, :] *= 1 - application_mask

    # Add random noise to the weights that are being mutated
    rand = torch.empty(num_individuals, num_weights).normal_(0, 0.1)
    individuals[:, WEIGHT_INDEX, :] += application_mask * rand


def disable_weight_mutation(individuals: Tensor, p: float):
    """
    Disable the weights of the individuals.

    Args:
        individuals: A tensor of shape (pop_size, genome_depth, genome_length)
            containing the individuals.
        p: The probability of mutating a weight.

    Returns:
        None
    """
    # Extract the shape of the population
    num_children, _, num_weights = individuals.shape

    # Create a mask of random values between 0 and 1
    random_values = torch.rand(num_children, num_weights)
    application_mask = (random_values < p).float()

    # Zero out the masks that are being mutated
    individuals[:, MASK_INDEX, :] *= 1 - application_mask


def enable_weight_mutation(individuals: Tensor, p: float):
    """
    Enable the weights of the individuals.

    Args:
        individuals: A tensor of shape (pop_size, genome_depth, genome_length)
            containing the individuals.
        p: The probability of mutating a weight.

    Returns:
        None
    """
    # Extract the shape of the population
    num_children, _, num_weights = individuals.shape

    # Create a mask of random values between 0 and 1
    random_values = torch.rand(num_children, num_weights)
    application_mask = (random_values < p).float()

    # Change the masks that are being mutated to 1 (they may be 1 already)
    individuals[:, MASK_INDEX, :] += application_mask * (
        1 - individuals[:, MASK_INDEX, :]
    )


def next_generation(
    model: MaskedNetwork,
    population: Tensor,
    fitness: Tensor,
    children: Tensor,
    elitism: int,
) -> Tensor:
    """
    Create the next generation of the population.

    Args:
        model: The model to train.
        population: The current population.
        fitness: The fitness of the current population.
        children: The children of the current population.
        elitism: The number of elite individuals to keep.

    Returns:
        The next generation of the population.
    """
    # Find the elite individuals
    _, elite_idxs = torch.topk(fitness, k=elitism, largest=False)

    # Create the next generation
    pop_shape = (population.shape[0], 2, model.num_parameters())
    next_population = torch.zeros(pop_shape)

    # Add the elite individuals to the next generation
    next_population[:elitism, :, :] = population[elite_idxs, :, :]

    # Add the children to the next generation
    next_population[elitism:, :, :] = children

    # Return the next generation
    return next_population
