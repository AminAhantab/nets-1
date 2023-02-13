import logging
from typing import Dict

import pandas as pd

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.multiprocessing import Manager, Process

from . import genetic
from .nn import MaskedNetwork, train_model, evaluate_model

logger = logging.getLogger("nets")


def test_callback(
    df: pd.DataFrame,
    train_data: Dataset,
    test_data: Dataset,
    epochs: int = 1,
    batch_size: int = 64,
) -> Tensor:
    """Create a callback function for testing a population of networks."""

    def callback(
        model: MaskedNetwork,
        population: Tensor,
        fitness_results: Dict[str, Tensor],
        generation: int,
    ) -> Tensor:
        """Evaluate the fitness of each individual in the population."""
        # Initialise data loaders
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=None)

        # Determine population size
        pop_size = population.shape[0]
        logger.info(
            "Testing population of %d individuals after %d epochs of SGD.",
            pop_size,
            epochs,
        )

        # Evaluate each individual in the population
        test_losses = torch.zeros(pop_size)
        test_accs = torch.zeros(pop_size)
        for i in range(pop_size):
            logger.debug(f"Testing individual {i}.")

            # Load weights into model
            genetic.load_weights(model, population[i], requires_grad=True)

            # Train model on training set
            logger.debug("Training network on training set for %d epochs.", epochs)
            opt = torch.optim.SGD(model.parameters(), lr=1e-3)
            train_model(model, train_loader, opt, epochs=epochs)

            # Evaluate model on test set
            test_losses[i], test_accs[i] = evaluate_model(model, test_loader)

            # Save results to dataframe
            df.loc[len(df)] = [
                generation,
                i,
                test_losses[i].item(),
                test_accs[i].item(),
            ]

        fitness_results["test_loss"] = test_losses
        fitness_results["test_acc"] = test_accs

        # Log and return results
        logger.debug("Finished testing: mean test loss: %.4f", test_losses.mean())

    return callback


def test_callback_parallel(
    df: pd.DataFrame,
    train_data: Dataset,
    test_data: Dataset,
    epochs: int = 1,
    batch_size: int = 64,
) -> Tensor:
    """Create a callback function for testing a population of networks."""

    def callback(
        model: MaskedNetwork,
        population: Tensor,
        fitness_results: Dict[str, Tensor],
        generation: int,
    ) -> Tensor:
        """Evaluate the fitness of each individual in the population."""
        # Initialise data loaders
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=None)

        # Determine population size
        pop_size = population.shape[0]
        logger.info(
            "Testing population of %d individuals after %d epochs of SGD.",
            pop_size,
            epochs,
        )

        # Store results in shared memory
        manager = Manager()
        test_losses = manager.list([0.0] * pop_size)
        test_accs = manager.list([0.0] * pop_size)

        # Individual evaluation function
        def calc_test_loss(i):
            logger.debug(f"Testing individual {i}.")

            # Load weights into model
            genetic.load_weights(model, population[i], requires_grad=True)

            # Train model on training set
            logger.debug("Training network on training set for %d epochs.", epochs)
            opt = torch.optim.SGD(model.parameters(), lr=1e-3)
            train_model(model, train_loader, opt, epochs=epochs)

            # Evaluate model on test set
            test_losses[i], test_accs[i] = evaluate_model(model, test_loader)

        # Evaluate each individual in the population
        proccesses = []
        for i in range(pop_size):
            p = Process(target=calc_test_loss, args=(i,))
            p.start()
            proccesses.append(p)

        # Wait for all processes to finish
        for p in proccesses:
            p.join()

        # Convert to tensors
        test_losses = torch.tensor(test_losses)
        test_accs = torch.tensor(test_accs)

        # Save results to dataframe
        for i in range(pop_size):
            df.loc[len(df)] = [
                generation,
                i,
                test_losses[i],
                test_accs[i],
            ]

        fitness_results["test_loss"] = test_losses
        fitness_results["test_acc"] = test_accs

        # Log and return results
        logger.debug("Finished testing: mean test loss: %.4f", test_losses.mean())

    return callback


def init_test_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["generation", "chromosome", "test_loss", "test_acc"])


def log_callback():
    def log_results(
        model: MaskedNetwork,
        population: Tensor,
        fitness_results: Dict[str, Tensor],
        generation: int,
    ):
        """Log results of the current generation."""

        # Extract fitness results
        fitness = fitness_results["fitness"]
        val_losses = fitness_results["val_loss"]
        val_accs = fitness_results["val_acc"]
        penalties = fitness_results["penalty"]
        densities = fitness_results["density"]

        # Best individual
        best_idx = fitness.argmax().item()
        best_fitness = fitness[best_idx].item()
        best_val_loss = val_losses[best_idx].item()
        best_val_acc = val_accs[best_idx].item()
        best_penalty = penalties[best_idx].item()
        best_density = densities[best_idx].item()

        # Population average
        mean_fitness = fitness.mean().item()
        mean_val_loss = val_losses.mean().item()
        mean_val_acc = val_accs.mean().item()
        mean_penalty = penalties.mean().item()
        mean_density = densities.mean().item()

        # Extreme values
        min_val_loss = val_losses.min().item()
        max_val_acc = val_accs.max().item()
        min_penalty = penalties.min().item()
        min_density = densities.min().item()

        # Construct log string
        summary = (
            f"Best {best_fitness:.4f}: VL={best_val_loss:.4f}, "
            + f"VA={best_val_acc:.4f}, "
            + f"DEN={best_density:.4f} ({best_penalty:.4f})\n"
        )

        summary += (
            f"Mean {mean_fitness:.4f}: VL={mean_val_loss:.4f}, "
            + f"VA={mean_val_acc:.4f}, "
            + f"DEN={mean_density:.4f} ({mean_penalty:.4f})\n"
        )

        summary += (
            f"Extrema:     VL={min_val_loss:.4f}, VA={max_val_acc:.4f}, "
            + f"DEN={min_density:.4f} ({min_penalty:.4f})"
        )

        logger.info(f"Generation {generation} complete:\n{summary}")

    return log_results
