import logging

import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from .masked import MaskedNetwork

logger = logging.getLogger("nets.train")


def train_model(
    model: MaskedNetwork,
    data: DataLoader,
    opt: Optimizer,
    epochs: int = None,
    iterations: int = None,
    debug_every: int = 100,
    device: torch.device = None,
) -> float:
    """
    Train the model for a given number of epochs or iterations.

    Args:
        model: The model to train.
        data: The data to train on.
        opt: The optimizer to use.
        epochs: The number of epochs to train for.
        iterations: The number of iterations to train for.

    Returns:
        The average loss per epoch.
    """
    if epochs is None and iterations is None:
        epochs = 1

    if iterations is None:
        iterations = len(data) * epochs

    # Set the model to training mode
    model.train()

    # Train the model
    current_iteration = 0
    epoch = 1
    logger.info(f"Beginning training loop for {epochs} epochs.")
    logger.debug(f"Training for {iterations} iterations.")
    while current_iteration < iterations:
        for X, y in data:
            # Move data to device
            if device is not None:
                X = X.to(device)
                y = y.to(device)

            # Update the current iteration
            current_iteration += 1

            # Forward and backward pass
            logits = model(X)
            logger.debug(
                f"Iteration {current_iteration}: X={X.shape}, y={y.shape} logits={logits.shape}"
            )
            loss = model.loss(logits, y)
            loss.backward()
            opt.step()
            opt.zero_grad()

            # Log the loss
            if current_iteration % debug_every == 0:
                logger.debug(f"Iteration {current_iteration}: loss={loss.item():.4f}")

            # Check if we've reached the maximum number of iterations
            if current_iteration >= iterations:
                break

        logger.info(f"Epoch {epoch}/{epochs} complete: loss={loss.item():.4f}")
        epoch += 1

    return loss.item()


def evaluate_model(model: MaskedNetwork, data: DataLoader) -> tuple[float, float]:
    """
    Evaluate the model on the given data.

    Args:
        model: The model to evaluate.
        data: The data to evaluate on.

    Returns:
        The average loss and accuracy.
    """
    with torch.no_grad():
        model.eval()
        losses = torch.empty(len(data))
        accuracies = torch.empty(len(data))
        for i, (X, y) in enumerate(data):
            logits = model(X.unsqueeze(0))
            losses[i] = model.loss(logits, torch.tensor(y).unsqueeze(0))
            accuracies[i] = model.accuracy(logits, torch.tensor(y).unsqueeze(0))

    return losses.mean().item(), accuracies.mean().item()
