import logging
from typing import Callable, Dict

import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from .masked import MaskedNetwork

logger = logging.getLogger("nets.train")


def train_model(
    model: MaskedNetwork,
    data: DataLoader,
    opt: Optimizer,
    device: torch.device = None,
    callbacks: Dict[str, Callable] = None,
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
    # Set the model to training mode
    model.train()

    # Train the model
    current_iteration = 0
    stopping_triggered = False
    while not stopping_triggered:
        epoch = current_iteration // len(data) + 1
        logger.info(f"Epoch {epoch}...")

        for X, y in data:
            # Move data to device
            if device is not None:
                X = X.to(device)
                y = y.to(device)

            # Update the current iteration
            current_iteration += 1

            # Forward and backward pass
            logits = model(X)
            loss = model.loss(logits, y)
            loss.backward()
            opt.step()
            opt.zero_grad()

            _iteration_callbacks(model, callbacks, current_iteration, epoch, loss.item())

            stopping_triggered = _check_early_stopping(model, callbacks, current_iteration, epoch, loss.item())
            if stopping_triggered:
                logger.info("Early stopping triggered.")
                break

        _epoch_callbacks(model, callbacks, current_iteration, epoch, loss.item())

    return loss.item()


def evaluate_model(model: MaskedNetwork, data: DataLoader, device: torch.device = None) -> tuple[float, float]:
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
            # Move data to device
            if device is not None:
                X = X.to(device)
                y = y.to(device)
                
            logits = model(X)
            losses[i] = model.loss(logits, y)
            accuracies[i] = model.accuracy(logits, y)

    return losses.mean().item(), accuracies.mean().item()


def _check_early_stopping(model: MaskedNetwork, callbacks: Dict[str, Callable], iteration: int, epoch: int, loss: float) -> bool:
    if callbacks is None:
        return False
    
    if "early_stopping" in callbacks:
        for callback in callbacks["early_stopping"]:
            if callback(model, iteration, epoch, loss):
                return True
    
    return False

def _iteration_callbacks(model: MaskedNetwork, callbacks: Dict[str, Callable], iteration: int, epoch: int, loss: float):
    if callbacks is None:
        return

    if "iteration" in callbacks:
        for callback in callbacks["iteration"]:
            callback(model, iteration, epoch, loss)

def _epoch_callbacks(model: MaskedNetwork, callbacks: Dict[str, Callable], iteration: int, epoch: int, loss: float):
    if callbacks is None:
        return

    if "epoch" in callbacks:
        for callback in callbacks["epoch"]:
            callback(model, iteration, epoch, loss)
    