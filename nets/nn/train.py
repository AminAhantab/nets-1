import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from .masked import MaskedNetwork


def train_model(
    model: MaskedNetwork,
    data: DataLoader,
    opt: Optimizer,
    epochs: int = None,
    iterations: int = None,
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
    while current_iteration < iterations:
        for X, y in data:
            current_iteration += 1
            logits = model(X)
            loss = model.loss(logits, y)
            loss.backward()
            opt.step()
            opt.zero_grad()

            if current_iteration >= iterations:
                break

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
