import logging

import pandas as pd


logger = logging.getLogger("nets_cli.train")


def log_train_loss(df: pd.DataFrame, every: int = 100) -> None:
    assert every is None or every > 0

    def _cb(model, iteration: int, epoch: int, loss: float):
        if every is not None and iteration % every != 0:
            return

        logger.debug(f"Iteration {iteration} (Epoch {epoch}) — train loss: {loss:.4f}")
        df.loc[iteration, "train_loss"] = loss
        df.loc[iteration, "epoch"] = epoch

    return _cb


def log_val_loss(df: pd.DataFrame, val_loader, every: int = None, device=None) -> None:
    from torch.utils.data import DataLoader
    from nets import MaskedNetwork
    from nets.nn import evaluate_model

    assert isinstance(val_loader, DataLoader)
    assert every is None or every >= 0
    if every == 0:
        every = None

    def _cb(model: MaskedNetwork, iteration: int, epoch: int, _loss: float):
        if every is None or iteration % every != 0:
            return

        val_loss, val_acc = evaluate_model(model, val_loader, device=device)
        logger.info(
            f"Iteration {iteration} (Epoch {epoch}) — val loss: {val_loss:.4f}, val accuracy: {val_acc:.4f}"
        )
        df.loc[iteration, "val_loss"] = val_loss
        df.loc[iteration, "val_acc"] = val_acc
        df.loc[iteration, "epoch"] = epoch

    return _cb


def log_test_loss(
    df: pd.DataFrame, test_loader, every: int = None, device=None
) -> None:
    from torch.utils.data import DataLoader
    from nets import MaskedNetwork
    from nets.nn import evaluate_model

    assert isinstance(test_loader, DataLoader)
    assert every is None or every >= 0
    if every == 0:
        every = None

    def _cb(model: MaskedNetwork, iteration: int, epoch: int, _loss: float):
        if every is None or iteration % every != 0:
            return

        test_loss, test_acc = evaluate_model(model, test_loader, device=device)
        logger.info(
            f"Iteration {iteration} (Epoch {epoch}) — test loss: {test_loss:.4f}, test accuracy: {test_acc:.4f}"
        )
        df.loc[iteration, "test_loss"] = test_loss
        df.loc[iteration, "test_acc"] = test_acc
        df.loc[iteration, "epoch"] = epoch

    return _cb


def log_gpu_memory(every: int = None):
    import torch
    import pynvml

    pynvml.nvmlInit()

    def _cb(model, iteration: int, epoch: int, loss: float):
        if every is not None and iteration % every != 0:
            return
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_memory = info.total / 1024**3
        used_memory = info.used / 1024**3
        percent_used = used_memory / total_memory * 100
        free_memory = total_memory - used_memory
        logger.debug(
            f"GPU memory: {used_memory:.4f}/{total_memory:.4f} GB ({percent_used:.2f}%, {free_memory:.4f} GB free)"
        )

        temperature = pynvml.nvmlDeviceGetTemperature(handle, 0)
        utilisations = pynvml.nvmlDeviceGetUtilizationRates(handle)
        logger.debug(
            f"GPU temperature: {temperature} C, GPU utilisation: {utilisations.gpu}%, Memory utilisation: {utilisations.memory}%"
        )

    return _cb


def max_epochs(df: pd.DataFrame, max_epochs: int):
    def _cb(model, iteration: int, epoch: int):
        if max_epochs is None:
            return False

        if epoch > max_epochs:
            return True

    return _cb


def max_iterations(df: pd.DataFrame, max_iterations: int):
    def _cb(model, iteration: int, epoch: int):
        if max_iterations is None:
            return False

        if iteration > max_iterations:
            return True

    return _cb


def max_seconds(df: pd.DataFrame, max_seconds: int):
    import time

    start_time = time.time()

    def _cb(model, iteration: int, epoch: int):
        if max_seconds is None:
            return False

        if time.time() - start_time >= max_seconds:
            return True

    return _cb


def min_val_loss(df: pd.DataFrame, min_loss: float, patience: int = 0):
    def _cb(model, iteration: int, epoch: int):
        if min_loss is None:
            return False

        if not hasattr(_cb, "counter"):
            _cb.counter = 0

        if "val_loss" not in df.columns:
            return False

        if df.loc[iteration - 1, "val_loss"] <= min_loss:
            _cb.counter += 1

        if _cb.counter > patience:
            return True

    return _cb
