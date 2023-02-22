import logging

import pandas as pd


from ..args import TrainArgs
from ..config import configure_logger, configure_seed, configure_torch
from ..io import load_model, write_df, write_model
from ..mapper import hydrate_dataset, hydrate_optimiser

logger = logging.getLogger("nets_cli.train")


def run_train(args: TrainArgs) -> None:
    # Configure environment
    configure_logger(args)
    device = configure_torch(args.no_cuda)
    configure_seed(args.seed)

    import torch
    from torch.utils.data.dataset import Subset
    from torch.utils.data import DataLoader
    from nets.nn.train import train_model

    # Get relevant arguments
    model_path = args.model_path
    dataset = args.dataset
    optimiser = args.optimiser
    learning_rate = args.learning_rate
    max_epochs = args.max_epochs
    batch_size = args.batch_size
    val_size = args.val_size
    log_every = args.log_every
    log_val_every = args.log_val_every
    log_test_every = args.log_test_every

    # Load model from disk
    model = load_model(model_path)
    model.to(device)

    # Initialise data
    train_data, val_data, test_data = hydrate_dataset(dataset, val_size=val_size)

    # Initialise optimiser
    optimiser_class = hydrate_optimiser(optimiser)
    opt = optimiser_class(model.parameters(), lr=learning_rate)

    # Initialise the data loaders
    kwargs = {"pin_memory": True, "num_workers": 0}
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_data, batch_size=99_999, **kwargs)
    test_loader = DataLoader(test_data, batch_size=99_999, **kwargs)

    # Initialise the callbacks
    df = pd.DataFrame()
    # add_first_row(df, model, train_loader, val_loader, test_loader, device=device)
    callbacks = {
        "iteration": [
            log_train_loss(df, every=log_every),
            log_val_loss(df, val_loader, every=log_val_every, device=device),
            log_test_loss(df, test_loader, every=log_test_every, device=device),
        ],
        "epoch": [
            log_val_loss(df, val_loader, device=device, every=1),
            log_test_loss(df, test_loader, device=device, every=1),
        ],
    }

    if device == "cuda":
        callbacks["epoch"].append(log_gpu_memory())

    # Train model
    train_model(
        model,
        train_loader,
        opt,
        epochs=max_epochs,
        device=device,
        callbacks=callbacks,
    )

    # Save model and results
    write_model(model, args.out_path, file_name="trained.pt", overwrite=True)
    write_df(df, args.csv_path, file_name="results.csv", overwrite=True)


def log_train_loss(df: pd.DataFrame, every: int = 100) -> None:
    assert every is None or every > 0

    def _cb(model, iteration: int, epoch: int, loss: float):
        if every is not None and iteration % every != 0:
            return

        logger.debug(f"Train loss: {loss:.4f}")
        df.loc[iteration, "train_loss"] = loss
        df.loc[iteration, "epoch"] = epoch

    return _cb


def log_val_loss(df: pd.DataFrame, val_loader, every: int = None, device=None) -> None:
    from torch.utils.data import DataLoader
    from nets import MaskedNetwork
    from nets.nn import evaluate_model

    assert isinstance(val_loader, DataLoader)
    assert every is None or every > 0

    def _cb(model: MaskedNetwork, iteration: int, epoch: int, _loss: float):
        if every is None or iteration % every != 0:
            return

        val_loss, val_acc = evaluate_model(model, val_loader, device=device)
        logger.info(f"Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.4f}")
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
    assert every is None or every > 0

    def _cb(model: MaskedNetwork, iteration: int, epoch: int, _loss: float):
        if every is None or iteration % every != 0:
            return

        test_loss, test_acc = evaluate_model(model, test_loader, device=device)
        logger.info(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")
        df.loc[iteration, "test_loss"] = test_loss
        df.loc[iteration, "test_acc"] = test_acc
        df.loc[iteration, "epoch"] = epoch

    return _cb


def add_first_row(
    df: pd.DataFrame,
    model,
    train_loader,
    val_loader,
    test_loader,
    device=None,
):
    from nets.nn import evaluate_model

    logger.info("Evaluating initial model")
    # train_loss, train_acc = evaluate_model(model, train_loader, device=device)
    val_loss, val_acc = evaluate_model(model, val_loader, device=device)
    test_loss, test_acc = evaluate_model(model, test_loader, device=device)
    # logger.debug("Train loss: %f, Train accuracy: %f", train_loss, train_acc)
    logger.debug("Val loss: %f, Val accuracy: %f", val_loss, val_acc)
    logger.debug("Test loss: %f, Test accuracy: %f", test_loss, test_acc)

    df.loc[0, "train_loss"] = None
    df.loc[0, "val_loss"] = val_loss
    df.loc[0, "val_acc"] = val_acc
    df.loc[0, "test_loss"] = test_loss
    df.loc[0, "test_acc"] = test_acc
    df.loc[0, "epoch"] = 0

    logger.info("Initial val loss: %f", val_loss)
    logger.info("Initial val accuracy: %f", val_acc)
    logger.info("Initial test loss: %f", test_loss)
    logger.info("Initial test accuracy: %f", test_acc)


def log_gpu_memory():
    import torch
    import pynvml

    pynvml.nvmlInit()

    def _cb(model, iteration: int, epoch: int, loss: float):
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        logger.debug(f"GPU memory: {info.used / 1e9:.4f} GB")

    return _cb
