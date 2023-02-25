from typing import Tuple, Union
from torch import Generator

from torch.utils.data import random_split
from torchvision import datasets as torch_datasets, transforms
from torchvision.datasets import VisionDataset


TrainData = VisionDataset
ValData = VisionDataset
TestData = VisionDataset

TrainTestData = Tuple[TrainData, TestData]
TrainValTestData = Tuple[TrainData, ValData, TestData]

CIFAR10Data = Union[TrainTestData, TrainValTestData]


def load(
    root: str = ".",
    download: bool = False,
    val_size: int = None,
    generator: Generator = None,
) -> CIFAR10Data:
    """
    Load the CIFAR10 dataset.

    Args:
        root: Root directory of datasets or where to download them if download=True.
        download: If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        val_size: If not None, the size of the validation set. If None, no
            validation set is returned.

    Returns:
        A tuple of the train and test datasets. If ``val_size`` is not None,
        a third dataset is returned.
    """
    transform_fns = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    transform = transforms.Compose(transform_fns)

    kwargs = {"root": root, "download": download, "transform": transform}
    train = torch_datasets.CIFAR10(train=True, **kwargs)
    test = torch_datasets.CIFAR10(train=False, **kwargs)

    if val_size is not None and val_size > 0:
        assert val_size < len(train)
        train_len = len(train) - val_size
        train, val = random_split(train, [train_len, val_size], generator=generator)
        return train, val, test
    else:
        return train, test
