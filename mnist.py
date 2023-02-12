from typing import Tuple, Union

from torch.utils.data import random_split
from torchvision import datasets as torch_datasets, transforms
from torchvision.datasets import VisionDataset


TrainData = VisionDataset
ValData = VisionDataset
TestData = VisionDataset

TrainTestData = Tuple[TrainData, TestData]
TrainValTestData = Tuple[TrainData, ValData, TestData]

MNISTData = Union[TrainTestData, TrainValTestData]


def load(root: str = ".", download: bool = False, val_size: int = None) -> MNISTData:
    """
    Load the MNIST dataset.

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
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
    transform = transforms.Compose(transform_fns)

    kwargs = {"root": root, "download": download, "transform": transform}
    train = torch_datasets.MNIST(train=True, **kwargs)
    test = torch_datasets.MNIST(train=False, **kwargs)

    if val_size is not None and val_size > 0:
        assert val_size < len(train)
        train_len = len(train) - val_size
        train, val = random_split(train, [train_len, val_size])
        return train, val, test
    else:
        return train, test
