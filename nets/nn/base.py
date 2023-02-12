from abc import ABC, abstractmethod
from typing import Union

import torch
from torch import nn

Batch = tuple[torch.Tensor, torch.Tensor]


class BaseNetwork(ABC, nn.Module):
    """
    Base class for all networks.

    Attributes:
        layers (list[nn.Module]): The layers of the network.

    Methods:
        forward(x): Forward pass of the network.
        loss(logits, labels): Loss function.
        accuracy(logits, labels): Accuracy of the network.
        num_parameters(): Number of parameters of the network.
        num_connections(): Number of connections of the network.
    """

    layers: Union[nn.ModuleList, list[nn.Module]]

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): The input of the network.

        Returns:
            torch.Tensor: The output of the network.
        """
        pass

    @abstractmethod
    def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Loss function.

        Args:
            logits (torch.Tensor): The logits of the network.
            labels (torch.Tensor): The labels of the batch.

        Returns:
            torch.Tensor: The loss of the network.
        """
        pass

    @abstractmethod
    def accuracy(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Accuracy of the network.

        Args:
            logits (torch.Tensor): The logits of the network.
            labels (torch.Tensor): The labels of the batch.

        Returns:
            float: The accuracy of the network.
        """
        pass

    def num_parameters(self) -> int:
        """
        Number of parameters of the network.

        Returns:
            int: The number of parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
