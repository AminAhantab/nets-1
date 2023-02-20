from itertools import cycle
from typing import List

import torch
from torch.nn import functional as F

from nets import MaskedNetwork, MaskedLinear


class LeNetFeedForwardNetwork(MaskedNetwork):
    """
    A LeNet feed-forward network with two hidden layers.

    This network is based on the "Two-Hidden-Layer Fully Connected Multilayer
    NN" (LeCun et al., 1998) with ReLU activations, negative log likelihood
    (`log_softmax`) output, cross-entropy loss, and variable input/output sizes.

    References:
        - LeCun, Y., Bottou, L., Bengio, Y., Haffner, P., 1998. Gradient-based
        learning applied to document recognition. Proceedings of the IEEE 86,
        2278–2324. https://doi.org/10.1109/5.726791

    Attributes:
        layer_1 (MaskedLinear): The first hidden layer.
        layer_2 (MaskedLinear): The second hidden layer.
        layer_3 (MaskedLinear): The output layer.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Initialize the network.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
        """
        super().__init__()

        h1, h2 = 300, 100
        self.layer_1 = MaskedLinear(in_features, h1, bias=bias)
        self.layer_2 = MaskedLinear(h1, h2, bias=bias)
        self.layer_3 = MaskedLinear(h2, out_features, bias=bias)

    @property
    def layers(self) -> List[MaskedLinear]:
        """
        The layers of the network.

        Returns:
            list[MaskedLinear]: The layers of the network.
        """
        return [self.layer_1, self.layer_2, self.layer_3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): The input of the network.

        Returns:
            torch.Tensor: The output of the network.
        """
        # Flatten the input
        batch_size = x.size()[0]
        x = x.view(batch_size, -1)

        # Apply the layers
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx < len(self.layers) - 1:
                x = F.relu(x)

        # Apply softmax
        x = torch.log_softmax(x, dim=1)
        return x

    def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Cross-entropy loss function.

        Args:
            logits (torch.Tensor): The logits of the network.
            labels (torch.Tensor): The labels of the batch.

        Returns:
            torch.Tensor: The loss.
        """
        return F.cross_entropy(logits, labels)

    def accuracy(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Accuracy metric.

        Args:
            logits (torch.Tensor): The logits of the network.
            labels (torch.Tensor): The labels of the batch.

        Returns:
            torch.Tensor: The accuracy.
        """
        preds = torch.argmax(logits, dim=1)
        return torch.sum(preds == labels).float() / len(labels)


if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt

    def run(iterations=1_000, batch_size=64, learning_rate=0.01, log_interval=100):
        """
        Train the network on MNIST.
        """
        from torch.utils.data import DataLoader
        from torchvision.datasets import MNIST
        from torchvision import transforms

        # transforms
        # prepare transforms standard to MNIST
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        # data
        mnist_train = MNIST(root="data", train=True, download=True, transform=transform)
        mnist_train = DataLoader(mnist_train, batch_size=batch_size)

        mnist_val = MNIST(root="data", train=False, download=True, transform=transform)
        mnist_val = DataLoader(mnist_val, batch_size=batch_size)

        # model
        model = LeNetFeedForwardNetwork(28 * 28, 10)

        # infinite iterator over the training set
        mnist_train = cycle(mnist_train)

        # optimiser
        optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

        def validate(X, y):
            logits = model(X)
            loss = model.loss(logits, y)
            acc = model.accuracy(logits, y)
            return loss.item(), acc.item()

        def validation_loop():
            val_loss, val_acc = 0.0, 0.0
            for batch in mnist_val:
                loss, acc = validate(*batch)
                val_loss += loss
                val_acc += acc
            val_loss /= len(mnist_val)
            val_acc /= len(mnist_val)
            return val_loss, val_acc

        def train(X, y):
            logits = model(X)
            loss = model.loss(logits, y)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            return loss.item()

        def train_loop():
            iteration = 0
            columns = ["iteration", "loss", "val_loss", "val_acc"]
            df = pd.DataFrame(columns=columns)
            for batch in mnist_train:
                loss = train(*batch)
                if iteration % log_interval == 0:
                    val_loss, val_acc = validation_loop()
                    res = [iteration, loss, val_loss, val_acc]
                    df.loc[len(df)] = res
                    print(
                        f"Iteration: {iteration}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                    )

                if iteration == iterations:
                    break

                iteration += 1
            return df

        return train_loop()

    df = run(iterations=500, batch_size=64, learning_rate=0.01, log_interval=100)
    df.plot(x="iteration", y=["loss", "val_loss"])
    plt.show()
