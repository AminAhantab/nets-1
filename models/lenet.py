from itertools import cycle

import torch
from torch.nn import functional as F

from nets import MaskedNetwork, MaskedLinear


class LeNetFeedForwardNetwork(MaskedNetwork):
    """A LeNet feed-forward network."""

    def __init__(self, h1: int = 300, h2: int = 100, bias: bool = True):
        """
        Initialize the network.

        Args:
            h1 (int): The number of neurons in the first hidden layer.
            h2 (int): The number of neurons in the second hidden layer.
        """
        super().__init__()

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = MaskedLinear(28 * 28, h1, bias=bias)
        self.layer_2 = MaskedLinear(h1, h2, bias=bias)
        self.layer_3 = MaskedLinear(h2, 10, bias=bias)

    @property
    def layers(self) -> list[MaskedLinear]:
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
        batch_size, _, _, _ = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        # layer 1 (b, 1*28*28) -> (b, 300)
        x = self.layer_1(x)
        x = torch.relu(x)

        # layer 2 (b, 300) -> (b, 100)
        x = self.layer_2(x)
        x = torch.relu(x)

        # layer 3 (b, 100) -> (b, 10)
        x = self.layer_3(x)

        # probability distribution over labels
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
        model = LeNetFeedForwardNetwork()

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

                if iteration == iterations:
                    break

                iteration += 1
            return df

        return train_loop()

    df = run(iterations=500, batch_size=64, learning_rate=0.01, log_interval=100)
    df.plot(x="iteration", y=["loss", "val_loss"])
    plt.show()
