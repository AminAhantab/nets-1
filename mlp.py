import torch

from stronglth.nn import MaskedNetwork, MaskedLinear


class MultiLayerPerceptron(MaskedNetwork):
    """A multi-layer perceptron network."""

    def __init__(self, h1: int = 10):
        """
        Initialize the network.

        Args:
            h1 (int): The number of neurons in the first hidden layer.
        """
        super().__init__()

        # xor inputs are (2), xor outputs are (1)
        self.layer_1 = MaskedLinear(2, h1)
        self.layer_2 = MaskedLinear(h1, 1)

    @property
    def layers(self) -> list[MaskedLinear]:
        """
        The layers of the network.

        Returns:
            list[MaskedLinear]: The layers of the network.
        """
        return [self.layer_1, self.layer_2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): The input of the network.

        Returns:
            torch.Tensor: The output of the network.
        """
        # layer 1 (b, 2) -> (b, 10)
        x = self.layer_1(x)
        x = torch.sigmoid(x)

        # layer 2 (b, 10) -> (b, 1)
        x = self.layer_2(x)
        x = torch.sigmoid(x)

        return x

    def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Mean squared error loss function.

        Args:
            logits (torch.Tensor): The logits of the network.
            labels (torch.Tensor): The labels of the batch.

        Returns:
            torch.Tensor: The loss.
        """
        return ((logits - labels) ** 2).mean()

    def accuracy(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Accuracy of the network.

        Args:
            logits (torch.Tensor): The logits of the network.
            labels (torch.Tensor): The labels of the batch.

        Returns:
            torch.Tensor: The accuracy.
        """
        x = (logits.round() - labels).abs().sum()
        return 1 - x / labels.shape[0]


if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    import xor
    from itertools import cycle
    from torch.utils.data import DataLoader

    def run(iterations=1_000, learning_rate=0.01, log_interval=100):
        """
        Train the network on MNIST.
        """

        data = xor.load()
        model = MultiLayerPerceptron()

        # infinite iterator over the training set
        loader = DataLoader(data)
        loader = cycle(iter(loader))
        optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

        def train(X, y):
            logits = model(X)
            loss = model.loss(logits, y)
            acc = model.accuracy(logits, y)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            return loss.item(), acc.item()

        def train_loop():
            iteration = 0
            columns = ["iteration", "loss", "acc"]
            df = pd.DataFrame(columns=columns)
            for batch in loader:
                loss, acc = train(*batch)
                if iteration % log_interval == 0:
                    res = [iteration, loss, acc]
                    df.loc[len(df)] = res
                    print(res)

                if iteration == iterations:
                    break

                iteration += 1
            return df

        return train_loop()

    df = run(iterations=500, batch_size=64, learning_rate=0.1, log_interval=100)
    df.plot(x="iteration", y=["loss", "acc"])
    plt.show()
