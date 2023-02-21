from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F

from nets.nn import MaskedConv2d, MaskedLinear, MaskedNetwork


class ConvTwoNeuralNetwork(MaskedNetwork):
    """
    A convolutional neural network with two convolutional layers and three
    fully-connected layers.

    This network is based on the variants of Simoyan and Zisserman's (2015)
    orignal VGG architecture presented in Frankle and Carbin (2019).

    References:
        - Simonyan, K., Zisserman, A., 2015. Very deep convolutional networks
          for large-scale image recognition.
          https://doi.org/10.48550/arXiv.1409.1556
        - Frankle, J., Carbin, M., 2019. The lottery ticket hypothesis: Finding
          sparse, trainable neural networks.
          https://doi.org/10.48550/arXiv.1803.03635

    Attributes:
        conv1 (MaskedConv2d): The first convolutional layer.
        conv2 (MaskedConv2d): The second convolutional layer.
        fc1 (MaskedLinear): The first fully-connected layer.
        fc2 (MaskedLinear): The second fully-connected layer.
        fc3 (MaskedLinear): The third fully-connected layer.
    """

    def __init__(
        self,
        in_channels: int,
        in_features: Tuple[int, int],
        out_features: int,
        bias: bool = True,
    ):
        super().__init__(in_channels, in_features, out_features)

        # Define the sizes of the convolutional layers
        # from Frankle and Carbin (2019)
        conv_sizes = [64, 64]
        conv_1_channels = (in_channels, conv_sizes[0])
        conv_2_channels = (conv_sizes[0], conv_sizes[1])

        # Define the sizes of the fully-connected layers
        # from Frankle and Carbin (2019)
        fc_sizes = [256, 256, out_features]
        post_conv_size = (in_features[0] - 2 * 2, in_features[1] - 2 * 2)
        fc1_size = conv_sizes[1] * (post_conv_size[0] // 2) * (post_conv_size[1] // 2)

        # Define the layers
        self.conv1 = MaskedConv2d(*conv_1_channels, 3, padding=1, bias=bias)
        self.conv2 = MaskedConv2d(*conv_2_channels, 3, padding=1, bias=bias)
        self.pool1 = nn.MaxPool2d(2, 2, padding=0)
        self.fc1 = MaskedLinear(fc1_size, fc_sizes[0], bias=bias)
        self.fc2 = MaskedLinear(fc_sizes[0], fc_sizes[1], bias=bias)
        self.fc3 = MaskedLinear(fc_sizes[1], fc_sizes[2], bias=bias)

    @property
    def layers(self):
        return [
            self.conv1,
            self.conv2,
            # pool layers are not masked
            self.fc1,
            self.fc2,
            self.fc3,
        ]

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]

        # convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        # fully-connected layers
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # softmax
        x = torch.log_softmax(x, dim=1)

        return x

    def loss(self, x, y):
        return F.cross_entropy(x, y)

    def accuracy(self, x, y):
        return (x.argmax(dim=1) == y).float().mean()


class ConvFourNeuralNetwork(MaskedNetwork):
    """
    A convolutional neural network with four convolutional layers and three
    fully-connected layers.

    This network is based on the variants of Simoyan and Zisserman's (2015)
    orignal VGG architecture presented in Frankle and Carbin (2019).

    References:
        - Simonyan, K., Zisserman, A., 2015. Very deep convolutional networks
          for large-scale image recognition.
          https://doi.org/10.48550/arXiv.1409.1556
        - Frankle, J., Carbin, M., 2019. The lottery ticket hypothesis: Finding
          sparse, trainable neural networks.
          https://doi.org/10.48550/arXiv.1803.03635

    Attributes:
        conv1 (MaskedConv2d): The first convolutional layer.
        conv2 (MaskedConv2d): The second convolutional layer.
        conv3 (MaskedConv2d): The third convolutional layer.
        conv4 (MaskedConv2d): The fourth convolutional layer.
        fc1 (MaskedLinear): The first fully-connected layer.
        fc2 (MaskedLinear): The second fully-connected layer.
        fc3 (MaskedLinear): The third fully-connected layer.
    """

    def __init__(
        self,
        in_channels: int,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__(in_channels, in_features, out_features)

        # Define the sizes of the convolutional layers
        # from Frankle and Carbin (2019)
        conv_sizes = [64, 64, 128, 128]
        conv_1_channels = (in_channels, conv_sizes[0])
        conv_2_channels = (conv_sizes[0], conv_sizes[1])
        conv_3_channels = (conv_sizes[1], conv_sizes[2])
        conv_4_channels = (conv_sizes[2], conv_sizes[3])

        # Define the sizes of the fully-connected layers
        # from Frankle and Carbin (2019)
        fc_sizes = [256, 256, out_features]
        post_conv_size = in_features
        for _ in range(len(conv_sizes) // 2):
            post_conv_size = (post_conv_size[0] - 2 * 2, post_conv_size[1] - 2 * 2)
            post_conv_size = (post_conv_size[0] // 2, post_conv_size[1] // 2)
        fc1_size = conv_sizes[-1] * post_conv_size[0] * post_conv_size[1]

        # Define the layers
        self.conv1 = MaskedConv2d(*conv_1_channels, 3, padding=1, bias=bias)
        self.conv2 = MaskedConv2d(*conv_2_channels, 3, padding=1, bias=bias)
        self.pool1 = nn.MaxPool2d(2, 2, padding=0)
        self.conv3 = MaskedConv2d(*conv_3_channels, 3, padding=1, bias=bias)
        self.conv4 = MaskedConv2d(*conv_4_channels, 3, padding=1, bias=bias)
        self.pool2 = nn.MaxPool2d(2, 2, padding=0)
        self.fc1 = MaskedLinear(fc1_size, fc_sizes[0], bias=bias)
        self.fc2 = MaskedLinear(fc_sizes[0], fc_sizes[1], bias=bias)
        self.fc3 = MaskedLinear(fc_sizes[1], fc_sizes[2], bias=bias)

    @property
    def layers(self):
        return [
            self.conv1,
            self.conv2,
            # pool layers are not masked
            self.conv3,
            self.conv4,
            # pool layers are not masked
            self.fc1,
            self.fc2,
            self.fc3,
        ]

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]

        # convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        # fully-connected layers
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # softmax
        x = torch.log_softmax(x, dim=1)

        return x

    def loss(self, x, y):
        return F.cross_entropy(x, y)

    def accuracy(self, x, y):
        return (x.argmax(dim=1) == y).float().mean()


class ConvSixNeuralNetwork(MaskedNetwork):
    """
    A convolutional neural network with four convolutional layers and three
    fully-connected layers.

    This network is based on the variants of Simoyan and Zisserman's (2015)
    orignal VGG architecture presented in Frankle and Carbin (2019).

    References:
        - Simonyan, K., Zisserman, A., 2015. Very deep convolutional networks
          for large-scale image recognition.
          https://doi.org/10.48550/arXiv.1409.1556
        - Frankle, J., Carbin, M., 2019. The lottery ticket hypothesis: Finding
          sparse, trainable neural networks.
          https://doi.org/10.48550/arXiv.1803.03635

    Attributes:
        conv1 (MaskedConv2d): The first convolutional layer.
        conv2 (MaskedConv2d): The second convolutional layer.
        conv3 (MaskedConv2d): The third convolutional layer.
        conv4 (MaskedConv2d): The fourth convolutional layer.
        fc1 (MaskedLinear): The first fully-connected layer.
        fc2 (MaskedLinear): The second fully-connected layer.
        fc3 (MaskedLinear): The third fully-connected layer.
    """

    def __init__(
        self,
        in_channels: int,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__(in_channels, in_features, out_features)

        # Define the sizes of the convolutional layers
        # from Frankle and Carbin (2019)
        conv_sizes = [64, 64, 128, 128, 256, 256]
        conv_1_channels = (in_channels, conv_sizes[0])
        conv_2_channels = (conv_sizes[0], conv_sizes[1])
        conv_3_channels = (conv_sizes[1], conv_sizes[2])
        conv_4_channels = (conv_sizes[2], conv_sizes[3])
        conv_5_channels = (conv_sizes[3], conv_sizes[4])
        conv_6_channels = (conv_sizes[4], conv_sizes[5])

        # Define the sizes of the fully-connected layers
        # from Frankle and Carbin (2019)
        fc_sizes = [256, 256, out_features]
        post_conv_size = in_features
        for _ in range(len(conv_sizes) // 2):
            print(post_conv_size)
            post_conv_size = (post_conv_size[0] - 2 * 2, post_conv_size[1] - 2 * 2)
            post_conv_size = (post_conv_size[0] // 2, post_conv_size[1] // 2)
        fc1_size = conv_sizes[-1] * post_conv_size[0] * post_conv_size[1]
        print(fc1_size)

        # Define the layers
        self.conv1 = MaskedConv2d(*conv_1_channels, 3, padding=1, bias=bias)
        print(self.conv1.weight.shape)
        self.conv2 = MaskedConv2d(*conv_2_channels, 3, padding=1, bias=bias)
        print(self.conv2.weight.shape)
        self.pool1 = nn.MaxPool2d(2, 2, padding=0)
        self.conv3 = MaskedConv2d(*conv_3_channels, 3, padding=1, bias=bias)
        print(self.conv3.weight.shape)
        self.conv4 = MaskedConv2d(*conv_4_channels, 3, padding=1, bias=bias)
        print(self.conv4.weight.shape)
        self.pool2 = nn.MaxPool2d(2, 2, padding=0)
        self.conv5 = MaskedConv2d(*conv_5_channels, 3, padding=1, bias=bias)
        print(self.conv5.weight.shape)
        self.conv6 = MaskedConv2d(*conv_6_channels, 3, padding=1, bias=bias)
        print(self.conv6.weight.shape)
        self.pool3 = nn.MaxPool2d(2, 2, padding=0)
        self.fc1 = MaskedLinear(fc1_size, fc_sizes[0], bias=bias)
        print(self.fc1.weight.shape)
        self.fc2 = MaskedLinear(fc_sizes[0], fc_sizes[1], bias=bias)
        print(self.fc2.weight.shape)
        self.fc3 = MaskedLinear(fc_sizes[1], fc_sizes[2], bias=bias)
        print(self.fc3.weight.shape)

    @property
    def layers(self):
        return [
            self.conv1,
            self.conv2,
            # pool layers are not masked
            self.conv3,
            self.conv4,
            # pool layers are not masked
            self.conv5,
            self.conv6,
            # pool layers are not masked
            self.fc1,
            self.fc2,
            self.fc3,
        ]

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]

        # convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)

        # fully-connected layers
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # softmax
        x = torch.log_softmax(x, dim=1)

        return x

    def loss(self, x, y):
        return F.cross_entropy(x, y)

    def accuracy(self, x, y):
        return (x.argmax(dim=1) == y).float().mean()


if __name__ == "__main__":
    from itertools import cycle

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
        model = ConvTwoNeuralNetwork(1, (28, 28), 10, bias=False)

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

    df = run(iterations=500, batch_size=64, learning_rate=0.001, log_interval=100)
    df.plot(x="iteration", y=["loss", "val_loss"])
    plt.show()
