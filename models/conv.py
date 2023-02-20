import torch
from torch import nn
from torch.nn import functional as F
from nets.nn.layers import MaskedConv2d, MaskedLinear

from nets.nn.masked import MaskedNetwork


class ConvTwoNeuralNetwork(MaskedNetwork):
    """
    A convolutional neural network with two convolutional layers and two
    fully-connected layers.

    This network are based on the variants of Simoyan and Zisserman's (2015)
    orignal VGG architecture presented in Frankle and Carbin (2018).

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
        pool1 (nn.MaxPool2d): The first max-pooling layer.
        fc1 (MaskedLinear): The first fully-connected layer.
        fc2 (MaskedLinear): The second fully-connected layer.

    """

    def __init__(self, input_channels, output_size):
        super().__init__()

        self.conv1 = MaskedConv2d(input_channels, 64, 3, padding=1)
        self.conv2 = MaskedConv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2, padding=0)
        self.fc1 = MaskedLinear(64 * 14 * 14, 256)
        self.fc2 = MaskedLinear(256, 256)
        self.fc3 = MaskedLinear(256, output_size)

    @property
    def layers(self):
        return [
            self.conv1,
            self.conv2,
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
        
        x = torch.log_softmax(x, dim=1)

        return x

    def loss(self, x, y):
        return F.cross_entropy(x, y)

    def accuracy(self, x, y):
        return (x.argmax(dim=1) == y).float().mean()


class ConvFourNeuralNetwork(MaskedNetwork):
    """
    A convolutional neural network with four convolutional layers and two
    fully-connected layers.

    This network are based on the variants of Simoyan and Zisserman's (2014)
    orignal VGG architecture presented in Frankle and Carbin (2018).

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
        pool1 (nn.MaxPool2d): The first pooling layer.
        conv3 (MaskedConv2d): The third convolutional layer.
        conv4 (MaskedConv2d): The fourth convolutional layer.
        pool2 (nn.MaxPool2d): The second pooling layer.
        fc1 (MaskedLinear): The first fully-connected layer.
        fc2 (MaskedLinear): The second fully-connected layer.

    """

    def __init__(self, input_channels, output_size):
        super().__init__()

        self.conv1 = MaskedConv2d(input_channels, 64, 3)
        self.conv2 = MaskedConv2d(64, 64, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = MaskedConv2d(64, 128, 3)
        self.conv4 = MaskedConv2d(128, 128, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = MaskedLinear(256, 256)
        self.fc2 = MaskedLinear(256, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 256)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def loss(self, x, y):
        return F.cross_entropy(x, y)

    def accuracy(self, x, y):
        return (x.argmax(dim=1) == y).float().mean()


class ConvSixNeuralNetwork(MaskedNetwork):
    """
    A convolutional neural network with six convolutional layers and two
    fully-connected layers.

    This network are based on the variants of Simoyan and Zisserman's (2014)
    orignal VGG architecture presented in Frankle and Carbin (2018).

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
        pool1 (nn.MaxPool2d): The first pooling layer.
        conv3 (MaskedConv2d): The third convolutional layer.
        conv4 (MaskedConv2d): The fourth convolutional layer.
        pool2 (nn.MaxPool2d): The second pooling layer.
        fc1 (MaskedLinear): The first fully-connected layer.
        fc2 (MaskedLinear): The second fully-connected layer.

    """

    def __init__(self, input_channels, output_size):
        super().__init__()

        self.conv1 = MaskedConv2d(input_channels, 64, 3)
        self.conv2 = MaskedConv2d(64, 64, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = MaskedConv2d(64, 128, 3)
        self.conv4 = MaskedConv2d(128, 128, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = MaskedConv2d(128, 256, 3)
        self.conv6 = MaskedConv2d(256, 256, 3)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = MaskedLinear(256, 256)
        self.fc2 = MaskedLinear(256, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = x.view(-1, 256)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def loss(self, x, y):
        return F.cross_entropy(x, y)

    def accuracy(self, x, y):
        return (x.argmax(dim=1) == y).float().mean()
