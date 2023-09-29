import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv2d_1 = nn.Conv2d(1, 32, (1, 7), 1)
        self.conv2d_2 = nn.Conv2d(32, 32, (1, 7), 2)
        self.dense_1 = nn.Linear(1984, 128)
        self.dense_2 = nn.Linear(128, 128)
        self.dense_3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = F.relu(self.conv2d_1(x))
        x = F.max_pool2d(x, (4, 3), (1, 3))
        x = F.relu(self.conv2d_2(x))
        x = F.max_pool2d(x, (1, 3), (1, 3))
        x = self.flatten(x)
        x = F.relu(self.dense_1(x))
        x = self.dropout(x)
        x = F.relu(self.dense_2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.dense_3(x))
        return x.squeeze()


def init_layer(layer):
    """Initialize a Linear or Convolutional layer.
    Ref: He, Kaiming, et al. "Delving deep into rectifiers: Surpassing
    human-level performance on imagenet classification." Proceedings of the
    IEEE international conference on computer vision. 2015.

    Input
        layer: torch.Tensor - The current layer of the neural network
    """

    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
    elif layer.weight.ndimension() == 3:
        (n_out, n_in, height) = layer.weight.size()
        n = n_in * height
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """
    Initialize a Batchnorm layer.

    Input
        bn: torch.Tensor - The batch normalisation layer
    """

    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock1d(nn.Module):
    """
    Creates an instance of a 1D convolutional layer. This includes the
    convolutional filter but also the type of normalisation "batch" or
    "weight", the activation function, and initialises the weights.
    """
    def __init__(self, in_channels, out_channels, kernel, stride, pad,
                 normalisation, dil=1):
        super(ConvBlock1d, self).__init__()
        self.norm = normalisation
        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel,
                               stride=stride,
                               padding=pad,
                               dilation=dil)
        if self.norm == 'bn':
            self.bn1 = nn.BatchNorm1d(out_channels)
        elif self.norm == 'wn':
            self.conv1 = nn.utils.weight_norm(self.conv1, name='weight')
        else:
            self.conv1 = self.conv1
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        Initialises the weights of the current layer
        """
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input):
        """
        Passes the input through the convolutional filter

        Input
            input: torch.Tensor - The current input at this stage of the network
        """
        x = input
        if self.norm == 'bn':
            x = self.relu(self.bn1(self.conv1(x)))
        else:
            x = self.relu(self.conv1(x))

        return x


def init_att_layer(layer):
    """
    Initilise the weights and bias of the attention layer to 1 and 0
    respectively. This is because the first iteration through the attention
    mechanism should weight each time step equally.

    Input
        layer: torch.Tensor - The current layer of the neural network
    """
    layer.weight.data.fill_(1.)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


class FullyConnected(nn.Module):
    """
    Creates an instance of a fully-connected layer. This includes the
    hidden layers but also the type of normalisation "batch" or
    "weight", the activation function, and initialises the weights.
    """
    def __init__(self, in_channels, out_channels, activation, normalisation,
                 att=None):
        super(FullyConnected, self).__init__()
        self.att = att
        self.norm = normalisation
        self.fc = nn.Linear(in_features=in_channels,
                            out_features=out_channels)
        if activation == 'sigmoid':
            self.act = nn.Sigmoid()
            self.norm = None
        elif activation == 'softmax':
            self.act = nn.Softmax(dim=-1)
            self.norm = None
        elif activation == 'global':
            self.act = None
            self.norm = None
        else:
            self.act = nn.ReLU()
            if self.norm == 'bn':
                self.bnf = nn.BatchNorm1d(out_channels)
            elif self.norm == 'wn':
                self.wnf = nn.utils.weight_norm(self.fc, name='weight')

        self.init_weights()

    def init_weights(self):
        """
        Initialises the weights of the current layer
        """
        if self.att:
            init_att_layer(self.fc)
        else:
            init_layer(self.fc)
        if self.norm == 'bn':
            init_bn(self.bnf)

    def forward(self, input):
        """
        Passes the input through the fully-connected layer

        Input
            input: torch.Tensor - The current input at this stage of the network
        """
        x = input
        if self.norm is not None:
            if self.norm == 'bn':
                x = self.act(self.bnf(self.fc(x)))
            else:
                x = self.act(self.wnf(x))
        else:
            if self.att:
                if self.act:
                    x = self.act(self.fc(x))
                else:
                    x = self.fc(x)
            else:
                if self.act:
                    x = self.act(self.fc(x))
                else:
                    x = self.fc(x)

        return x


class CustomMel7(nn.Module):
    def __init__(self):
        super(CustomMel7, self).__init__()
        self.conv = ConvBlock1d(in_channels=40,
                                out_channels=128,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.drop = nn.Dropout(0.05)
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=False)
        self.fc = FullyConnected(in_channels=128,
                                 out_channels=1,
                                 activation='sigmoid',
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv(x)
        x = self.pool(x)
        x = self.drop(x)
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))
        return x


"""
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv2d_1 = nn.Conv2d(1, 32, (1, 7), 1)
        self.conv2d_2 = nn.Conv2d(32, 32, (1, 7), 2)
        self.dense_1 = nn.Linear(5952, 128)
        self.dense_2 = nn.Linear(128, 128)
        self.dense_3 = nn.Linear(128, 1)
        # self.batch_norm_2d = nn.BatchNorm2d(32)
        # self.batch_norm_1d = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.avg_pool = nn.AvgPool2d((31, 2))
        self.seq1 = nn.Sequential([
            nn.Conv2d(1, 32, (1, 7), 1),
            nn.ReLU(),
        ])

    def new_forward(self, x):
        x = F.relu(self.conv2d_1(x))
        x = F.max_pool2d(x, (4, 3), (1, 3))
        x = self.batch_norm_2d(x)  # TODO: this was not in the original network
        x = F.relu(self.conv2d_2(x))
        # x = F.max_pool2d(x, (1, 3), (1, 3))  # TODO: REMOVED
        x = F.relu(self.conv2d_2(x))  # TODO: INCLUDED
        x = self.avg_pool(x)  # TODO: INCLUDED
        x = self.flatten(x)
        x = F.relu(self.dense_1(x))
        x = self.dropout(x)  # TODO: this was not in the original network
        x = self.batch_norm_1d(x)  # TODO: this was not in the original network
        x = F.relu(self.dense_2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.dense_3(x))
        return x.squeeze()

    # TODO copy resnet batch norm
    def forward(self, x):
        x = F.relu(self.conv2d_1(x))
        x = self.seq1(x)
        x = F.max_pool2d(x, (4, 3), (1, 3))
        x = F.relu(self.conv2d_2(x))
        x = F.max_pool2d(x, (1, 3), (1, 3))  # TODO: REMOVED
        # x = F.relu(self.conv2d_2(x))  # TODO: INCLUDED
        x = self.flatten(x)
        x = F.relu(self.dense_1(x))
        x = self.dropout(x)  # TODO: this was not in the original network
        x = F.relu(self.dense_2(x))
        x = self.dropout(x)
        # x = torch.sigmoid(self.dense_3(x))
        x = self.dense_3(x)
        return x.squeeze()

    def forward(self, x):
        print(x.shape)
        x = F.relu(self.conv2d_1(x))
        print(x.shape)
        x = F.max_pool2d(x, (4, 3), (1, 3))
        print(x.shape)
        x = self.batch_norm_2d(x)  # TODO: this was not in the original network
        x = F.relu(self.conv2d_2(x))
        print(x.shape)
        # x = F.max_pool2d(x, (1, 3), (1, 3))  # TODO: REMOVED
        x = F.relu(self.conv2d_2(x))  # TODO: INCLUDED
        print(x.shape)
        x = self.avg_pool(x)  # TODO: INCLUDED
        print(x.shape)
        x = self.flatten(x)
        print(x.shape)
        x = F.relu(self.dense_1(x))
        print(x.shape)
        x = self.dropout(x)  # TODO: this was not in the original network
        x = self.batch_norm_1d(x)  # TODO: this was not in the original network
        x = F.relu(self.dense_2(x))
        print(x.shape)
        x = self.dropout(x)
        x = torch.sigmoid(self.dense_3(x))
        print(x.shape)
        return x.squeeze()
"""