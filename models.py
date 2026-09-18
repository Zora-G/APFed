import torch
from torch import nn
import torch.nn.functional as F


def _activation(name):
    if name == 'relu':
        return F.relu
    if name == 'gelu':
        return F.gelu
    if name == 'tanh':
        return torch.tanh
    raise ValueError('unknown activation: ' + str(name))

def _pool(x, name):
    if name == 'max':
        return F.max_pool2d(x, 2)
    if name == 'avg':
        return F.avg_pool2d(x, 2)
    raise ValueError('unknown pooling: ' + str(name))

def _widths(value, expected, name):
    if isinstance(value, str):
        value = tuple(int(piece.strip()) for piece in value.split(',') if piece.strip())
    widths = tuple(int(width) for width in value)
    if len(widths) != expected or any(width <= 0 for width in widths):
        raise ValueError(name + ' must contain ' + str(expected) + ' positive widths')
    return widths

class CNNMnist(nn.Module):
    def __init__(self, num_classes=10, channels=(16, 32, 32, 64),
                 fc_width=128, activation='relu', pooling='max'):
        super(CNNMnist, self).__init__()
        channels = _widths(channels, 4, 'mnist_channels')
        if int(fc_width) <= 0:
            raise ValueError('mnist_fc_width must be positive')
        self.conv1 = nn.Conv2d(1, channels[0], kernel_size=3)
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=3)
        self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=3)
        self.conv4 = nn.Conv2d(channels[2], channels[3], kernel_size=2)
        self.fc1 = nn.Linear(channels[3] * 4 * 4, int(fc_width))
        self.fc2 = nn.Linear(int(fc_width), num_classes)
        self.activation_name = activation
        self.pooling_name = pooling

    def forward(self, x):
        activation = _activation(self.activation_name)
        x = activation(self.conv1(x))
        x = activation(self.conv2(x))
        x = _pool(x, self.pooling_name)
        x = activation(self.conv3(x))
        x = activation(self.conv4(x))
        x = _pool(x, self.pooling_name)
        x = x.view(x.size(0), -1)
        x = activation(self.fc1(x))
        return self.fc2(x)
