import torch.nn as nn
from collections import OrderedDict

def get_network(network_name: str):

    if network_name == 'test':
        return nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv2', nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)),
            ('relu2', nn.ReLU()),
            ('pool2', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv3', nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
            ('relu3', nn.ReLU()),
            ('pool3', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('flatten', nn.Flatten()),
            ('fc1', nn.Linear(64 * 28 * 28, 256)),
            ('relu4', nn.ReLU()),
            ('fc2', nn.Linear(256, 2))
        ]))
    else:
        raise NotImplementedError("Not defined yet...")