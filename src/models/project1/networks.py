import torch.nn as nn
from collections import OrderedDict

def get_network(network_name: str):

    if network_name == 'initial':
        return nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 6, 5)),
            ('maxpool1', nn.MaxPool2d(2, 2)),

            ('conv2', nn.Conv2d(6, 16, 5)),

            ('linear1', nn.Linear(16 * 5 * 5, 120)),
            ('linear2', nn.Linear(120, 84)),
            ('classifier', nn.Linear(84, 2)),
        ]))
    else:
        raise NotImplementedError("Not defined yet...")