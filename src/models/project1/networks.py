import torch.nn as nn
from collections import OrderedDict

def get_network(network_name: str):

    if network_name == 'test':
        return nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, kernel_size=(7,7), stride=(1,1), padding=(1,1))),    # B x 64 x 222 x 222
            ('maxpool1', nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))),                    # B x 64 x 111 x 111

            ('linear1', nn.Linear(64*111*111, 128)),                                        # B x 128
            ('classifier', nn.Linear(128, 2)),
        ]))
    else:
        raise NotImplementedError("Not defined yet...")