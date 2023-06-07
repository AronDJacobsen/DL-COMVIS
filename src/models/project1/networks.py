import torch.nn as nn
from collections import OrderedDict

def get_network(network_name: str, args):

    def norm(norm_type, layer_num, num_features):
        if norm_type == 'batchnorm':
            return (f'bn{layer_num}', nn.BatchNorm2d(num_features))
        elif norm_type == 'layernorm':
            # image shaped divided by 2 ** number of times we've done maxpooling
            channel_dim = 224 / 2**(layer_num - 1)
            return (f'ln{layer_num}', nn.LayerNorm([num_features, channel_dim, channel_dim]))
        elif norm_type == 'instancenorm':
            return (f'in{layer_num}', nn.InstanceNorm2d(num_features))
        elif norm_type == 'none':
            #pass
            return (f'none{layer_num}', nn.Identity())
        else:
            raise NotImplemented('Normalization-technique not yet implemented')

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

    elif network_name == 'initial':
        return nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)),
            ('relu1', nn.ReLU()),
            norm(norm_type = args.norm, layer_num = 1, num_features = 16),
            ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('dropout1', nn.Dropout(p=0.2)),
            
            ('conv2', nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)),
            ('relu2', nn.ReLU()),
            norm(norm_type = args.norm, layer_num = 1, num_features = 16),
            ('pool2', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('dropout2', nn.Dropout(p=0.2)),
            
            ('conv3', nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
            ('relu3', nn.ReLU()),
            norm(norm_type = args.norm, layer_num = 3, num_features = 64),
            ('pool3', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('dropout3', nn.Dropout(p=0.2)),
            
            ('conv4', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)),
            ('relu4', nn.ReLU()),
            norm(norm_type = args.norm, layer_num = 4, num_features = 128),
            ('pool4', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('dropout4', nn.Dropout(p=0.2)),
            
            ('flatten', nn.Flatten()),
            ('fc1', nn.Linear(128 * 14 * 14, 256)),
            ('relu4', nn.ReLU()),
            ('dropout5', nn.Dropout(p=0.2)),
            ('fc2', nn.Linear(256, 2))
        ]))
	

    else:
        raise NotImplementedError("Not defined yet...")
