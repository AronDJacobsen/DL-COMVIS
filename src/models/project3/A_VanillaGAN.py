import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.net = nn.Sequential(OrderedDict([
            (f'linear1', nn.Linear(100, 2848)),
            (f'act1', nn.LeakyReLU()),
            (f'batchnorm1', nn.BatchNorm1d(2848)),
            (f'linear2', nn.Linear(2848, 2848)),
            (f'act2', nn.LeakyReLU()),
            (f'batchnorm2', nn.BatchNorm1d(2848)),
            (f'linear3', nn.Linear(2848, 2848)),
            (f'act3', nn.LeakyReLU()),
            (f'batchnorm3', nn.BatchNorm1d(2848)),
            (f'linear4', nn.Linear(2848, 2848)),
            (f'act4', nn.LeakyReLU()),
            (f'batchnorm3', nn.BatchNorm1d(2848)),
            (f'linear_out', nn.Linear(2848, 28*28)),
            (f'act_out', nn.Tanh()),
          ]))

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), 1, 28, 28)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.net = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(28*28, 1024)),
            ('act1', nn.LeakyReLU()),
            ('dropout1', nn.Dropout(p=0.2)),
            ('linear2', nn.Linear(1024, 512)),
            ('act2', nn.LeakyReLU()),
            ('dropout2', nn.Dropout(p=0.2)),
            ('linear3', nn.Linear(512, 256)),
            ('act3', nn.LeakyReLU()),
            ('dropout3', nn.Dropout(p=0.2)),
            ('classifier', nn.Linear(256, 1))
        ]))


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.net(x)
        return x

class VanillaGAN:
    """
    Wrapper class for the two modules.
    """
    def __init__(self, device):

        self.device = device

        # Initialize networks
        self.d = Discriminator().to(device)
        self.g = Generator().to(device)

        # Initialize optimizers
        self.d_opt = torch.optim.Adam(self.d.parameters(), 0.0004, (0.5, 0.999))
        self.g_opt = torch.optim.Adam(self.g.parameters(), 0.0001, (0.5, 0.999))

    def discriminator_loss(self, x_real, x_fake):
        # detach x_fake for not backtracking through generator
        return - F.logsigmoid(self.d(x_real)).mean() - torch.log(1 - F.sigmoid(self.d(x_fake.detach()))).mean() 
    
    def generator_loss(self, x_real, x_fake):
        return torch.log(1 - F.sigmoid(self.d(x_fake))).mean()