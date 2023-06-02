#!/usr/bin/env python3


import torch
import torch.nn as nn
import torch.nn.functional as F
#import pytorch_lightning as pl
import lightning.pytorch as pl

def get_model(model_name, loss_function):
    
    # get loss function
    if loss_function == 'CrossEntropy':
        loss_fun = nn.CrossEntropyLoss()
    elif loss_function == 'NLL':
        loss_fun = NLL
    # get model
    if model_name == 'initial':
        network = Network(initial, loss_fun)

    elif model_name == 'transfer':
        todo=1

    return network

# loss functions
def NLL(output, target):
    return F.nll_loss(torch.log(output), target)

# models

# basemodel
class Network(pl.LightningModule):
    def __init__(self, network, loss_fun):
        super(Network, self).__init__()
        self.network = network
        self.loss_fun = loss_fun

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        output = self.network(x)

        loss = self.loss_fun(output, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# models
initial = nn.Sequential(
    nn.Conv2d(3, 6, 5),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(6, 16, 5),
    nn.Linear(16 * 5 * 5, 120),
    nn.Linear(120, 84),
    nn.Linear(84, 10),
    nn.Softmax(dim=1)
    )

# transfer model

