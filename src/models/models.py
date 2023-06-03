#!/usr/bin/env python3


import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import BinaryAccuracy
#import pytorch_lightning as pl
import lightning.pytorch as pl
from pytorch_lightning.callbacks import ModelCheckpoint 

def get_model(model_name, loss_function):
    
    # get loss function
    if loss_function == 'CrossEntropy':
        loss_fun = nn.CrossEntropyLoss()
    elif loss_function == 'NLL':
        loss_fun = NLL
    elif loss_function == 'BCE':
        loss_fun = nn.BCEWithLogitsLoss()
    # get model
    if model_name == 'initial':
        network = Network(initial, loss_fun)

    elif model_name == 'transfer':
        network = HotDogEfficientNet(loss_fun)

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
        return torch.optim.Adam(self.parameters(), lr=1e-3)

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
class HotDogEfficientNet(pl.LightningModule):
    def __init__(self, loss_fun):
        super().__init__()

        self.device1 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = loss_fun #nn.BCEWithLogitsLoss() # nn.BCELoss()
        self.accuracy = BinaryAccuracy().to(self.device1)
        self.efficient_net = timm.create_model('efficientnet_b4', pretrained=True, num_classes=2)
        self.model_checkpoint = ModelCheckpoint(monitor = "val_loss",
                                                verbose=True,
                                                filename="{epoch}_{val_loss:.4f}")

    def forward(self,x):
        return self.efficient_net(x) # out # torch.argmax(nn.functional.softmax(out, dim=1), dim = 1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr = 1e-4)

    def training_step(self,batch,batch_idx):
        x,y = batch
        y = torch.nn.functional.one_hot(y, num_classes=2) 

        y = y.to(torch.float32)

        y_hat = self(x)
        loss = self.criterion(y_hat,y)

        # logs metrics for each training_step - [default:True],
        # the average across the epoch, to the progress bar and logger-[default:False]
        acc = self.accuracy(y_hat,y)
        self.log("train_acc",acc,on_step=False,on_epoch=True,prog_bar=True,logger=True),
        self.log("train_loss",loss,on_step=False,on_epoch=True,prog_bar=True,logger=True)
        return loss

    def validation_step(self,batch,batch_idx):
        x,y = batch
        x,y = x.to(self.device1), y.to(self.device1)
        y = torch.nn.functional.one_hot(y, num_classes=2) 
        
        y = y.to(torch.float32)#.reshape(-1,1)

        y_hat = self(x)
        loss = self.criterion(y_hat,y)
        acc = self.accuracy(y_hat,y)
        # logs metrics for each validation_step - [default:False]
        #the average across the epoch - [default:True]
        self.log("val_acc",acc,prog_bar=True,logger=True),
        self.log("val_loss",loss,prog_bar=True,logger=True)

