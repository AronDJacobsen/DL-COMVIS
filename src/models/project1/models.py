import timm
import torch
from torch import nn

# import lightning.pytorch as pl
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint 
from torchmetrics.classification import BinaryAccuracy

from .networks import get_network

def get_model(network_name):
    return HotdogEfficientNet if 'efficient' in network_name else CNNModel

def get_optimizer(args, network):
    if args.optimizer == 'Adam':
        return torch.optim.Adam(network.parameters(), lr = args.lr)
    else:
        raise NotImplementedError("Implement other optimizers when getting this error...")


### BASEMODEL ###
class CNNModel(pl.LightningModule):
    def __init__(self, args):
        super(CNNModel, self).__init__()
        
        self.args = args
        
        # Load network
        self.network = get_network(args.network_name)

        # Define metrics and loss criterion
        self.criterion = nn.BCEWithLogitsLoss()
        self.accuracy = BinaryAccuracy().to(self.device)

        # Set up logging option
        self.model_checkpoint = ModelCheckpoint(
            monitor = "val_loss",
            verbose = args.verbose,
            filename = "{epoch}_{val_loss:.4f}",
        )

    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        return get_optimizer(self.args, self.network)
    
    def training_step(self, batch, batch_idx):
        # Extract and process input
        x, y = batch
        y = torch.nn.functional.one_hot(y, num_classes=2) 
        y = y.to(torch.float32)

        # Get prediction, loss and accuracy
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat, y)

        # logs metrics for each training_step - [default:True],
        # the average across the epoch, to the progress bar and logger-[default:False]
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True),
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Extract and process input
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        y = torch.nn.functional.one_hot(y, num_classes=2) 
        y = y.to(torch.float32) 

        # Get prediction, loss and accuracy
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat, y)

        # logs metrics for each validation_step - [default:False]
        #the average across the epoch - [default:True]
        self.log("val_acc", acc, prog_bar=True, logger=True),
        self.log("val_loss", loss, prog_bar=True, logger=True)



### Transfer learning model ###
class HotdogEfficientNet(pl.LightningModule):
    def __init__(self, args):
        super(HotdogEfficientNet, self).__init__()

        self.args = args
        
        # Load model
        self.network = timm.create_model(args.network_name, pretrained=True, num_classes=2)
        # Freeze weights
        for param in self.network.parameters():
            param.requires_grad = False
        
        # Require gradient for classification layer
        self.network.classifier.requires_grad_()

        # Define metrics and loss criterion
        self.criterion = nn.BCEWithLogitsLoss()
        self.accuracy = BinaryAccuracy().to(self.device)

        # Set up logging option
        self.model_checkpoint = ModelCheckpoint(
            monitor = "val_loss",
            verbose = args.verbose,
            filename = "{epoch}_{val_loss:.4f}",
        )

    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        return get_optimizer(self.args, self.network)
        
    def training_step(self, batch, batch_idx):
        # Extract and process input
        x, y = batch
        y = torch.nn.functional.one_hot(y, num_classes=2) 
        y = y.to(torch.float32)

        # Get prediction, loss and accuracy
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat, y)

        # logs metrics for each training_step - [default:True],
        # the average across the epoch, to the progress bar and logger-[default:False]
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True),
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Extract and process input
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        y = torch.nn.functional.one_hot(y, num_classes=2) 
        y = y.to(torch.float32) 

        # Get prediction, loss and accuracy
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat, y)

        # logs metrics for each validation_step - [default:False]
        #the average across the epoch - [default:True]
        self.log("val_acc", acc, prog_bar=True, logger=True),
        self.log("val_loss", loss, prog_bar=True, logger=True)