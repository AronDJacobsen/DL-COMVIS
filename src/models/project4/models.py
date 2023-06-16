import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateFinder
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import timm
from torchmetrics.classification import BinaryAccuracy


from src.utils import accuracy, IoU


def get_model(model_name, args, loss_fun, optimizer, out=False):
    if model_name == 'EfficientNet':
        return EfficientNet(args, loss_fun, optimizer, out=out)
    else:
        raise ValueError('unknown model name')



### BASEMODEL ###
class BaseModel(pl.LightningModule):
    '''
    Contains all recurring functionality
    '''
    def __init__(self, args, loss_fun, optimizer, out=False):
        super().__init__()
        self.args = args
        self.lr = self.args.lr
        self.loss_fun = loss_fun
        self.optimizer = optimizer
        self.out = out
        self.offset = 0

        # what to log in training and validation
        self.logs = {
            'acc': accuracy,
            }
        # what to calculate when predicting
        self.metrics = {
            'IoU'         : IoU,
            }
        self.log_dict = {}

        
        # checkpointing and logging
        self.model_checkpoint = ModelCheckpoint(
            monitor = "val_loss",
            verbose = args.verbose,
            filename = "{epoch}_{val_loss:.4f}",
        )
        
        self.save_hyperparameters()

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr = self.args.lr)
        
    def training_step(self, batch, batch_idx):
        # extract input
        x, y = batch
        # predict
        y_hat = self.forward(x)
        # loss
        loss = self.loss_fun(y, y_hat)
        # metrics
        y_hat_sig = F.sigmoid(y_hat)
        threshold = torch.tensor([0.5], device = self.device)

        y_hat_sig = (y_hat_sig>threshold).float()*1
        y_hat_sig = y_hat_sig.int()
        y = y.int()
        # log
        for name, fun in self.logs.items():
            self.log('train_'+name, fun(y_hat_sig, y), prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # extract input
        x, y = batch
        # predict
        y_hat = self.forward(x)
        # loss
        loss = self.loss_fun(y, y_hat)
        y_hat_sig = F.sigmoid(y_hat)
        threshold = torch.tensor([0.5], device = self.device)

        y_hat_sig = (y_hat_sig>threshold).float()*1
        y_hat_sig = y_hat_sig.int()
        y = y.int()
        # log
        for name, fun in self.logs.items():
            self.log('val_'+name, fun(y_hat_sig, y), prog_bar=True, logger=True)
        for name, fun in self.metrics.items():
            self.log('val_'+name, fun(y_hat_sig, y))
        self.log("val_loss", loss, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx): 
        # extract input
        x, y = batch
        # predict
        y_hat = self.forward(x)
        # loss
        loss = self.loss_fun(y, y_hat)
        # predicting
        y_hat_sig = F.sigmoid(y_hat)#.detach().cpu() # todo?
        threshold = torch.tensor([0.5], device = self.device)

        y_hat_sig = (y_hat_sig>threshold).float()*1
        y_hat_sig = y_hat_sig.int()
        y = y.int()
        # getting output values
        self.log('Test loss', loss) #, prog_bar=True)
        for name, fun in self.logs.items():
            self.log('Test '+name, fun(y_hat_sig, y))
        for name, fun in self.metrics.items():
            self.log('Test '+name, fun(y_hat_sig, y))

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        # predicting
        y_hat_sig = F.sigmoid(y_hat)#.detach().cpu() # todo?
        threshold = torch.tensor([0.5], device = self.device)
        y_hat_sig = (y_hat_sig>threshold).float()*1
        y_hat_sig = y_hat_sig.int()
        y_target = y.int()
        
        if batch_idx == 0:
        
            batch_size = len(x)

            for k in range(batch_size):
                plt.subplot(3, batch_size, k+1)
                plt.imshow(np.rollaxis(x[k].detach().cpu().numpy(), 0, 3), cmap='gray')
                plt.title('Real')
                plt.axis('off')

                plt.subplot(3, batch_size, k+1+batch_size)
                plt.imshow(y_hat_sig[k, 0].detach().cpu().numpy(), cmap='gray')
                plt.title('Output')
                plt.axis('off')
				
                plt.subplot(3, batch_size, k+1+2*batch_size)                
                plt.imshow(y_target[k, 0].detach().cpu().numpy().transpose(1,2,0), cmap='gray')
                plt.title('Label')
                plt.axis('off')

            plt.savefig(f"{self.args.log_path}/{self.args.experiment_name}/{self.args.model_name}_fold{self.fold}/prediction.png")

        
        if self.out:
            for k in range(len(x)):
                fig = plt.figure()
                plt.imshow(y_hat_sig[k, 0].detach().cpu().numpy(), cmap='gray')
                plt.savefig(f"{self.args.log_path}/{self.args.experiment_name}/{self.args.model_name}_fold{self.fold}/prediction_mask{k+self.offset}.png")
                plt.close(fig)

                fig = plt.figure()
                plt.imshow(np.rollaxis(x[k].detach().cpu().numpy(), 0, 3), cmap='gray')
                plt.savefig(f"{self.args.log_path}/{self.args.experiment_name}/{self.args.model_name}_fold{self.fold}/prediction_img{k+self.offset}.png", bbox_inches='tight')
                plt.close(fig)
            self.offset += len(x)

        return y_hat_sig  


class EfficientNet(pl.LightningModule):
    def __init__(self, args):
        super(EfficientNet, self).__init__()

        self.args = args
        self.lr = args.lr
        
        # Load model
        self.network = timm.create_model(args.network_name, pretrained=True, num_classes=2)
        if args.percentage_to_freeze != -1.0:
            self.freeze_parameters(args.percentage_to_freeze)

        # Define metrics and loss criterion
        self.criterion = nn.BCEWithLogitsLoss()
        self.accuracy = BinaryAccuracy().to(self.device)

        # Set up logging option
        self.model_checkpoint = ModelCheckpoint(
            monitor = "val_loss",
            verbose = args.verbose,
            filename = "{epoch}_{val_loss:.4f}",
        )
        # Setup initial learning rate finder
        self.lr_finder = LearningRateFinder(
            min_lr=args.min_lr,
            max_lr=args.max_lr,
            num_training_steps=args.initial_lr_steps,
        )

    def freeze_parameters(self, percentage_to_freeze):
        # Freeze weights
        if percentage_to_freeze is None:
            print(f"Freezing classification layer ! ")
            for param in self.network.parameters():
                param.requires_grad = False
            
            # Require gradient for classification layer
            self.network.classifier.requires_grad_()

        else:
            total_params = sum(p.numel() for p in self.network.parameters())  # Count total parameters
            params_to_freeze = int(percentage_to_freeze * total_params)  # Calculate number of parameters to freeze

            frozen_params = 0
            non_frozen_params = 0
            for param in self.network.parameters():
                if frozen_params < params_to_freeze:
                    param.requires_grad = False  # Freeze the parameter
                    frozen_params += param.numel()  # Update the count of frozen parameters
                else:
                    non_frozen_params += param.numel()

            print(f"Froze {frozen_params}/{frozen_params + non_frozen_params} = {frozen_params / (frozen_params + non_frozen_params)}%")

    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        return self.optimizer(self.args, self.network)
        
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
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True),
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