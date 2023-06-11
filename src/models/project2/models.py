import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateFinder
import pytorch_lightning as pl

from src.utils import accuracy, specificity, sensitivity, iou, dice_score


def get_model(model_name, args, loss_fun, optimizer):
    if model_name == 'SegCNN':
        return SegCNN(args, loss_fun, optimizer)
    if model_name == 'UNet':
        return UNet(args, loss_fun, optimizer)



### BASEMODEL ###
class BaseModel(pl.LightningModule):
    '''
    Contains all recurring functionality
    '''
    def __init__(self, args, loss_fun, optimizer):
        super().__init__()
        self.args = args
        self.lr = self.args.lr
        self.loss_fun = loss_fun
        self.optimizer = optimizer

        # what to log in training and validation
        self.logs = {
            'acc': accuracy,
            }
        # what to calculate when predicting
        self.metrics = {
            'Specificity' : specificity,
            'Sensitivity' : sensitivity,
            'IoU'         : iou,
            'Dice score': dice_score,
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



class SegCNN(BaseModel):
    '''
    Inherits functionality from basemodel
    '''
    def __init__(self, args, loss_fun, optimizer):
        super().__init__(args, loss_fun, optimizer)
        
        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool0 = nn.MaxPool2d(2, 2)  # 256 -> 128
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 128 -> 64
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 64 -> 32
        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 32 -> 16

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # decoder (upsampling)
        self.upsample0 = nn.Upsample(scale_factor=2)  # 16 -> 32
        self.dec_conv0 = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2)  # 32 -> 64
        self.dec_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2)  # 64 -> 128
        self.dec_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=2)  # 128 -> 256
        self.dec_conv3 = nn.Conv2d(64, 1, 3, padding=1)
    
    def forward(self, x):
        # encoder
        e0 = self.pool0(F.relu(self.enc_conv0(x)))
        e1 = self.pool1(F.relu(self.enc_conv1(e0)))
        e2 = self.pool2(F.relu(self.enc_conv2(e1)))
        e3 = self.pool3(F.relu(self.enc_conv3(e2)))

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3))

        # decoder
        d0 = F.relu(self.dec_conv0(self.upsample0(b)))
        d1 = F.relu(self.dec_conv1(self.upsample1(d0)))
        d2 = F.relu(self.dec_conv2(self.upsample2(d1)))
        d3 = self.dec_conv3(self.upsample3(d2))  # no activation
        return d3


class UNet(BaseModel):
    def __init__(self, args, loss_fun, optimizer):
        super().__init__(args, loss_fun, optimizer)
    

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool0 = nn.Conv2d(64, 64, 3, padding=1, stride=2) # nn.MaxPool2d(2, 2)  # 128 -> 64

        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.Conv2d(64, 64, 3, padding=1, stride=2) # nn.MaxPool2d(2, 2)  # 64 -> 32

        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.Conv2d(64, 64, 3, padding=1, stride=2) # nn.MaxPool2d(2, 2)  # 32 -> 16

        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool3 = nn.Conv2d(64, 64, 3, padding=1, stride=2) # nn.MaxPool2d(2, 2)  # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # decoder (upsampling)
        self.upsample0 = nn.ConvTranspose2d(64, 64, 3, padding=1, stride=2, output_padding=1) #nn.Upsample(16)  # 8 -> 16
        self.dec_conv0 = nn.Conv2d(128, 64, 3, padding=1)

        self.upsample1 = nn.ConvTranspose2d(64, 64, 3, padding=1, stride=2, output_padding=1) # nn.Upsample(32)  # 16 -> 32
        self.dec_conv1 = nn.Conv2d(128, 64, 3, padding=1)

        self.upsample2 = nn.ConvTranspose2d(64, 64, 3, padding=1, stride=2, output_padding=1) #nn.Upsample(64)  # 32 -> 64
        self.dec_conv2 = nn.Conv2d(128, 64, 3, padding=1)
        
        self.upsample3 = nn.ConvTranspose2d(64, 64, 3, padding=1, stride=2, output_padding=1) # nn.Upsample(128)  # 64 -> 128
        self.dec_conv3 = nn.Conv2d(128, 1, 3, padding=1)

    def forward(self, x):
        # encoder
        e0 = F.relu(self.enc_conv0(x))
        e0_pool = self.pool0(e0)
        e1 = F.relu(self.enc_conv1(e0_pool))
        e1_pool = self.pool1(e1)
        e2 = F.relu(self.enc_conv2(e1_pool))
        e2_pool = self.pool2(e2)
        e3 = F.relu(self.enc_conv3(e2_pool))
        e3_pool = self.pool3(e3)

        # print(e3_pool.shape)

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3_pool))
        # print("b", b.shape)

        # decoder - # Concatenating last encoder layer to first decode, within the relu
        d0 = F.relu(self.dec_conv0(torch.cat([self.upsample0(b), e3], dim = 1)))
        # print("d0", d0.shape)
        d1 = F.relu(self.dec_conv1(torch.cat([self.upsample1(d0), e2], dim = 1)))
        # print("d1", d1.shape)
        d2 = F.relu(self.dec_conv2(torch.cat([self.upsample2(d1), e1], dim = 1)))
        # print("d2", d2.shape)
        d3 = self.dec_conv3(torch.cat([self.upsample3(d2), e0], dim = 1))
        # I think this is correct, as the last layer should

        return d3
