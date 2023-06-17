import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateFinder
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import timm
from torchmetrics.classification import Accuracy
from torchvision.ops import box_iou


from src.utils import accuracy, IoU


def get_model(model_name, args, loss_fun, optimizer, out=False, num_classes=2, region_size=(512,512)):
    if model_name == 'testnet':
        return TestNet(args, loss_fun, optimizer, out=out, num_classes=num_classes, region_size=region_size)
    elif model_name == 'efficientnet_b4':
        return EfficientNet(args, loss_fun, optimizer, out=out, num_classes=num_classes, region_size=region_size)
    else:
        raise ValueError('unknown model name')



### BASEMODEL ###
class BaseModel(pl.LightningModule):
    '''
    Contains all recurring functionality
    '''
    def __init__(self, args, loss_fun, optimizer, out, num_classes):
        super().__init__()
        self.args = args
        self.lr = self.args.lr
        self.loss_fun = loss_fun
        self.optimizer = optimizer
        self.out = out
        self.offset = 0
        self.num_classes = num_classes
        self.iou_threshold = .5 # TODO: appropriate???
        # what to log in training and validation
        self.logs = {
        #    'acc': accuracy,
            }
        # what to calculate when predicting
        self.metrics = {
        #    'IoU'         : IoU,
            }
        self.log_dict = {}

        
        # checkpointing and logging
        self.model_checkpoint = ModelCheckpoint(
            monitor = "val_loss",
            verbose = args.verbose,
            filename = "{epoch}_{val_loss:.4f}",
        )
        
        self.save_hyperparameters(ignore=['loss_fun'])

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr = self.args.lr)
        
    # TODO: make a general step, for e.g. the first part of all setps
    def general_step(self, batch, batch_idx):
        y_hat = 0
        loss = 0
        return y_hat, loss
    
    def compare_boxes(self, bboxes, pred_bboxes):
        # initializing
        num_gt_boxes, num_pred_boxes = bboxes.shape[0], pred_bboxes.shape[0]
        gt_matches = torch.zeros(num_gt_boxes, dtype=torch.bool)
        pred_matches = torch.zeros(num_pred_boxes, dtype=torch.bool)
        # no match is marked -1
        pred_boxes = -torch.ones(num_pred_boxes, dtype=torch.long)
        # for each predicted box
        iou = box_iou(bboxes, pred_bboxes)
        for pred_idx in range(num_pred_boxes):
            # rows are true, columns are how they compare to each estimates
            iou_score = iou[pred_idx] # get row
            # check if there are any matching
            max_iou = torch.max(iou_score)
            if max_iou < self.iou_threshold:
                break
            # get
            gt_idx = torch.argmax(iou_score)
            # for mAP?
            gt_matches[gt_idx] = True
            pred_matches[pred_idx] = True
            # for finding the box
            pred_boxes[pred_idx] = int(gt_idx)


        return pred_matches, gt_matches, pred_boxes



    def training_step(self, batch, batch_idx):
        
        # extract input
        loss = 0
        y = torch.nn.functional.one_hot(y, num_classes=self.num_classes) 
        y = y.to(torch.float32)
        # for each image
        for (img, cat_id, bboxes_data, pred_bboxes_data) in batch:
            # for each bounding box
            (bboxes, regions) = bboxes_data
            (pred_bboxes, pred_regions) = pred_bboxes_data
            # bbox class prediction
            y_hat = self.forward(pred_regions)
            # find corresponding gt box
            pred_matches, gt_matches, pred_boxes = self.compare_boxes(bboxes, pred_bboxes)
            # gather data
            mask = pred_boxes >= 0 # masking the bboxes similar to gt
            bbox_class_pred = y_hat[mask].to(torch.float) # predicted logits
            cat_pred = torch.take(cat_id, pred_boxes[mask]) # corresponding categories
            one_hot_cat_pred = torch.nn.functional.one_hot(cat_pred, num_classes=self.num_classes).to(torch.float)
            loss += self.loss_fun(bbox_class_pred, one_hot_cat_pred)
        
        y = torch.nn.functional.one_hot(y, num_classes=self.num_classes) 
        y = y.to(torch.float32)

        # Get prediction, loss and accuracy
        loss = self.loss_fun(y_hat, y)
        # log
        for name, fun in self.logs.items():
            self.log('train_'+name, fun(y_hat, y), prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # extract input
        loss = 0
        # for each image
        for (img, cat_id, bboxes_data, pred_bboxes_data) in batch:
            # for each bounding box
            (bboxes, regions) = bboxes_data
            (pred_bboxes, pred_regions) = pred_bboxes_data
            # bbox class prediction
            y_hat = self.forward(pred_regions)
            # find corresponding gt box
            pred_matches, gt_matches, pred_boxes = self.compare_boxes(bboxes, pred_bboxes)
            # gather data
            mask = pred_boxes >= 0 # masking the bboxes similar to gt
            bbox_class_pred = y_hat[mask].to(torch.float) # predicted logits
            cat_pred = torch.take(cat_id, pred_boxes[mask]) # corresponding categories
            one_hot_cat_pred = torch.nn.functional.one_hot(cat_pred, num_classes=self.num_classes).to(torch.float)
            loss += self.loss_fun(bbox_class_pred, one_hot_cat_pred)
        
        # TODO:
        # log
        for name, fun in self.logs.items():
            self.log('val_'+name, fun(y_hat, y), prog_bar=True, logger=True)
        for name, fun in self.metrics.items():
            self.log('val_'+name, fun(y_hat, y))
        self.log("val_loss", loss, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx): 
        
        # extract input
        loss = 0
        y = torch.nn.functional.one_hot(y, num_classes=self.num_classes) 
        y = y.to(torch.float32)
        # for each image
        for (img, cat_id, bboxes_data, pred_bboxes_data) in batch:
            # for each bounding box
            (bboxes, regions) = bboxes_data
            (pred_bboxes, pred_regions) = pred_bboxes_data
            # bbox class prediction
            y_hat = self.forward(pred_regions)
            # find corresponding gt box
            pred_matches, gt_matches, pred_boxes = self.compare_boxes(bboxes, pred_bboxes)
            # gather data
            mask = pred_boxes >= 0 # masking the bboxes similar to gt
            bbox_class_pred = y_hat[mask].to(torch.float) # predicted logits
            cat_pred = torch.take(cat_id, pred_boxes[mask]) # corresponding categories
            one_hot_cat_pred = torch.nn.functional.one_hot(cat_pred, num_classes=self.num_classes).to(torch.float)
            loss += self.loss_fun(bbox_class_pred, one_hot_cat_pred)
        


        y = torch.nn.functional.one_hot(y, num_classes=self.num_classes) 
        y = y.to(torch.float32)
        loss = self.loss_fun(y, y_hat)
        # predicting
        # getting output values
        self.log('Test loss', loss) #, prog_bar=True)
        for name, fun in self.logs.items():
            self.log('Test '+name, fun(y_hat, y))
        for name, fun in self.metrics.items():
            self.log('Test '+name, fun(y_hat, y))

    def predict_step(self, batch, batch_idx):
        x, y = batch
        # predicting
        y_hat = self.forward(x)
        
        if batch_idx == 0:
        
            batch_size = len(x)

            for k in range(batch_size):
                plt.subplot(3, batch_size, k+1)
                plt.imshow(np.rollaxis(x[k].detach().cpu().numpy(), 0, 3), cmap='gray')
                plt.title('Real')
                plt.axis('off')

                plt.subplot(3, batch_size, k+1+batch_size)
                plt.imshow(y_hat[k, 0].detach().cpu().numpy(), cmap='gray')
                plt.title('Output')
                plt.axis('off')
				
                plt.subplot(3, batch_size, k+1+2*batch_size)                
                plt.imshow(y[k, 0].detach().cpu().numpy().transpose(1,2,0), cmap='gray')
                plt.title('Label')
                plt.axis('off')

            plt.savefig(f"{self.args.log_path}/{self.args.experiment_name}/{self.args.model_name}/prediction.png")

        
        if self.out:
            for k in range(len(x)):
                fig = plt.figure()
                plt.imshow(y_hat[k, 0].detach().cpu().numpy(), cmap='gray')
                plt.savefig(f"{self.args.log_path}/{self.args.experiment_name}/{self.args.model_name}/prediction_mask{k+self.offset}.png")
                plt.close(fig)

                fig = plt.figure()
                plt.imshow(np.rollaxis(x[k].detach().cpu().numpy(), 0, 3), cmap='gray')
                plt.savefig(f"{self.args.log_path}/{self.args.experiment_name}/{self.args.model_name}/prediction_img{k+self.offset}.png", bbox_inches='tight')
                plt.close(fig)
            self.offset += len(x)


class TestNet(BaseModel):
    def __init__(self, args, loss_fun, optimizer, out, num_classes, region_size):
        super().__init__(args, loss_fun, optimizer, out, num_classes)
        h, w = region_size
        self.fc1 = nn.Linear(h*w*3, 128)  # 5*5 from image dimension
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        #self.softmax = nn.Softmax()


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        #x = self.softmax(x)
        return x



### OLD ### have to adjust and remove the train test val steps as they are in the base model


class EfficientNet(BaseModel):
    def __init__(self, args, loss_fun, optimizer, out, num_classes, region_size):
        super().__init__(args, loss_fun, optimizer, out, num_classes)

        self.args = args
        self.lr = args.lr
        
        # Load model
        self.network = timm.create_model(args.model_name, pretrained=True, num_classes=self.num_classes)
        # num_classes for 28 categories + 1 background
        if args.percentage_to_freeze != -1.0:
            self.freeze_parameters(args.percentage_to_freeze)

        # Define metrics and loss criterion
        self.loss_fun = loss_fun
        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes).to(self.device)

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

        
    def training_step(self, batch, batch_idx):
        # Extract and process input
        x, y = batch
        y = torch.nn.functional.one_hot(y, num_classes=self.num_classes).squeeze(1)
        y = y.to(torch.float32)

        # Get prediction, loss and accuracy
        y_hat = self(x)
        loss = self.loss_fun(y_hat, y)
        acc = self.accuracy(y_hat, y)

        # logs metrics for each training_step - [default:True],
        # the average across the epoch, to the progress bar and logger-[default:False]
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True),
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Extract and process input
        x, y = batch
        y = torch.nn.functional.one_hot(y, num_classes=self.num_classes).squeeze(1)
        y = y.to(torch.float32) 

        # Get prediction, loss and accuracy
        y_hat = self(x)
        loss = self.loss_fun(y_hat, y)
        acc = self.accuracy(y_hat, y)

        # logs metrics for each validation_step - [default:False]
        #the average across the epoch - [default:True]
        self.log("val_acc", acc, prog_bar=True, logger=True),
        self.log("val_loss", loss, prog_bar=True, logger=True)
