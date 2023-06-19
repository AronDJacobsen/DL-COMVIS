import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateFinder
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import timm
from torchmetrics.classification import Accuracy
from torchvision.ops import box_iou, nms
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from collections import Counter, nms

from src.utils import accuracy, IoU, plot_SS

def get_model(model_name, args, loss_fun, optimizer, out=False, num_classes=2, region_size=(512,512), id2cat=None):
    if model_name == 'testnet':
        return TestNet(args, loss_fun, optimizer, out=out, num_classes=num_classes, region_size=region_size, id2cat=id2cat)
    elif model_name == 'efficientnet_b4':
        return EfficientNet(args, loss_fun, optimizer, out=out, num_classes=num_classes, region_size=region_size, id2cat=id2cat)
    else:
        raise ValueError('unknown model name')



### BASEMODEL ###
class BaseModel(pl.LightningModule):
    '''
    Contains all recurring functionality
    '''
    def __init__(self, args, loss_fun, optimizer, out, num_classes, id2cat):
        super().__init__()
        self.args = args
        self.lr = self.args.lr
        self.loss_fun = loss_fun
        self.optimizer = optimizer
        self.out = out
        self.offset = 0
        self.num_classes = num_classes
        self.iou_threshold = .5 # TODO: appropriate???
        self.mAP = MeanAveragePrecision()
        self.id2cat = id2cat
        
        # checkpointing and logging
        self.model_checkpoint = ModelCheckpoint(
            monitor = "mAP/val",
            verbose = args.verbose,
            filename = "{epoch}_{val_loss:.4f}",
        )
        
        self.save_hyperparameters(ignore=['loss_fun'])

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr = self.args.lr)
        
    def compare_boxes(self, bboxes, cat_ids, pred_bboxes, num_classes):
        # initializing
        num_gt_boxes, num_pred_boxes    = bboxes.shape[0], pred_bboxes.shape[0]
        gt_matches                      = torch.zeros(num_gt_boxes, dtype=torch.bool)
        pred_matches                    = torch.zeros(num_pred_boxes, dtype=torch.bool)

        # Mark no match as background index (which is num_classes - 1 as defined in data loader)
        pred_boxes  = (num_classes - 1) * torch.ones(num_pred_boxes, dtype=torch.long) 
        # Get IoU matrix
        iou         = box_iou(bboxes, pred_bboxes)

        for pred_idx in range(num_pred_boxes):
            # rows are true, columns are how they compare to each estimates
            iou_score = iou[:, pred_idx] # get row

            # check if there are any matches between pred_box and gt_bboxes
            max_iou = torch.max(iou_score)
            if max_iou >= self.iou_threshold:
                # Get index of max score
                gt_idx = torch.argmax(iou_score)
                # Store matches
                gt_matches[gt_idx] = True
                pred_matches[pred_idx] = True
                # for finding the box later on
                pred_boxes[pred_idx] = cat_ids[gt_idx][0]

        return pred_matches, gt_matches, pred_boxes

    def training_step(self, batch, batch_idx):
        # extract input
        loss, acc = 0, 0

        # for each image
        for (img, cat_ids, bboxes_data, pred_bboxes_data) in batch:
            # for each bounding box
            (bboxes, regions)           = bboxes_data
            (pred_bboxes, pred_regions) = pred_bboxes_data

            # find corresponding gt box
            pred_matches, gt_matches, pred_labels = self.compare_boxes(bboxes, cat_ids, pred_bboxes, self.num_classes)
            
            # Downsample background to 25% non-background vs 75% background
            non_background      = pred_labels != (self.num_classes - 1) 
            n_non_background    = non_background.sum().item()
            n_background_sample = (n_non_background + len(regions)) * 3
            # Get subset background idxs
            background_idxs     = np.random.permutation(np.arange(len(pred_labels))[pred_labels == (self.num_classes - 1)])[:n_background_sample]

            # Filter data to subset
            pred_bboxes         = torch.concat([pred_bboxes[non_background], pred_bboxes[background_idxs]])
            pred_labels         = torch.concat([pred_labels[non_background], pred_labels[background_idxs]])
            pred_regions        = torch.concat([pred_regions[non_background], pred_regions[background_idxs]])
            
            all_regions         = torch.concat([pred_regions, regions])            
            all_labels          = torch.concat([pred_labels.to(self.device), cat_ids.flatten().to(self.device)])

            # Classify proposed regions
            y_hat = self.forward(all_regions)

            # Encode data and compute loss
            one_hot_cat_pred    = torch.nn.functional.one_hot(all_labels, num_classes=self.num_classes).to(torch.float)
            loss               += self.loss_fun(y_hat, one_hot_cat_pred)
            acc                += (y_hat.detach().cpu().argmax(dim=1) == all_labels.detach().cpu()).to(torch.float).mean().item()

        loss /= len(batch)
        acc /= len(batch)

        # Log performance
        self.log('loss/train_step',  loss, batch_size=len(batch), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('loss/train_epoch', loss, batch_size=len(batch), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('acc/train_step',  acc, batch_size=len(batch), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('acc/train_epoch', acc, batch_size=len(batch), on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        # extract input
        loss, acc, IoU, mAP = 0, 0, 0, 0
        y_hat = []
        # for each image
        for (img, cat_id, bboxes_data, pred_bboxes_data) in batch:
            # for each bounding box
            (bboxes, regions)           = bboxes_data
            (pred_bboxes, pred_regions) = pred_bboxes_data

            # Classify proposed regions
            y_hat = self.forward(pred_regions)

            # maximum probabilities
            outputs = torch.nn.functional.softmax(y_hat, dim=1)
            pred_prob, pred_cat = torch.max(outputs, 1)
            # Applying NMS (remove redundant boxes)
            keep_indices = nms(pred_bboxes.to(torch.float), pred_prob, self.iou_threshold)
            # Computing AP
            preds = [{'boxes':  pred_bboxes[keep_indices], 
                      'scores': pred_prob[keep_indices], 
                      'labels': pred_cat[keep_indices]}]
            
            targets = [{'boxes':  bboxes, 
                        'labels': cat_id.flatten()}]
            # update mAP class
            self.mAP.update(preds, targets)
            # calculate
            map += self.mAP.compute()['map_50']

        # Compute performance
        IoU = 1. # TODO: change
        mAP = 1. # TODO: change

            
            # maximum probabilities
            outputs = torch.nn.functional.softmax(y_hat, dim=1)
            pred_prob, pred_cat = torch.max(outputs, 1)
            # Applying NMS (remove redundant boxes)
            keep_indices = nms(pred_bboxes.to(torch.float), pred_prob, self.iou_threshold)
            # Computing AP
            preds = [{'boxes': pred_bboxes[keep_indices], 'scores':pred_prob[keep_indices], 'labels':pred_cat[keep_indices]}]
            targets = [{'boxes':bboxes, 'labels':cat_id.flatten()}]
            # update mAP class
            self.mAP.update(preds, targets)
            # calculate
            mAP += self.mAP.compute()['map_50']

    
        # Normalize
        IoU /= len(batch)
        mAP /= len(batch)
        # Log performance
        self.log('IoU/val', IoU, batch_size=len(batch), prog_bar=True, logger=True)
        self.log('mAP/val', mAP, batch_size=len(batch), prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        # extract input
        loss, acc = 0, 0

        # for each image
        for (img, cat_id, bboxes_data, pred_bboxes_data) in batch:
            # for each bounding box
            (bboxes, regions)           = bboxes_data
            (pred_bboxes, pred_regions) = pred_bboxes_data

            # Classify proposed regions
            y_hat = self.forward(pred_regions)

        # Compute performance
        IoU = 1. # TODO: change
        mAP = 1. # TODO: change

        # Log performance
        self.log('IoU/val', IoU, batch_size=len(batch), prog_bar=True, logger=True)
        self.log('mAP/val', mAP, batch_size=len(batch), prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx):

        # for each image
        for i, (img, cat_id, bboxes_data, pred_bboxes_data) in enumerate(batch):
            # for each bounding box
            (bboxes, regions)           = bboxes_data # - not available at this point
            (pred_bboxes, pred_regions) = pred_bboxes_data

            # Classify proposed regions
            y_hat = self.forward(pred_regions)

            # maximum probabilities
            outputs = torch.nn.functional.softmax(y_hat, dim=1)
            pred_prob, pred_cat = torch.max(outputs, 1)

            print("pred_cat:", pred_cat)

            # Applying NMS (remove redundant boxes)
            keep_indices = nms(pred_bboxes.to(torch.float), pred_prob, 0.5)

            # Computing AP
            preds = {'boxes': pred_bboxes[keep_indices][pred_cat[keep_indices] != max(self.id2cat.keys())], 
                    'scores': pred_prob[keep_indices][pred_cat[keep_indices] != max(self.id2cat.keys())], 
                    'labels': pred_cat[keep_indices][pred_cat[keep_indices] != max(self.id2cat.keys())]} 
            
            targets = {'boxes':  bboxes, 
                        'labels': cat_id.flatten()}

            plot_SS(img, targets['boxes'], targets['labels'], preds['boxes'], preds['labels'], preds['scores'], i, batch_idx, self.id2cat)

class TestNet(BaseModel):
    def __init__(self, args, loss_fun, optimizer, out, num_classes, region_size, id2cat):
        super().__init__(args, loss_fun, optimizer, out, num_classes, id2cat)
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
    def __init__(self, args, loss_fun, optimizer, out, num_classes, region_size, id2cat):
        super().__init__(args, loss_fun, optimizer, out, num_classes, id2cat)


        # Load model
        self.network = timm.create_model(args.model_name, pretrained=True, num_classes=self.num_classes)
        # num_classes for 28 categories + 1 background
        if args.percentage_to_freeze != -1.0:
            self.freeze_parameters(args.percentage_to_freeze)

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