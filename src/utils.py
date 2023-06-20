import torch
import numpy as np
import os

import torch.optim as optim
from torchvision import transforms

import selectivesearch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def set_seed(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)

def invertNormalization(train_mean, train_std):
    return transforms.Compose([
        transforms.Normalize(
            mean=[0., 0., 0.],
            std=[1/0.0214, 1/0.0208, 1/0.0223]
        ),
        transforms.Normalize(
            mean=[-0.5132, -0.4369, -0.3576],
            std=[1., 1., 1.]
        )
    ])


def get_optimizer(optimizer):
    if optimizer == 'Adam':
        return optim.Adam
    elif optimizer == 'SGD':
        return optim.SGD
    else:
        raise ValueError('unknown optimizer')


epsilon = 1e-7

def accuracy(y, y_hat):
    """accuracy of segmentation wrt. ground truth mask"""
    return (y_hat == y).sum().item() / (y.numel() + epsilon)

def specificity(y, y_hat):
    """specificity of segmentation wrt. ground truth mask"""
    return ((y_hat == y) & (y == 0)).sum().item() / ((y == 0).sum().item() + epsilon)

def sensitivity(y, y_hat):
    """sensitivity of segmentation wrt. ground truth mask"""
    return ((y_hat == y) & (y == 1)).sum().item() / ((y == 1).sum().item() + epsilon)

def iou(y, y_hat):
    """intersection over union of segmentation wrt. ground truth mask"""
    return (y_hat & y).sum().item() / ((y_hat | y).sum().item() + epsilon)

def dice_score(y, y_hat):
    """dice coefficient of segmentation wrt. ground truth mask"""
    return 2 * (y_hat & y).sum().item() / ((y_hat.sum().item() + y.sum().item()) + epsilon)

def IoU(y, y_hat):
    """IoU for objection detection, expects bounding boxes
    [x, y, w, h]
    """
    # for intersection area
    x1 = torch.max(y_hat[:, 0], y[:, 0])
    y1 = torch.max(y_hat[:, 1], y[:, 1])
    x2 = torch.min(y_hat[:, 0] + y_hat[:, 2], y[:, 0] + y[:, 2])
    y2 = torch.min(y_hat[:, 1] + y_hat[:, 3], y[:, 1] + y[:, 3])
    intersection = torch.clamp((x2 - x1), min=0) * torch.clamp((y2 - y1), min=0)
    # their sum minus intersection
    union = y_hat[:, 2] * y_hat[:, 3] + y[:, 2] * y[:, 3] - intersection
    return intersection / union if not np.allclose(union, 0) else 0.0



def Recall(y, y_hat):
    """IoU for objection detection, expects bounding boxes
    [x1, y1, x2, x2]
    """
    # for intersection area
    x1 = torch.max(y_hat[:, 0], y[:, 0])
    y1 = torch.max(y_hat[:, 1], y[:, 1])
    x2 = torch.min(y_hat[:, 2], y[:, 2])
    y2 = torch.min(y_hat[:, 3], y[:, 3])
    intersection = torch.clamp((x2 - x1), min=0) * torch.clamp((y2 - y1), min=0)
    # their sum minus intersection
    union = y_hat[:, 2] * y_hat[:, 3] + y[:, 2] * y[:, 3] - intersection
    gt = (y[:, 0]- y[:, 2]) * (y[:, 1]- y[:, 3])
    return intersection / gt if not np.allclose(gt, 0) else 0.0



def mAP(preds, targets):
    """mean average precision for object detection"""
    return MeanAveragePrecision()(preds, targets)

def non_maximum_suppression(y, y_hat, iou_threshold=0.5):
    """non maximum suppression for object detection"""
    iou_matrix = IoU(y, y_hat)
    iou_matrix[iou_matrix < iou_threshold] = 0
    return iou_matrix

def selective_search(transformed_img, scale=500, sigma=0.9, min_size=10):

    _, regions = selectivesearch.selective_search(transformed_img.permute(1,2,0), scale=scale, sigma=sigma, min_size=min_size)

    return set([bb['rect'] for bb in regions])
    
def plot_SS_old(transformed_img, bboxes, idx = None, batch_idx = None, path = '/work3/s194253/02514/project4_results/predict_imgs'):
    # Albumentations
    fig, ax = plt.subplots(figsize=(6, 6))

    folder_path = f"{path}/{path.split('/')[-1]}_batchidx{batch_idx}"

    print(transformed_img)
    print(bboxes)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    ax.imshow(transforms.ToPILImage()(transformed_img))
    for x1,y1,x2,y2 in bboxes.cpu().numpy():
        print(x1,y1,x2,y2)
        bbox = mpatches.Rectangle(
            (x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(bbox)
        break

    plt.axis('off')
    plt.savefig(f'{folder_path}/idx{idx}.png')
    plt.close()

def plot_SS(transformed_img, GTs, GT_labels, bboxes, bbox_labels, bbox_scores, idx = None, batch_idx = None, id2cat = None, path = '/work3/s184984/02514/project4_results/predict_imgs'):
    # Albumentations
    fig, ax = plt.subplots(figsize=(6, 6))
    folder_path = f"{path}/{path.split('/')[-1]}_batchidx{batch_idx}"

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    ax.imshow(transforms.ToPILImage()(transformed_img))
    for i, (x1,y1,x2,y2) in enumerate(bboxes.cpu().numpy()):
        bbox = mpatches.Rectangle(
            (x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(bbox)
        ax.add_artist(bbox)
        rx, ry = bbox.get_xy()
        cx = rx + bbox.get_width()/2.0
        cy = ry + bbox.get_height()/2.0

        ax.annotate(f"{id2cat[int(bbox_labels[i].cpu().numpy())]}, prob {bbox_scores[i].cpu().numpy():.2f}", (cx, cy), color='red', weight='bold', 
                    fontsize=6, ha='center', va='center')
        
    for i, (x1,y1,x2,y2) in enumerate(GTs.cpu().numpy()):
        bbox = mpatches.Rectangle(
            (x1, y1), x2-x1, y2-y1, fill=False, edgecolor='green', linewidth=1)
        ax.add_patch(bbox)
        ax.add_artist(bbox)
        rx, ry = bbox.get_xy()
        #cx = rx + bbox.get_width()/2.0
        #cy = ry + bbox.get_height()/2.0

        ax.annotate(f"{id2cat[int(GT_labels[i].cpu().numpy())]}", (rx+5, ry-5), color='green', weight='bold', 
                    fontsize=6, ha='center', va='center')        

    plt.axis('off')
    plt.savefig(f'{folder_path}/idx{idx}.png')
    plt.close()
