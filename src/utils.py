import torch
import numpy as np

from torchvision import transforms

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

def accuracy(y_pred, y_true):
    """accuracy of segmentation wrt. ground truth mask"""
    return (y_pred == y_true).sum().item() / y_true.numel() 

def specificity(y_pred, y_true):
    """specificity of segmentation wrt. ground truth mask"""
    return ((y_pred == y_true) & (y_true == 0)).sum().item() / (y_true == 0).sum().item()

def sensitivity(y_pred, y_true):
    """sensitivity of segmentation wrt. ground truth mask"""
    return ((y_pred == y_true) & (y_true == 1)).sum().item() / (y_true == 1).sum().item()

def iou(y_pred, y_true):
    """intersection over union of segmentation wrt. ground truth mask"""
    return (y_pred & y_true).sum().item() / (y_pred | y_true).sum().item()

def dice(y_pred, y_true):
    """dice coefficient of segmentation wrt. ground truth mask"""
    return 2 * (y_pred & y_true).sum().item() / (y_pred.sum().item() + y_true.sum().item())

