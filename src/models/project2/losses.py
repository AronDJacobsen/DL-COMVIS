
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss(loss_functions):
    if loss_functions == 'BCE':
        return bce_loss
    elif loss_functions == 'BCE_total_variation':
        return bce_total_variation
    elif loss_functions == 'FOCAL':
        return focal_loss
    elif loss_functions == 'DICE':
        return dice_loss
    else:
        raise ValueError('unknown loss function')

# define parameters
def bce_loss(y_real, y_pred):
    return F.binary_cross_entropy_with_logits(y_pred, y_real)

def bce_total_variation(y_real, y_pred):
    # yip1 = torch.roll(y_pred, 1, 2)
    # yjp1 = torch.roll(y_pred, 1, 3)
    return bce_loss(y_real, y_pred) + 0.1 * (torch.sum(torch.abs(F.sigmoid(y_pred[1:,:]) - F.sigmoid(y_pred[:-1,:]))) + torch.sum(torch.abs(F.sigmoid(y_pred[:,1:]) - F.sigmoid(y_pred[:,:-1]))))

def focal_loss(y_real, y_pred):
    gamma = 2
    conf = F.sigmoid(y_pred)
    return - torch.sum( (1-conf)**gamma * y_real * torch.log(conf) + (1-y_real) * torch.log(1-conf) )

def dice_loss(targets, inputs):
    # Andreas
    smooth = 1
    inputs = F.sigmoid(inputs)

    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()
    dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

    return 1 - dice

def dice_loss_phil(y_real, y_pred):
    X, Y = y_real, y_pred
    # Found here instead https://gist.github.com/weiliu620/52d140b22685cf9552da4899e2160183
    numerator = 2 * torch.sum(X * Y)
    denominator = torch.sum(X + Y)
    return 1 - (numerator + 1) / (denominator + 1)