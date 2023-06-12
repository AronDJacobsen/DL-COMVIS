
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss(loss_functions, regularization, regularization_coef):

    # LOSS
    if loss_functions == 'BCE':
        loss_fun = bce_loss
    elif loss_functions == 'FOCAL':
        loss_fun =  focal_loss
    elif loss_functions == 'DICE':
        loss_fun =  dice_loss
    else:
        raise ValueError('unknown loss function')

    # REGULARIZATION
    if regularization != 'none':
        regularization_fun = get_regularization(regularization)
        return lambda y, y_hat: loss_fun(y, y_hat) + regularization_coef * regularization_fun(y_hat)
    else:
        return loss_fun

# LOSS FUNCTIONS
def bce_loss(y, y_hat):
    return F.binary_cross_entropy_with_logits(y_hat, y)

def focal_loss(y, y_hat):
    gamma = 2
    conf = F.sigmoid(y_hat)
    return - torch.sum( (1-conf)**gamma * y * torch.log(conf) + (1-y) * torch.log(1-conf) )

def dice_loss(y, y_hat):
    # Andreas
    smooth = 1
    y_hat = F.sigmoid(y_hat)

    y_hat = y_hat.view(-1)
    y = y.view(-1)

    intersection = (y_hat * y).sum()
    dice = (2.*intersection + smooth)/(y_hat.sum() + y.sum() + smooth)

    return 1 - dice

def dice_loss_phil(y_real, y_pred):
    # TODO: change variable names

    X, Y = y_real, y_pred
    # Found here instead https://gist.github.com/weiliu620/52d140b22685cf9552da4899e2160183
    numerator = 2 * torch.sum(X * Y)
    denominator = torch.sum(X + Y)
    return 1 - (numerator + 1) / (denominator + 1)




# REGULARIZATION FUNCTIONS
def get_regularization(reg):
    if reg == 'centered':
        return centered
    elif reg == 'sparsity':
        return sparsity
    elif reg == 'tv':
        return total_variation

    else:
        raise ValueError('unknown regularization function')


def centered(y_hat):
    _, _, h, w = y_hat.shape
    return (1-F.sigmoid(y_hat[:,:,int(h/2), int(w/2)])).sum()


def sparsity(y_hat):
    return F.sigmoid(y_hat).sum()


def total_variation(y_hat):
    return (torch.sum(torch.abs(F.sigmoid(y_hat[1:,:]) - F.sigmoid(y_hat[:-1,:]))) + torch.sum(torch.abs(F.sigmoid(y_hat[:,1:]) - F.sigmoid(y_hat[:,:-1]))))


