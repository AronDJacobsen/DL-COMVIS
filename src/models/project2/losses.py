
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss(loss_functions):
    if loss_functions == 'BCE':
        return bce_loss
    else:
        raise ValueError('unknown loss function')



# define parameters
def bce_loss(y_real, y_pred):
    return F.binary_cross_entropy_with_logits(y_pred, y_real,
                                              pos_weight=torch.tensor([0.5]))





