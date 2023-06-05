import torch.nn as nn

def get_loss(loss_function):

    # get loss function
    if loss_function == 'CrossEntropy':
        return nn.CrossEntropyLoss()
    
    elif loss_function == 'BCE':
        return nn.BCEWithLogitsLoss()