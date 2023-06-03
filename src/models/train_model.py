#!/usr/bin/env python3

import lightning.pytorch as pl
import torch

from torchvision.datasets import MNIST

import os
import sys

from src.utils import set_seed
from src.data.datasets import get_data
from src.models.models import get_model
from src.data.project1.dataloader import get_loaders 


#sys.path.insert(0, os.getcwd()+os.sep+'src')
if __name__ == '__main__':
    # arguments
    model_name = 'transfer' # 'initial'
    loss_function = 'BCE' # 'NLL'# []'CrossEntropy', 'NLL']
    # data_name = 'CIFAR10'
    data_path = '/dtu/datasets1/02514/hotdog_nothotdog'
    batch_size = 64
    epochs = 10
    seed = 42
    set_seed(seed)
    # load model
    model = get_model(model_name=model_name, loss_function=loss_function)
    # train_loader, test_loader = get_data(data_name=data_name, batch_size=batch_size)
    loaders = get_loaders('/dtu/datasets1/02514/hotdog_nothotdog', batch_size=batch_size, seed=seed)

    # train model
    trainer = pl.Trainer(devices = 1, 
                         accelerator = 'gpu' if torch.cuda.is_available() else 'cpu', 
                         max_epochs = epochs, 
                         callbacks = [model.model_checkpoint]) 

    trainer.fit(model=model,
                train_dataloaders = loaders['train'],
                val_dataloaders = loaders['validation']) 

    #manually you can save best checkpoints - 
    trainer.save_checkpoint("hotdog_efficient_net.ckpt")

    # todo: predict etc