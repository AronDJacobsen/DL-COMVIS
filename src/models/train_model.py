#!/usr/bin/env python3

import lightning.pytorch as pl

from torchvision.datasets import MNIST

import os
import sys

from src.data.datasets import get_data
from src.models.models import get_model


#sys.path.insert(0, os.getcwd()+os.sep+'src')




if __name__ == '__main__':
    # arguments
    model_name = 'initial' # transfer
    loss_function = 'NLL'# []'CrossEntropy', 'NLL']
    data_name = 'CIFAR10'
    batch_size = 64
    epochs = 10
    # 
    model = get_model(model_name=model_name, loss_function=loss_function)
    train_loader, test_loader = get_data(data_name=data_name, batch_size=batch_size)

    # train model
    trainer = pl.Trainer(max_epochs=epochs)
    trainer.fit(model=model, train_dataloaders=train_loader)

    # todo: predict etc