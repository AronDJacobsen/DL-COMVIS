from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from ..utils import set_seed

def get_normalization_constants(root: str, seed: int = 0):
    # Set seed for split control
    set_seed(seed)

    # Define transforms (only resize as we want to compute means...)
    normalization_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    trainvalset     = ImageFolder(f'{root}/train', transform=normalization_transform)

    # Get size of validation split
    N_trainval  = trainvalset.__len__()
    N_val       = int(0.2 * N_trainval)
    
    # Get trainset
    trainset    = torch.utils.data.Subset(trainvalset, range(N_val, N_trainval))     

    # Compute means and standard deviations from training set
    train_mean = torch.stack([t.mean(1).mean(1) for t, c in tqdm(trainset, desc='Computing mean of training split...')]).mean(0)
    train_std  = torch.stack([t.std(1).std(1) for t, c in tqdm(trainset, desc='Computing std. dev. of training split...')]).std(0)
    print(f"\nMean: {train_mean}\nStd. dev.: {train_std}")    
    return train_mean, train_std

def get_loaders(root: str = '/dtu/datasets1/02514/hotdogs_nohotdogs', batch_size: int = 64, seed: int = 0) -> dict:

    train_mean, train_std = get_normalization_constants(root, seed)

    # Set seed for split control
    set_seed(seed)

    # Define transforms for training and test/validation data
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        # transforms.RandomRotation(degrees=(30, 70)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5132, 0.4369, 0.3576], # [0.5, 0.5, 0.5],
            std=[0.0214, 0.0208, 0.0223], #[0.5, 0.5, 0.5]
        )
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5132, 0.4369, 0.3576], # [0.5, 0.5, 0.5],
            std=[0.0214, 0.0208, 0.0223], # [0.5, 0.5, 0.5],
        )
    ])

    # Load images as datasets
    trainvalset = ImageFolder('/dtu/datasets1/02514/hotdog_nothotdog/train', transform=train_transforms)
    testset     = ImageFolder('/dtu/datasets1/02514/hotdog_nothotdog/test', transform=test_transforms)


    # Get validation set size
    N_trainval  = trainvalset.__len__()                                       # total training points
    N_val       = int(0.2 * N_trainval)                                       # take ~20% for validation

    # Split trainval dataset into train- and valset
    valset      = torch.utils.data.Subset(trainvalset, range(N_val))         
    trainset    = torch.utils.data.Subset(trainvalset, range(N_val, N_trainval))     

    # Change transforms of validation set
    valset.dataset.transform = test_transforms

    # Get dataloaders
    trainloader = DataLoader(trainset,  batch_size=batch_size, shuffle=True, num_workers=1)
    valloader   = DataLoader(valset,    batch_size=batch_size, shuffle=True, num_workers=1)
    testloader  = DataLoader(testset,   batch_size=batch_size, shuffle=False, num_workers=1)

    # Return loaders in dictionary
    return {'train': trainloader, 'validation': valloader, 'test': testloader}
    