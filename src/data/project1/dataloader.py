from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from src.utils import set_seed

class HotdogDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)
    
def get_normalization_constants(root: str, seed: int = 0):
    # Set seed for split control
    set_seed(seed)

    # Define transforms (only resize as we want to compute means...)
    normalization_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Get trainset
    trainset     = ImageFolder(f'{root}/train', transform=normalization_transform)

    # Compute means and standard deviations from training set
    train_mean = torch.stack([t.mean(1).mean(1) for t, c in tqdm(trainset, desc='Computing mean of training split...')]).mean(0)
    train_std  = torch.stack([t.std(1).std(1) for t, c in tqdm(trainset, desc='Computing std. dev. of training split...')]).std(0)
    print(f"\nMean: {train_mean}\nStd. dev.: {train_std}")    
    return train_mean, train_std

def get_loaders(
        root: str = '/dtu/datasets1/02514/hotdog_nohotdog', 
        batch_size: int = 64, seed: int = 0, 
        train_transforms=None, test_transforms=None, 
        num_workers=1,
    ) -> dict:

    # Set seed for split control
    set_seed(seed)

    # Load images as datasets
    trainset = ImageFolder(f'{root}/train', transform=train_transforms)
    testvalset     = ImageFolder(f'{root}/test') 

    # Get validation set size
    N_testval  = testvalset.__len__()                                       # total test points
    N_val       = int(0.2 * N_testval)                                      # take ~20% for validation

    # Split trainval dataset into train- and valset
    order = np.random.permutation(np.arange(N_testval))
    val_subset      = torch.utils.data.Subset(testvalset, order[:N_val])         
    test_subset    = torch.utils.data.Subset(testvalset, order[N_val:])     

    valset = HotdogDataset(val_subset, transform=test_transforms)
    testset = HotdogDataset(test_subset, transform=test_transforms)

    # Get dataloaders
    trainloader = DataLoader(trainset,  batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valloader   = DataLoader(valset,    batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader  = DataLoader(testset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Return loaders in dictionary
    return {'train': trainloader, 'validation': valloader, 'test': testloader, 'test_idxs': order[N_val:]}
