import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np

from albumentations.pytorch import ToTensorV2
import albumentations as A

from src.utils import set_seed

def NoOp(image, **kwargs):
    return image

class WasteDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return 1
        # return len(self.subset)
        
    # def __getitem__(self, index):
    #     x, y = self.subset[index]
    #     if self.transform:
    #         x = self.transform(x)
    #     return x, y
    
    def __getitem__(self, index):
        return torch.randn(3,224,224), torch.randint(0,29,(1,))

    
def get_loaders(dataset, batch_size=64, seed=1, num_workers=1, augmentations:dict={'rotate': False, 'flip': False}) -> dict:

    if dataset == 'waste':
        root = '/dtu/datasets1/02514/data_wastedetection'
    # Set seed for split control
    set_seed(seed)
    img_size = (224, 224)
    train_transform = A.Compose([
                A.Resize(img_size[0], img_size[1]),
                A.Rotate(limit=20, p=0.5) if augmentations['rotate'] else A.Lambda(NoOp, NoOp),
                A.HorizontalFlip(p=0.5) if augmentations['flip'] else A.Lambda(NoOp, NoOp),
                ToTensorV2() # does the same as transforms.ToTensor()
            ]) #, is_check_shapes=False) 
    
    test_transform = A.Compose([
            A.Resize(img_size[0], img_size[1]),
            ToTensorV2() # does the same as transforms.ToTensor()
        ])#, is_check_shapes=False) 

    # Load images as datasets    
    train_subset = None
    val_subset = None
    test_subset = None

    trainset = WasteDataset(train_subset, transform=train_transform)
    valset = WasteDataset(val_subset, transform=test_transform)
    testset = WasteDataset(test_subset, transform=test_transform)


    trainloader = DataLoader(trainset,  batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valloader   = DataLoader(valset,    batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader  = DataLoader(testset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Return loaders in dictionary
    return {'train': trainloader, 'validation': valloader, 'test': testloader}