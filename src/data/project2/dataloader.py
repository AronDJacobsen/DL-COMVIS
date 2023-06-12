
################

from collections import deque
import sklearn
import glob
import PIL.Image as Image
import random
import os
from torchvision import transforms
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


# Get loaders function
from torch.utils.data import DataLoader

def NoOp(image, **kwargs):
    return image

def get_loaders(dataset, batch_size=2, seed=1, num_workers=1, augmentations:dict={'rotate': False, 'flip': False}):

    if dataset == 'DRIVE':
        img_size = (256, 256)

        train_transform = A.Compose([
                        A.Resize(img_size[0], img_size[1]),
                        A.Rotate(limit=20, p=0.5) if augmentations['rotate'] else A.Lambda(NoOp, NoOp),
                        A.HorizontalFlip(p=0.5) if augmentations['flip'] else A.Lambda(NoOp, NoOp),
                        ToTensorV2() # does the same as transforms.ToTensor()
                    ], is_check_shapes=False) 
        test_transform = A.Compose([
                A.Resize(img_size[0], img_size[1]),
                ToTensorV2() # does the same as transforms.ToTensor()
            ], is_check_shapes=False) 
        
        # train_transform = transforms.Compose([transforms.Resize(img_size), 
        #                                 transforms.ToTensor()])

        # testval_transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])

        return {
            fold: {
                'train': DataLoader(
                    DRIVE(mode='train', fold=fold, transform=train_transform),
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                ),
                'test': DataLoader(
                    DRIVE(mode='test', fold=fold, transform=test_transform),
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                ),
                'validation': DataLoader(
                    DRIVE(mode='val', fold=fold, transform=test_transform),
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                ),
            }
            for fold in range(20)
        }
    
    elif dataset == 'PH2':
        return _extracted_from_get_loaders_(batch_size, num_workers, augmentations)
    else:
        raise ValueError('unknown dataset')
    

# TODO Rename this here and in `get_loaders`
def _extracted_from_get_loaders_(batch_size, num_workers, augmentations):
    # won't work if halving in the CNN structure will end up with an odd number, numbers must be divisible by 2^N
    img_size = (256, 256)
    
    train_transform = A.Compose([
                        A.Resize(img_size[0], img_size[1]),
                        A.Rotate(limit=20, p=0.5) if augmentations['rotate'] else A.Lambda(NoOp, NoOp),
                        A.HorizontalFlip(p=0.5) if augmentations['flip'] else A.Lambda(NoOp, NoOp),
                        ToTensorV2() # does the same as transforms.ToTensor()
                    ], is_check_shapes=False) 
    test_transform = A.Compose([
                A.Resize(img_size[0], img_size[1]),
                ToTensorV2() # does the same as transforms.ToTensor()
            ], is_check_shapes=False) 

    trainset = PH2_dataset(mode='train', transform=train_transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    valset = PH2_dataset(mode='val', transform=test_transform)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = PH2_dataset(mode='test', transform=test_transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return {'train': train_loader, 'validation': val_loader, 'test': test_loader}



## Dataset classes - DRIVE
class DRIVE(torch.utils.data.Dataset):
    def __init__(self, mode = 'train', fold = 0, transform = transforms.ToTensor(), data_path='/dtu/datasets1/02514/DRIVE', seed=420):
        'Initialization'
        self.transform = transform
        data_path = os.path.join(data_path, 'training')
        self.image_paths = sorted(glob.glob(f'{data_path}/images/*.tif'))
        self.label_paths = sorted(glob.glob(f'{data_path}/1st_manual/*.gif'))

        # Shuffling
        self.image_paths, self.label_paths = sklearn.utils.shuffle(self.image_paths, self.label_paths, random_state=seed)

        # rolling 
        self.image_paths, self.label_paths = deque(self.image_paths), deque(self.label_paths)
        self.image_paths.rotate(fold)
        self.label_paths.rotate(fold)

        # converting to list
        self.image_paths, self.label_paths = list(self.image_paths), list(self.label_paths)

        if mode == 'val':
            self.image_paths, self.label_paths = self.image_paths[15:-1], self.label_paths[15:-1]

        elif mode == 'train':
            self.image_paths, self.label_paths = self.image_paths[:15], self.label_paths[:15]

        elif mode == 'test':
            self.image_paths, self.label_paths = self.image_paths[-1:], self.label_paths[-1:]
            
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = np.array(Image.open(image_path))/ 255 
        image = image.astype(np.float32)
        label =  np.array(Image.open(label_path)) * 1.0
        transformed = self.transform(image=image, mask=label)
        X = transformed['image']
        Y = transformed['mask'].unsqueeze(0)

        return X, Y
    
## Dataset classes - PH2
class PH2_dataset(torch.utils.data.Dataset):
    def __init__(self, mode, transform, data_path='/dtu/datasets1/02514/PH2_Dataset_images', seed=420):
        # Initialization
        #data_path = '/Users/arond.jacobsen/Desktop/DTU/8_semester/02514_Deep_Learning_in_Computer_Vision/2_part/0_project/sample_data/PH2_Dataset_images'
        
        self.transform = transform
        self.image_paths = glob.glob(f'{data_path}/*/*_Dermoscopic_Image/*.bmp')
        self.label_paths = glob.glob(f'{data_path}/*/*_lesion/*.bmp')
        c = list(zip(self.image_paths, self.label_paths))
        random.seed(seed)
        random.shuffle(c)
        self.image_paths, self.label_paths = zip(*c)        

        train_size = int(0.7 * len(self.image_paths))
        val_size = int(0.1 * len(self.image_paths))

        if mode == 'train':
            self.image_paths = self.image_paths[:train_size]
            self.label_paths = self.label_paths[:train_size]

        elif mode == 'val':
            self.image_paths = self.image_paths[train_size:train_size+val_size]
            self.label_paths = self.label_paths[train_size:train_size+val_size]

        elif mode == 'test':
            self.image_paths = self.image_paths[train_size+val_size:]
            self.label_paths = self.label_paths[train_size+val_size:]

        else:
            raise ValueError('mode must be one of train, val or test')

        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        # Albumentations
        image = np.array(Image.open(image_path)) / 255 
        image = image.astype(np.float32)
        label =  np.array(Image.open(label_path)) * 1.0
        transformed = self.transform(image=image, mask=label)
        X = transformed['image']
        Y = transformed['mask'].unsqueeze(0)
    
        return X, Y 
