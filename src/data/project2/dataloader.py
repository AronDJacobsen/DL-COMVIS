
################

from collections import deque
import sklearn
import glob
import PIL.Image as Image
import random
import os
from torchvision import transforms
import torch



# Get loaders function
from torch.utils.data import DataLoader

def get_loaders(dataset, batch_size=2, seed=1, num_workers=1):

    if dataset == 'DRIVE':
        img_size = (256, 256)
        train_transform = transforms.Compose([transforms.Resize(img_size), 
                                            transforms.ToTensor()])

        testval_transform = transforms.Compose([transforms.Resize(img_size), 
                                            transforms.ToTensor()])

        # Creating loader-dicts:        loaders[fold_number]['train']
        # Remember to set seed within loader if specific seed is needed!
        loaders = {fold: {'train': DataLoader( DRIVE(mode = 'train', fold = fold, transform = train_transform  ), batch_size=batch_size, shuffle=True, num_workers=num_workers), 
                        'test' : DataLoader( DRIVE(mode = 'test',  fold = fold, transform = testval_transform), batch_size=batch_size, shuffle=True, num_workers=num_workers),
                        'validation'  : DataLoader( DRIVE(mode = 'val',   fold = fold, transform = testval_transform), batch_size=batch_size, shuffle=True, num_workers=num_workers)} 
                for fold in range(20) }

        return loaders
        
    if dataset == 'PH2':

        # won't work if halving in the CNN structure will end up with an uneven number
        img_size = (288, 384)
        train_transform = transforms.Compose([transforms.Resize(img_size), 
                                            transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.Resize(img_size), 
                                            transforms.ToTensor()])

        trainset = PH2_dataset(mode='train', transform=train_transform)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        valset = PH2_dataset(mode='val', transform=train_transform)
        val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        testset = PH2_dataset(mode='test', transform=train_transform)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
        loaders = {'train': train_loader, 'validation': val_loader, 'test': test_loader}
    
        return loaders









## Dataset classes - DRIVE
class DRIVE(torch.utils.data.Dataset):
    def __init__(self, mode = 'train', fold = 0, transform = transforms.ToTensor(), data_path='/dtu/datasets1/02514/DRIVE', seed=420):
        'Initialization'
        self.transform = transform
        data_path = os.path.join(data_path, 'training')
        self.image_paths = sorted(glob.glob(data_path + '/images/*.tif'))
        self.label_paths = sorted(glob.glob(data_path + '/1st_manual/*.gif'))

        # Shuffling
        self.image_paths, self.label_paths = sklearn.utils.shuffle(self.image_paths, self.label_paths, random_state=seed)

        # rolling 
        self.image_paths, self.label_paths = deque(self.image_paths), deque(self.label_paths)
        self.image_paths.rotate(fold)
        self.label_paths.rotate(fold)
        
        # converting to list
        self.image_paths, self.label_paths = list(self.image_paths), list(self.label_paths)
        
        if mode == 'train':
            self.image_paths, self.label_paths = self.image_paths[:14], self.label_paths[:14]

        elif mode == 'test':
            self.image_paths, self.label_paths = self.image_paths[14:-2], self.label_paths[14:-2]

        elif mode == 'val':
            self.image_paths, self.label_paths = self.image_paths[-2:], self.label_paths[-2:]   

        # Entire dataset
        else:
            pass 
            
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        image = Image.open(image_path)
        label = Image.open(label_path)
        Y = self.transform(label)
        X = self.transform(image)
        return X, Y
    
## Dataset classes - PH2
class PH2_dataset(torch.utils.data.Dataset):
    def __init__(self, mode, transform, data_path='/dtu/datasets1/02514/PH2_Dataset_images'):
        # Initialization
        self.transform = transform
        self.image_paths = glob.glob(data_path + '/*/*_Dermoscopic_Image/*.bmp')
        self.label_paths = glob.glob(data_path + '/*/*_lesion/*.bmp')
        c = list(zip(self.image_paths, self.label_paths))

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
        
        image = Image.open(image_path)
        label = Image.open(label_path)
        Y = 1*self.transform(label)
        X = self.transform(image)
        return X, Y    
        





