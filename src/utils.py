import torch
import numpy as np

from torchvision import transforms

def set_seed(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)

def invertNormalization(train_mean, train_std):
    return transforms.Compose([
        transforms.Normalize(
            mean=[0., 0., 0.],
            std=[1/0.0214, 1/0.0208, 1/0.0223]
        ),
        transforms.Normalize(
            mean=[-0.5132, -0.4369, -0.3576],
            std=[1., 1., 1.]
        )
    ])

def accuracy(y_pred, y_true):
    """accuracy of segmentation wrt. ground truth mask"""
    return (y_pred == y_true).sum().item() / y_true.numel()

def specificity(y_pred, y_true):
    """specificity of segmentation wrt. ground truth mask"""
    return ((y_pred == y_true) & (y_true == 0)).sum().item() / (y_true == 0).sum().item()

def sensitivity(y_pred, y_true):
    """sensitivity of segmentation wrt. ground truth mask"""
    return ((y_pred == y_true) & (y_true == 1)).sum().item() / (y_true == 1).sum().item()

def iou(y_pred, y_true):
    """intersection over union of segmentation wrt. ground truth mask"""
    return (y_pred & y_true).sum().item() / (y_pred | y_true).sum().item() 

def dice(y_pred, y_true):
    """dice coefficient of segmentation wrt. ground truth mask"""
    return 2 * (y_pred & y_true).sum().item() / (y_pred.sum().item() + y_true.sum().item())


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
        

# Get loaders function
from torch.utils.data import DataLoader

def get_loaders(dataset, size = 256, batch_size = 2):

    if dataset == 'DRIVE':

        train_transform = transforms.Compose([transforms.Resize((size, size)), 
                                            transforms.ToTensor()])

        testval_transform = transforms.Compose([transforms.Resize((size, size)), 
                                            transforms.ToTensor()])

        # Creating loader-dicts:        loaders[fold_number]['train']
        # Remember to set seed within loader if specific seed is needed!
        loaders = {fold: {'train': DataLoader( DRIVE(mode = 'train', fold = fold, transform = train_transform  ), batch_size=batch_size, shuffle=True, num_workers=3), 
                        'test' : DataLoader( DRIVE(mode = 'test',  fold = fold, transform = testval_transform), batch_size=batch_size, shuffle=True, num_workers=3),
                        'validation'  : DataLoader( DRIVE(mode = 'val',   fold = fold, transform = testval_transform), batch_size=batch_size, shuffle=True, num_workers=3)} 
                for fold in range(20) }

        return loaders
        
    if dataset == 'PH2':

        division = 1

        # won't work since halving in the CNN structure will end up with and uneven number
        #size = (int(560/division), int(768/division))
        h, w = size, size
        size = (int(h/division), int(w/division))

        train_transform = transforms.Compose([transforms.Resize(size), 
                                            transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.Resize(size), 
                                            transforms.ToTensor()])

        trainset = PH2_dataset(mode='train', transform=train_transform)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)

        valset = PH2_dataset(mode='val', transform=train_transform)
        val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=3)

        testset = PH2_dataset(mode='test', transform=train_transform)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=3)
    
        loaders = {'train': train_loader, 'validation': val_loader, 'test': test_loader}
    
        return loaders


