import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from PIL import Image, ExifTags
from pycocotools.coco import COCO

from albumentations.pytorch import ToTensorV2
import albumentations as A

from src.utils import set_seed

def NoOp(image, **kwargs):
    return image

class WasteDataset(Dataset):

    def __init__(self, mode, transform, data_path='/dtu/datasets1/02514/data_wastedetection', seed=420):
        # Initialization
        #data_path = '/Users/arond.jacobsen/Desktop/DTU/8_semester/02514_Deep_Learning_in_Computer_Vision/2_part/0_project/sample_data/PH2_Dataset_images'
        
        self.transform = transform
        # Read annotations
        self.data_path = data_path
        anns_file_path = self.data_path + '/' + 'annotations.json'
        self.coco = COCO(anns_file_path)
        with open(anns_file_path, 'r') as f:
            dataset = json.loads(f.read())
        # get image id and paths
        self.image_paths = [(self.coco.getAnnIds(imgIds=img_data['id'], catIds=[], iscrowd=None), img_data['file_name']) for img_data in dataset['images']]
        # shuffle     
        random.seed(seed)
        #random.shuffle(image_paths)
        
        
        # Obtain Exif orientation tag code
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                self.orientation = orientation
                break
        
        # todo: category id to label
        

        train_size = int(0.7 * len(self.image_paths))
        val_size = int(0.1 * len(self.image_paths))

        if mode == 'train':
            self.image_paths = self.image_paths[:train_size]

        elif mode == 'val':
            self.image_paths = self.image_paths[train_size:train_size+val_size]

        elif mode == 'test':
            self.image_paths = self.image_paths[train_size+val_size:]

        else:
            raise ValueError('mode must be one of train, val or test')

        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        # IMAGE
        # load ids
        anno_id, img_path = self.image_paths[idx]
        # Load and process image metadata
        image = Image.open(self.data_path + '/' + img_path)
        if image._getexif():
            exif = dict(image._getexif().items())
            # Rotate portrait and upside down images if necessary
            if self.orientation in exif:
                if exif[self.orientation] == 3:
                    image = image.rotate(180,expand=True)
                if exif[self.orientation] == 6:
                    image = image.rotate(270,expand=True)
                if exif[self.orientation] == 8:
                    image = image.rotate(90,expand=True)
                    
        image = np.array(image) / 255 
        image = image.astype(np.float32)
        # BBOX AND LABEL
        # Load mask ids
        category_ids, bboxes = zip(*[(item['category_id'], item['bbox']) for item in self.coco.loadAnns(anno_id)])
        
        
        # TRANSFORM
        transformed = self.transform(image=image, bboxes = bboxes, category_ids = category_ids)#, bbox=?)
        return transformed['image'], transformed['category_ids'], transformed['bboxes']



    
def get_loaders(dataset, batch_size=64, seed=1, num_workers=1, augmentations:dict={'rotate': False, 'flip': False}) -> dict:

    if dataset == 'waste':
        root = '/dtu/datasets1/02514/data_wastedetection'
    
    
    # Set seed for split control
    set_seed(seed)
    img_size = (1024, 1024)
    # Define transforms for training
    train_transform = A.Compose([
        A.Resize(img_size[0], img_size[1]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']), is_check_shapes=False)


    test_transform = A.Compose([
        A.Resize(img_size[0], img_size[1]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']), is_check_shapes=False)


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