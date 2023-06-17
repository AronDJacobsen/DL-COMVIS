import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from PIL import Image, ExifTags
from pycocotools.coco import COCO
import random
import json
import os

from albumentations.pytorch import ToTensorV2
import albumentations as A

from src.utils import set_seed



    
def get_loaders(dataset, batch_size=64, seed=1, num_workers=1, augmentations:dict={'rotate': False, 'flip': False}, img_size=(512, 512), region_size=(224, 224)) -> dict:

    if dataset == 'waste':
        root = '/dtu/datasets1/02514/data_wastedetection'
    
    
    # Set seed for split control
    set_seed(seed)
    #img_size = (512, 512)
    # Define transforms for training
    train_transform = A.Compose([
        A.Resize(img_size[0], img_size[1]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']), is_check_shapes=False)


    test_transform = A.Compose([
        A.Resize(img_size[0], img_size[1]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']), is_check_shapes=False)


    trainset = WasteDataset('train', transform=train_transform, region_size=region_size)
    valset = WasteDataset('val', transform=test_transform, region_size=region_size)
    testset = WasteDataset('test', transform=test_transform, region_size=region_size)


    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=lambda x: x)
    valloader   = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=lambda x: x)
    testloader  = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=lambda x: x)

    # Return loaders in dictionary
    return {'train': trainloader, 'validation': valloader, 'test': testloader}, trainset.num_classes



def NoOp(image, **kwargs):
    return image


def dummy_data():
    # an image
    image = Image.open(os.getcwd() + '/src/data/project4/dummy_data/img.jpg')
    image = np.array(image) / 255 
    image = image.astype(np.float32)
    # the bbox
    category_ids = 0
    bboxes = [50, 50, 100, 100]
    return image, category_ids, bboxes


class WasteDataset(Dataset):

    def __init__(self, mode, transform, data_path='/dtu/datasets1/02514/data_wastedetection', seed=420, region_size=(224, 224)):
       # Initialization
        #data_path = '/Users/arond.jacobsen/Desktop/DTU/8_semester/02514_Deep_Learning_in_Computer_Vision/2_part/0_project/sample_data/PH2_Dataset_images'

        # Read annotations
        self.transform = transform
        self.region_size = region_size
        self.region_transform = A.Compose([
            A.Resize(self.region_size[0], self.region_size[1]),
            ToTensorV2(),
            ])
        self.data_path = data_path
        anns_file_path = self.data_path + '/' + 'annotations.json'
        self.coco = COCO(anns_file_path)
        with open(anns_file_path, 'r') as f:
            dataset = json.loads(f.read())
        
        # useful data conversions
        self.categories = dataset['categories']
        # all unique
        self.super_cats = list(set([cat['supercategory'] for cat in self.categories]))
        self.num_classes = len(self.super_cats)
        # id/name conversions of super categories
        self.super_cat_id_2_name = {index: x for index, x in enumerate(self.super_cats)}
        self.name_2_super_cat_id = {x: index for index, x in enumerate(self.super_cats)}
        # the indidual categories to super
        self.cat_id_2_super_cat_id = {cat['id']: self.name_2_super_cat_id[cat['supercategory']] for cat in self.categories}
        
        
        # get image id and paths
        self.image_paths = [(self.coco.getAnnIds(imgIds=img_data['id'], catIds=[], iscrowd=None), img_data['file_name']) for img_data in dataset['images']]
        # shuffle     
        random.seed(seed)
        random.shuffle(self.image_paths)

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

    def extract_resize_region(self, image, bbox):
        (x, y, w, h) = map(int, bbox)
        return torch.tensor([x, y, x+w, y+h]), self.region_transform(image = image[y:y+h, x:x+w, :])['image']

    def __getitem__(self, idx):

        # IMAGE
        # load ids
        anno_id, img_path = self.image_paths[idx]
        # Load and process image metadata
        image = Image.open(self.data_path + '/' + img_path)#, cv2.COLOR_BGR2RGB
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
        category_ids, bboxes = zip(*[(self.cat_id_2_super_cat_id[item['category_id']], item['bbox']) for item in self.coco.loadAnns(anno_id)])
        
        # THE ESTIMATED
        # TODO: get the estimates
        pred_bboxes = bboxes
        num_pred_bboxes = len(pred_bboxes)
        
        # TRANSFORM with bboxes
        try:
            transformed = self.transform(image=image, bboxes = bboxes + pred_bboxes, category_ids = list(category_ids) + tuple([-1]*num_pred_bboxes))#, bbox=?)
        except:
            corrupted = np.where(np.any(np.array(bboxes)<0, axis=1))
            # TODO: remove the bounding box from bbox and category idx, problem, are tuples..
            transformed = self.transform(image=image, bboxes = bboxes + pred_bboxes, category_ids = category_ids + tuple([-1]*num_pred_bboxes))#, bbox=?)
            print('#### FAILED BOUNDING BOX ####')

        image = transformed['image']
        pred_found = transformed['category_ids'].count(-1)
        # get data and sort it
        category_ids = transformed['category_ids'][:-pred_found]
        # In case some are dropped:
        transformed_bboxes = transformed['bboxes'][:-pred_found]
        transformed_pred_bboxes = transformed['bboxes'][pred_found:]
        # extract their image
        modified_image = np.array(image.permute(2, 1, 0))
        bboxes, extracted_bboxes = zip(*(self.extract_resize_region(modified_image, bbox) for bbox in transformed_bboxes))
        pred_bboxes, extracted_pred_bboxes = zip(*(self.extract_resize_region(modified_image, bbox) for bbox in transformed_pred_bboxes))
        

        out = (image, torch.tensor(category_ids), (torch.stack(bboxes), torch.stack(extracted_bboxes)), (torch.stack(pred_bboxes), torch.stack(extracted_pred_bboxes)))
        
        return out

