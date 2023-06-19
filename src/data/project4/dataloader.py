import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from PIL import Image, ExifTags
from pycocotools.coco import COCO
import random
import json
import os
import pickle

from typing import Tuple

from albumentations.pytorch import ToTensorV2
import albumentations as A

from src.utils import set_seed

class WasteDataset(Dataset):

    def __init__(
            self, mode, transform, 
            data_path='/dtu/datasets1/02514/data_wastedetection', 
            seed=420, region_size=(224, 224),
            use_super_categories=True
        ):

        # Read annotations
        self.transform = transform
        self.region_transform = A.Compose([
            A.Resize(region_size[0], region_size[1]),
            ToTensorV2(),
            ])
        self.transform_pred_bbox = A.Compose([
            A.Resize(region_size[0], region_size[1]),
            ToTensorV2(),
        ])
        
        # Load dataset
        self.data_path = data_path
        anns_file_path = self.data_path + '/' + 'annotations.json'
        self.coco = COCO(anns_file_path)
        with open(anns_file_path, 'r') as f:
            dataset = json.loads(f.read())
        
        # Extract categories, supercategories and other useful information
        self.use_super_categories = use_super_categories
        if self.use_super_categories:
            self.categories = list(set([cat['supercategory'] for cat in dataset['categories']]))
            # Create dictionary for mapping between ids and chosen category
            supercat2id = {x: index for index, x in enumerate(self.categories)}
            self.id2supercatid = {cat['id']: supercat2id[cat['supercategory']] for cat in dataset['categories']}
        else:
            self.categories = list(set([cat['name'] for cat in dataset['categories']]))
            # Create dictionary for mapping between ids and chosen category
            cat2id = {x: index for index, x in enumerate(self.categories)}
            self.id2catid = {cat['id']: cat2id[cat['name']] for cat in dataset['categories']}

        # Get number of classes and category mapping
        self.num_classes = len(self.categories) + 1
        self.id2cat = {index: x for index, x in enumerate(self.categories)}
        # add background class
        self.id2cat[len(self.id2cat)] = 'Background'

        # get image id and paths
        self.image_paths = [(
            self.coco.getAnnIds(imgIds=img_data['id'], catIds=[], iscrowd=None), 
            img_data['file_name']) 
        for img_data in dataset['images']]
        
        # shuffle image order 
        random.seed(seed)
        random.shuffle(self.image_paths)

        # Obtain Exif orientation tag code
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                self.orientation = orientation
                break   

        # Compute splits sizes
        train_size = int(0.7 * len(self.image_paths))
        val_size = int(0.1 * len(self.image_paths))

        # Get dataset split
        if mode == 'train':
            self.image_paths = self.image_paths[:train_size]
        elif mode == 'val':
            self.image_paths = self.image_paths[train_size:train_size+val_size]
        elif mode == 'test':
            self.image_paths = self.image_paths[train_size+val_size:]
        else:
            raise ValueError('mode must be one of train, val or test')

        # Load proposed bounding boxes
        with open('/work3/s184984/02514/project4/bboxes/bboxes_no_zeros.pkl', 'rb') as fp:
            proposed_bboxes = pickle.load(fp)

        # Restrict to train, test or validation set
        selected_images         = [img_path[1] for img_path in self.image_paths]
        self.proposed_bboxes    = {
            img_name: tuple([list(bbox_) for bbox_ in proposed_bboxes_]) 
            for img_name, proposed_bboxes_ in proposed_bboxes.items() 
            if img_name in selected_images
        }

    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def extract_resize_region(self, image, bbox, box_type: str):
        (x, y, w, h) = map(int, bbox)
        if box_type == 'gt':
            return torch.tensor([x, y, x+w, y+h]), self.region_transform(image = image[y:y+h, x:x+w, :])['image']
        elif box_type == 'predicted':
            return torch.tensor([x, y, x+w, y+h]), self.transform_pred_bbox(image = image[y:y+h, x:x+w, :])['image']

    def process_image(self, image):

        # Rotate portrait and upside down images if necessary
        if image._getexif():
            exif = dict(image._getexif().items())
            if self.orientation in exif:
                if exif[self.orientation] == 3:
                    image = image.rotate(180,expand=True)
                if exif[self.orientation] == 6:
                    image = image.rotate(270,expand=True)
                if exif[self.orientation] == 8:
                    image = image.rotate(90,expand=True)
                    
        image = np.array(image) / 255 
        return image.astype(np.float32)
        
    def __getitem__(self, idx):
        ### IMAGE ###
        # Load ids
        anno_id, img_path = self.image_paths[idx]

        # Load and process image metadata
        image = Image.open(self.data_path + '/' + img_path)
        image = self.process_image(image)

        ### BBOX AND LABEL ###
        # Load bbox and bbox ids
        category_ids, bboxes = zip(*[([item['category_id']], item['bbox']) for item in self.coco.loadAnns(anno_id)])
        if self.use_super_categories:
            category_ids = tuple([[self.id2supercatid[id_[0]]] for id_ in category_ids])
        else:
            category_ids = tuple([[self.id2catid[id_[0]]] for id_ in category_ids])

        # Extract proposed bounding boxes
        pred_bboxes = torch.tensor(self.proposed_bboxes[img_path])
        
        # Transform image
        try:
            transformed = self.transform(image=image, bboxes = bboxes, category_ids = category_ids)
        except:
            corrupted = np.where(np.any(np.array(bboxes)<0, axis=1))
            # TODO: remove the bounding box from bbox and category idx, problem, are tuples..
            transformed = self.transform(image=image, bboxes = bboxes, category_ids = category_ids)
            print('#### FAILED BOUNDING BOX ####')

        image = transformed['image']
        
        # Extract categories, bboxes and new image after transformations are applied
        category_ids = transformed['category_ids']
        transformed_bboxes = transformed['bboxes']
        modified_image = np.array(image.permute(1,2,0))
        bboxes, extracted_bboxes = zip(*(self.extract_resize_region(modified_image, bbox, box_type='gt') for bbox in transformed_bboxes))        
        pred_bboxes, extracted_pred_bboxes = zip(*(self.extract_resize_region(modified_image, bbox, box_type='predicted') for bbox in pred_bboxes))
             
        return (image, torch.tensor(category_ids), (torch.stack(bboxes), torch.stack(extracted_bboxes)), (torch.stack(pred_bboxes), torch.stack(extracted_pred_bboxes)))


def get_loaders(
        dataset, 
        batch_size=64, seed=1, num_workers=1, 
        augmentations:dict={'rotate': False, 'flip': False}, 
        img_size=(512, 512), region_size=(224, 224),
        use_super_categories=True
    ) -> Tuple[dict, int]:

    if dataset == 'waste':
        root = '/dtu/datasets1/02514/data_wastedetection'
    
    # Set seed for split control
    set_seed(seed)

    # Define transforms for training
    train_transform = A.Compose([
        A.Resize(img_size[0], img_size[1]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']), is_check_shapes=False)

    # Define transforms for training
    test_transform = A.Compose([
        A.Resize(img_size[0], img_size[1]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']), is_check_shapes=False)

    # Get train, validation and test sets
    trainset    = WasteDataset('train', transform=train_transform, region_size=region_size, use_super_categories=use_super_categories)
    valset      = WasteDataset('val', transform=test_transform, region_size=region_size, use_super_categories=use_super_categories)
    testset     = WasteDataset('test', transform=test_transform, region_size=region_size, use_super_categories=use_super_categories)

    # Get dataloaders
    trainloader = DataLoader(trainset,  batch_size=batch_size, shuffle=True,  num_workers=num_workers, collate_fn=lambda x: x)
    valloader   = DataLoader(valset,    batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=lambda x: x)
    testloader  = DataLoader(testset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=lambda x: x)

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