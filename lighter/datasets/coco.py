import os
from time import time
import random
from collections import OrderedDict

import numpy as np
from PIL import Image
from io import BytesIO
import cv2

import torch
from torch.utils.data import Dataset

from pycocotools.coco import COCO



class COCODataset(Dataset):
    """
    Dataset for the Microsoft Common Objects in Context dataset

    Parameters
    ----------
    data_dir: String
        Location of the COCO images
    coco: COCO class
        COCO class instance to use for accessing COCO JSON API
    x_transforms: PyTorch Transform
        PyTorch transform to apply to images
    y_transforms: PyTorch Transform
        PyTorch transform to apply to annotations
    joint_transforms: PyTorch Transform
        PyTorch transforms to apply to images and annotations simultaneously
    categories: List
        List of categories to load
        Default is empty, to load all categories
    crowds: Boolean
        Whether to include crowds in the annotations
        Default is False
    load_image: Boolean
        Whether to bother loading images
        Default is true for training, can be set to false to just load the annotations faster
        We do not provide an option for doing the converse as time taken to load annotations is significantly less than images and does not slow it down
    """
    def __init__(self, data_dir, coco, x_transforms = None, y_transforms = None, joint_transforms = None, categories = [], crowds = False, load_image = True):
        super(COCODataset, self).__init__()
        self.data_dir = data_dir
        self.x_transforms = x_transforms
        self.y_transforms = y_transforms
        self.joint_transforms = joint_transforms
        self.crowds = crowds
        self.load_image = load_image

        self.coco = coco

        self.category_ids = self.coco.getCatIds(catNms = categories)
        self.categories = self.coco.loadCats(self.category_ids)
        if len(categories) == 0:
            self.img_ids = self.coco.getImgIds()
        else:
            self.img_ids = self.coco.getImgIds(catIds = self.category_ids)
        self.img_dicts = self.coco.loadImgs(self.img_ids)
        self.num_classes = max(self.category_ids) + 1


    def __getitem__(self, idx):
        t0 = time()
        img_dict = self.img_dicts[idx]

        # Load the targets
        ann_ids = self.coco.getAnnIds(imgIds = img_dict['id'], iscrowd = int(self.crowds))
        anns = self.coco.loadAnns(ann_ids) + [img_dict] # We add img_dict to pass on metadata of the image

        if self.y_transforms:
            y = self.y_transforms(anns)

        # Load the image
        if self.load_image:
            img = cv2.imread(os.path.join(self.data_dir, img_dict['file_name']))
            if len(img.shape) == 2: # Put all images into (H, W, 3)
                img = img[..., None]
                img = np.repeat(img, 3, axis = 2)

            if self.x_transforms and self.load_image:
                x = self.x_transforms(img)

            if self.joint_transforms:
                x, y = self.joint_transforms([x, y])

            # return both image and annotations if we loaded them both
            return x, y

        # Otherwise just return the annotations
        return y


    def __len__(self):
        return len(self.img_dicts)



class GetBbox(object):
    """
    Simple class to convert COCO Annotations to a NumPy array of coordinates and width/height for the bounding boxes
    Format of the returned data is [class_id, x, y, width, height]

    Parameters:
    -----------
    centre: Bool
        Whether to return the centre as the x, y coordinates instead of the top left
    max_boxes: Integer
        Maximum number of bounding boxes
    """
    def __init__(self, centre = True, max_boxes = 50):
        self.centre = centre
        self.max_boxes = max_boxes

    def __call__(self, anns):

        y = np.zeros((self.max_boxes, 5), dtype = np.float32) - 1 # Default value for no bounding box is -1

        # Iterate through all the objects and add them to our semantic segmentation targets
        # COCO bounding boxes are already in [x, y, width, height] format
        for i, a in enumerate(anns[:-1]):
            if not a['iscrowd']:
                bbox = np.array(a['bbox'])

                if self.centre:
                    bbox[:2] += bbox[2:] / 2.0

                bbox[::2] /= anns[-1]['width'] # Normalise to [0, 1]
                bbox[1::2] /= anns[-1]['height']

                y[i, 0] = a['category_id']
                y[i, 1:] = bbox

        return y
