import time, os, sys, argparse
from collections import OrderedDict
import numpy as np
import pandas as pd
np.random.seed(94103)

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.transforms import Compose, Lambda

torch.backends.cudnn.deterministic = True
torch.manual_seed(94103)

# Scikit Learn doesn't let us use different metrics, we want to use 1 - IOU as our metric
#from sklearn.cluster import KMeans
from nltk.cluster.kmeans import KMeansClusterer

from tqdm import tqdm

from pycocotools.coco import COCO

from lighter.datasets.coco import COCODataset, GetBbox
from lighter.datasets.transforms import Numpy2Tensor, Reshape, Permute, Resize, Bbox2Binary, Normalize

from lighter.models.densenet import DenseNet
from lighter.models.yolo import YOLOClassifier

from lighter.train import Trainer, AsynchronousLoader, DefaultClosure
from lighter.train.callbacks import ProgBarCallback, CheckpointCallback
from lighter.train.metrics import CombineLinear, BinaryAccuracy, IOUMetric, IOULoss, F1Metric, F1Loss

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', required = True, type = str, help = 'Root directory containing the folder with the DICOM files and the csv files')
parser.add_argument('--train_dir', required = True, type = str, help = 'Folder containing the training DICOM files')
parser.add_argument('--train_json', required = True, type = str, help = 'Filename of the csv label file for training')
parser.add_argument('--val_dir', required = True, type = str, help = 'Folder containing the validation DICOM files')
parser.add_argument('--val_json', required = True, type = str, help = 'Filename of the csv label file for validation')
parser.add_argument('--epochs', required = True, type = int, help = 'Number of epochs to train for')
parser.add_argument('--model_name', required = True, type = str, help = 'Output model name')
parser.add_argument('--num_bbox', default = 5, type = int, help = 'Number of bounding boxes for model to output')

args = parser.parse_args()

root_dir = args.root_dir
train_dir = os.path.join(root_dir, args.train_dir)
train_json = os.path.join(root_dir, args.train_json)
validation_dir = os.path.join(root_dir, args.val_dir)
validation_json = os.path.join(root_dir, args.val_json)

# Prepare PyTorch datasets

train_coco = COCO(train_json)
validation_coco = COCO(validation_json)

x_transforms = Compose([Resize((224, 224)), Permute((2, 0, 1)), Numpy2Tensor(), Lambda(lambda x: x.float()), Normalize(0, 255)])
y_transforms = Compose([GetBbox(), Numpy2Tensor(), Lambda(lambda x: x.float())]) # Shift everything back to same size as image

train_set = COCODataset(train_dir, train_coco, x_transforms, y_transforms)
kmeans_set = COCODataset(train_dir, train_coco, lambda x: None, y_transforms, load_image = False) # Faster dataset for just computing kmeans priors
validation_set = COCODataset(validation_dir, validation_coco, x_transforms, y_transforms)

# First we need to perform KMeans clustering to get bounding box priors
print('Creating dataset of bounding box dimensions for clustering for the priors')
train_wh = np.concatenate([y[:, 3:5] for y in tqdm(kmeans_set)], axis = 0)

'''
    Simple utility function for IOU for bounding boxes

    Note, (x1, y1), (x2, y2) are the centres and not the top left coordinates
'''
def bbox_iou(x1, y1, w1, h1, x2, y2, w2, h2):
    x_a = torch.max(x1 - w1 / 2.0, x2 - w2 / 2.0)
    y_a = torch.max(y1 - h1 / 2.0, y2 - h2 / 2.0)
    x_b = torch.min(x1 + h1 / 2.0, x2 + w2 / 2.0)
    y_b = torch.max(y1 + h1 / 2.0, y2 + h2 / 2.0)

    intersection = torch.clamp(x_b - x_a, min = 0) * torch.clamp(y_b - y_a, min = 0)
    union = w1 * h1 + w2 * h2 - intersection

    return intersection / (union + 1e-6)

kclusterer = KMeansClusterer(args.num_bbox, distance=nltk.cluster.util.cosine_distance, repeats=25)
assigned_clusters = kclusterer.cluster(data, assign_clusters=True)

kmeans_wh = KMeans(n_clusters = args.num_bbox)
kmeans_wh.fit(train_wh)
bbox_priors = kmeans_wh.cluster_centers_
np.save('priors.npy', bbox_priors)
bbox_priors = torch.from_numpy(bbox_priors).cuda()

# Set up the network

features = DenseNet(growth_rate = 8, block_config = (4, 8, 16, 32), activation = nn.LeakyReLU(inplace = True), input_channels = 3)

classifier = YOLOClassifier(features.output_channels, bbox_priors, train_set.num_classes)

model = nn.Sequential(features, nn.AdaptiveMaxPool2d((2, 2)), classifier)

