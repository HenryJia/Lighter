import time, os, sys, argparse
import numpy as np # linear algebra 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
np.random.seed(94103)

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.transforms import Compose, Lambda

from fractals.datasets.rsna import RSNADataset, split_validation, GetBbox
from fractals.datasets.transforms import Numpy2Tensor, Reshape, Resize, Bbox2Binary, Normalize

from fractals.models.model_lib.rsna import UNet

from fractals.train import Trainer, AsynchronousLoader, DefaultClosure, ProgBarCallback
from fractals.train.metrics import Accuracy

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type = str, help = 'Root directory containing the folder with the DICOM files and the csv files')
parser.add_argument('--train_dir', type = str, help = 'Folder containing the training DICOM files')
parser.add_argument('--train_csv', type = str, help = 'Filename of the csv label file for training')
#parser.add_argument('--details_csv', type = str, help = 'Filename of the csv detailed info file for training')

args = parser.parse_args()

root_dir = args.root_dir
dcm_dir = os.path.join(root_dir, args.train_dir)
data_df_dir = os.path.join(root_dir, args.train_csv)
#details_df_dir = os.path.join(root_dir, args.details_csv)

data_df = pd.read_csv(data_df_dir)
#details_df = pd.read_csv(details_df_dir)

train_df, validation_df = split_validation(data_df, 0.8)

print('Training dataframe lengths:\n', len(train_df))
print('Validation dataframe lengths:\n', len(validation_df))
print('Head of training dataframes:\n', train_df.head())
print('Head of validation dataframes:\n', train_df.head())

image_transforms = Compose([Resize((256, 256)), Numpy2Tensor(), Lambda(lambda x: x.float()), Reshape((1, 256, 256)), Normalize(0, 255)])
y_transforms = Compose([GetBbox(), Normalize(0, 1024), Bbox2Binary((256, 256)), Lambda(lambda x: x.float())])

train_set = RSNADataset(train_df, dcm_dir, [('pixel_array', image_transforms)], y_transforms)
validation_set = RSNADataset(validation_df, dcm_dir, [('pixel_array', image_transforms)], y_transforms)

#sample = train_set[0]
#print(sample[1][0].shape)

model = UNet().cuda()
loss = [nn.BCELoss().cuda()]
optim = Adam(model.parameters(), lr = 3e-4)
metrics = {}

closure = DefaultClosure(model = model, losses = loss, optimizer = optim, metrics = metrics, train = True)
loader = AsynchronousLoader(train_set, device = torch.device('cuda:0'), batch_size = 128, shuffle = True)
callbacks = [ProgBarCallback(decription = 'Training', check_queue = True)]

trainer = Trainer(loader, closure, callbacks)

next(trainer)
