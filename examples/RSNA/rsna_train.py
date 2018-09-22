import time, os, sys, argparse
import numpy as np
import pandas as pd
np.random.seed(94103)

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.transforms import Compose, Lambda

torch.backends.cudnn.deterministic = True
torch.manual_seed(94103)

from fractals.datasets.rsna import RSNADataset, split_validation, GetBbox
from fractals.datasets.transforms import Numpy2Tensor, Reshape, Resize, Bbox2Binary, Normalize

from fractals.models.model_lib.rsna import UNet

from fractals.train import Trainer, AsynchronousLoader, DefaultClosure
from fractals.train.callbacks import ProgBarCallback, CheckpointCallback
from fractals.train.metrics import BinaryAccuracy, IOUMetric, IOULoss, F1Metric, F1Loss

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type = str, help = 'Root directory containing the folder with the DICOM files and the csv files')
parser.add_argument('--train_dir', type = str, help = 'Folder containing the training DICOM files')
parser.add_argument('--train_csv', type = str, help = 'Filename of the csv label file for training')
parser.add_argument('--epochs', type = int, help = 'Number of epochs to train for')

args = parser.parse_args()

root_dir = args.root_dir
dcm_dir = os.path.join(root_dir, args.train_dir)
data_df_dir = os.path.join(root_dir, args.train_csv)

data_df = pd.read_csv(data_df_dir)

train_df, validation_df = split_validation(data_df, 0.8)

print('Training dataframe lengths:\n', len(train_df))
print('Validation dataframe lengths:\n', len(validation_df))
print('Head of training dataframes:\n', train_df.head())
print('Head of validation dataframes:\n', train_df.head())

image_transforms = Compose([Resize((256, 256)), Numpy2Tensor(), Lambda(lambda x: x.float()), Reshape((1, 256, 256)), Normalize(0, 255)])
y_transforms = Compose([GetBbox(), Normalize(0, 1024), Bbox2Binary((256, 256)), Lambda(lambda x: x.float())])

train_set = RSNADataset(train_df, dcm_dir, [('pixel_array', image_transforms)], y_transforms)
validation_set = RSNADataset(validation_df, dcm_dir, [('pixel_array', image_transforms)], y_transforms)

model = UNet().cuda()
loss = [F1Loss().cuda()]
optim = Adam(model.parameters(), lr = 3e-4)
metrics = [(0, BinaryAccuracy().cuda()), (0, F1Metric().cuda()), (0, IOUMetric().cuda())]

train_closure = DefaultClosure(model = model, losses = loss, optimizer = optim, metrics = metrics, train = True)
validation_closure = DefaultClosure(model = model, losses = loss, optimizer = optim, metrics = metrics, train = False)

train_loader = AsynchronousLoader(train_set, device = torch.device('cuda:0'), batch_size = 32, shuffle = True)
validation_loader = AsynchronousLoader(validation_set, device = torch.device('cuda:0'), batch_size = 32, shuffle = True)

train_callbacks = [ProgBarCallback(check_queue = True)]
validation_callback = train_callbacks + [CheckpointCallback('RSNA_UNet.pth', monitor = 'IOUMetric_0', save_best = True, mode = 'max')]

trainer = Trainer(train_loader, train_closure, train_callbacks)
validator = Trainer(validation_loader, validation_closure, validation_callback)

for i in range(args.epochs):
    print('Training Epoch {}'.format(i))
    next(trainer)
    print('Validating Epoch {}'.format(i))
    next(validator)
