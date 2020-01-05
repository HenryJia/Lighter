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

from lighter.datasets.rsna import RSNADataset, split_validation, GetBbox
from lighter.datasets.transforms import Numpy2Tensor, Reshape, Resize, Bbox2Binary, Normalize, JointRandomHFlip

from lighter.modules.model_lib.rsna import UNet
from lighter.modules.densenet import DenseNet

from lighter.train import SupervisedTrainer, AsynchronousLoader, SupervisedStep
from lighter.train.callbacks import ProgBarCallback, CheckpointCallback
from lighter.train.metrics import CombineLinear, BinaryAccuracy, IOUMetric, IOULoss, F1Metric, F1Loss

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', required = True, type = str, help = 'Root directory containing the folder with the DICOM files and the csv files')
parser.add_argument('--train_dir', required = True, type = str, help = 'Folder containing the training DICOM files')
parser.add_argument('--train_csv', required = True, type = str, help = 'Filename of the csv label file for training')
parser.add_argument('--epochs', required = True, type = int, help = 'Number of epochs to train for')
parser.add_argument('--model_name', required = True, type = str, help = 'Output model name')

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

x_transforms = Compose([Resize((256, 256)), Numpy2Tensor(), Lambda(lambda x: x.float()), Reshape((1, 256, 256)), Normalize(0, 255)])
y_transforms = Compose([GetBbox(), Normalize(0, 1024), Bbox2Binary((256, 256)), Lambda(lambda x: x.float())])
aug_transforms = JointRandomHFlip(p = 0.5)

train_set = RSNADataset(train_df, dcm_dir, [('pixel_array', x_transforms)], y_transforms, aug_transforms)
validation_set = RSNADataset(validation_df, dcm_dir, [('pixel_array', x_transforms)], y_transforms)

features = DenseNet(growth_rate = 8, block_config = (4, 8, 16, 32), activation = nn.LeakyReLU(inplace = True), input_channels = 1)
model = nn.Sequential(features,
                      nn.Conv2d(features.output_channels, 16, kernel_size = 1),
                      nn.LeakyReLU(inplace = True),
                      nn.Upsample(size = (256, 256), mode = 'nearest'),
                      nn.Conv2d(16, 1, kernel_size = 3, padding = 1),
                      nn.Sigmoid()).to(torch.device('cuda:0'))

loss = CombineLinear([IOULoss().cuda(), nn.BCELoss().cuda()], [1 - 1e-4, 1e-4])
optim = Adam(model.parameters(), lr = 3e-4)
metrics = [BinaryAccuracy().cuda(), F1Metric().cuda(), IOUMetric().cuda()]

train_step = SupervisedStep(model = model, loss = loss, optimizer = optim, metrics = metrics, train = True)
validation_step = SupervisedStep(model = model, loss = loss, optimizer = optim, metrics = metrics, train = False)

train_loader = AsynchronousLoader(train_set, device = torch.device('cuda:0'), batch_size = 32, shuffle = True)
validation_loader = AsynchronousLoader(validation_set, device = torch.device('cuda:0'), batch_size = 32, shuffle = True)

train_callbacks = [ProgBarCallback(check_queue = True)]
validation_callback = train_callbacks + [CheckpointCallback(args.model_name, monitor = 'IOUMetric', save_best = True, mode = 'max')]

trainer = SupervisedTrainer(train_loader, train_step, train_callbacks)
validator = SupervisedTrainer(validation_loader, validation_step, validation_callback)

for i in range(args.epochs):
    print('Training Epoch {}'.format(i))
    model.train()
    next(trainer)
    print('Validating Epoch {}'.format(i))
    model.eval()
    next(validator)
