import time, os, sys, argparse
import numpy as np # linear algebra 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
np.random.seed(94103)

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST

from fractals.models.layers import Flatten

from fractals.train import Trainer, AsynchronousLoader, DefaultClosure, ProgBarCallback
from fractals.train.metrics import CategoricalAccuracy, IOUMetric, IOULoss, F1Metric, F1Loss

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type = str, help = 'Root directory containing the folder with the DICOM files and the csv files')

args = parser.parse_args()

train_set = MNIST(args.root_dir, train = True, download = True, transform = ToTensor())
validation_set = MNIST(args.root_dir, train = False, download = True, transform = ToTensor())

model = nn.Sequential(nn.Conv2d(1, 16, 3, padding = 1),
                      nn.LeakyReLU(),
                      nn.MaxPool2d((2, 2)),
                      nn.Conv2d(16, 32, 3, padding = 1),
                      nn.LeakyReLU(),
                      nn.MaxPool2d((2, 2)),
                      Flatten(),
                      nn.Linear(32 * 7 * 7, 512),
                      nn.LeakyReLU(),
                      nn.Linear(512, 10),
                      nn.LogSoftmax(dim = 1)).cuda()

loss = [nn.NLLLoss().cuda()]
optim = Adam(model.parameters(), lr = 3e-4)
metrics = [(0, CategoricalAccuracy().cuda())]#, (0, F1Metric().cuda()), (0, F1Metric().cuda()), (0, IOUMetric().cuda())]

train_closure = DefaultClosure(model = model, losses = loss, optimizer = optim, metrics = metrics, train = True)
validation_closure = DefaultClosure(model = model, losses = loss, optimizer = optim, metrics = metrics, train = False)

train_loader = AsynchronousLoader(train_set, device = torch.device('cuda:0'), batch_size = 256, shuffle = True)
validation_loader = AsynchronousLoader(validation_set, device = torch.device('cuda:0'), batch_size = 256, shuffle = True)

callbacks = [ProgBarCallback(check_queue = True)]

trainer = Trainer(train_loader, train_closure, callbacks)
validator = Trainer(validation_loader, validation_closure, callbacks)

for i in range(10):
    print('Training Epoch {}'.format(i))
    next(trainer)
    print('Validating Epoch {}'.format(i))
    next(validator)
