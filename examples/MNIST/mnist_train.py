import time, os, sys, argparse
import numpy as np # linear algebra 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
np.random.seed(94103)

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST

from lighter.models.layers import Flatten

from lighter.train import Trainer, AsynchronousLoader, DefaultStep
from lighter.train.callbacks import ProgBarCallback, CheckpointCallback
from lighter.train.metrics import CategoricalAccuracy, IOUMetric, IOULoss, F1Metric, F1Loss

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type = str, help = 'Root directory containing the folder with the MNIST dataset')

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
metrics = [(0, CategoricalAccuracy().cuda())]

train_step = DefaultStep(model = model, losses = loss, optimizer = optim, metrics = metrics, train = True)
validation_step = DefaultStep(model = model, losses = loss, optimizer = optim, metrics = metrics, train = False)

train_loader = AsynchronousLoader(train_set, device = torch.device('cuda:0'), batch_size = 1024, shuffle = True)
validation_loader = AsynchronousLoader(validation_set, device = torch.device('cuda:0'), batch_size = 1024, shuffle = True)

train_callbacks = [ProgBarCallback(check_queue = True)]
validation_callback = train_callbacks + [CheckpointCallback('mnist.pth', monitor = 'CategoricalAccuracy_0', save_best = True, mode = 'max')]

trainer = Trainer(train_loader, train_step, train_callbacks)
validator = Trainer(validation_loader, validation_step, validation_callback)

for i in range(20):
    print('Training Epoch {}'.format(i))
    next(trainer)
    print('Validating Epoch {}'.format(i))
    next(validator)
