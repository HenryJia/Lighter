import time, os, sys, argparse
import numpy as np # linear algebra 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
np.random.seed(94103)

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage, Accuracy
from ignite.handlers import ModelCheckpoint, Timer
from ignite.contrib.handlers import ProgressBar

from lighter.models.layers import Flatten

from lighter.train.loaders import AsynchronousLoader
from lighter.train.steps import DefaultStep
from lighter.handlers import EpochTimer
#from lighter.train.callbacks import ProgBarCallback, CheckpointCallback
#from lighter.train.metrics import CategoricalAccuracy, IOUMetric, IOULoss, F1Metric, F1Loss

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type = str, help = 'Root directory of the MNIST dataset')

args = parser.parse_args()


# Load data and create loaders
train_set = MNIST(args.root_dir, train = True, download = True, transform = ToTensor())
validation_set = MNIST(args.root_dir, train = False, download = True, transform = ToTensor())

train_loader = AsynchronousLoader(train_set, device = torch.device('cuda:0'), batch_size = 1024, shuffle = True, workers = 10)
validation_loader = AsynchronousLoader(validation_set, device = torch.device('cuda:0'), batch_size = 1024, shuffle = True, workers = 10)


# Build model
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


# Set up the Engine
train_step = DefaultStep(model = model, losses = loss, optimizer = optim, train = True)
validation_step = DefaultStep(model = model, losses = loss, optimizer = optim, train = False)

trainer = Engine(train_step)
evaluator = Engine(validation_step)

# Note we don't need running averages on these because the way ignite metrics are implemented makes them averages already
# Also note we don't actually need to actually set it equal to something because we don't need to keep track of it
Accuracy(output_transform = lambda x: x['out_pairs']['output_0']).attach(trainer, name = 'accuracy_0')
Accuracy(output_transform = lambda x: x['out_pairs']['output_0']).attach(evaluator, name = 'accuracy_0')

RunningAverage(output_transform=lambda x: x['losses']['NLLLoss_0']).attach(trainer, 'NLLLoss_0')
RunningAverage(output_transform=lambda x: x['losses']['NLLLoss_0']).attach(evaluator, 'NLLLoss_0')

EpochTimer().attach(trainer)
EpochTimer().attach(evaluator)

trainer.add_event_handler(event_name = Events.EPOCH_COMPLETED, handler = lambda engine, loader: evaluator.run(loader), loader = validation_loader)

checkpoint = ModelCheckpoint('./', '', score_function = lambda eng: eng.state.metrics['accuracy_0'], require_empty = False)
evaluator.add_event_handler(event_name = Events.EPOCH_COMPLETED, handler = checkpoint, to_save={'model' : model})


trainer.run(train_loader, max_epochs = 10)
