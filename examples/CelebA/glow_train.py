import time, os, sys, argparse
import numpy as np # linear algebra 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
np.random.seed(94103)

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms
from torchvision.datasets import CelebA

from lighter.models.layers import Flatten

from lighter.train import Trainer, AsynchronousLoader, DefaultStep
from lighter.train.callbacks import ProgBarCallback, CheckpointCallback
from lighter.train.metrics import CategoricalAccuracy, IOUMetric, IOULoss, F1Metric, F1Loss

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type = str, help = 'Root directory containing the folder with the CelebA dataset')
parser.add_argument('--img_size', type = int, help = 'Image ize to set our dataset to', default = 128)
args = parser.parse_args()

transforms = transforms.Compose([transforms.CenterCrop((178, 178)),
                                 transforms.Resize(args.img_size),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))])

train_set = CelebA(args.root_dir, train = True, download = True, transform = transforms)
validation_set = CelebA(args.root_dir, train = False, download = True, transform = transforms)


