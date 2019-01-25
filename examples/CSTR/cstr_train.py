import time, os, sys, argparse
import numpy as np
import pandas as pd
np.random.seed(94103)

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.transforms import Compose

torch.backends.cudnn.deterministic = True
torch.manual_seed(94103)

from lighter.datasets.cstr import CSTRDataset
from lighter.datasets.transforms import Numpy2Tensor, Word2Vec, FixedLengthPad1D, QuantiseULaw

from lighter.train import Trainer, AsynchronousLoader, DefaultStep
from lighter.train.callbacks import ProgBarCallback, CheckpointCallback
from lighter.train.metrics import CategoricalAccuracy, IOUMetric, IOULoss, F1Metric, F1Loss

parser = argparse.ArgumentParser()
parser.add_argument('--text_dir', required = True, type = str, help = 'Directory of training text dir')
parser.add_argument('--audio_dir', required = True, type = str, help = 'Folder containing the training DICOM files')
parser.add_argument('--epochs', required = True, type = int, help = 'Number of epochs to train for')
parser.add_argument('--model_name', required = True, type = str, help = 'Output model name')

args = parser.parse_args()


model_w2v = Word2Vec(args.text_dir, dim = 128)
model_w2v.train()

text_transforms = Compose([model_w2v, FixedLengthPad1D(length = max([len(s) for s in model_w2v.sentences])), Numpy2Tensor()])
audio_transforms = Compose([QuantiseULaw(u = 255), Numpy2Tensor()])

train_set = CSTRDataset(args.text_dir, args.audio_dir, text_transforms = text_transforms, audio_transforms = audio_transforms)

x,y = train_set[0]
#print(train_set[0])
