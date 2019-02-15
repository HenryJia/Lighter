import time, os, sys, argparse
import numpy as np
import pandas as pd
import scipy
np.random.seed(94103)

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.transforms import Compose, Lambda
from torch.utils.data import SubsetRandomSampler

from tqdm import tqdm

torch.backends.cudnn.deterministic = True
torch.manual_seed(94103)

from lighter.datasets.cstr import CSTRDataset
from lighter.datasets.transforms import Numpy2Tensor, Char2Vec, FixLength1D, Permute, Int2OneHot, QuantiseULaw, ExpandULaw

from lighter.train import Trainer, AsynchronousLoader, DefaultStep
from lighter.train.callbacks import ProgBarCallback, CheckpointCallback
from lighter.train.metrics import CategoricalAccuracy, NLLLoss

from lighter.models.model_lib.cstr import CSTRCharacterModel

parser = argparse.ArgumentParser()
parser.add_argument('--text_dir', required = True, type = str, help = 'Directory of training text dir')
parser.add_argument('--audio_dir', required = True, type = str, help = 'Folder containing the training DICOM files')
parser.add_argument('--epochs', required = True, type = int, help = 'Number of epochs to train for')
parser.add_argument('--model_name', required = True, type = str, help = 'Output model name')
parser.add_argument('--train_split', required = False, type = float, help = 'Proportion of the data used to train', default = 0.9)

args = parser.parse_args()

# Note, we could use torch.utils.rnn.pack_sequence instead of padding everything to a fixed length, but this is codewise easier for now
text_transforms = Compose([Char2Vec(), FixLength1D(256), Permute((1, 0)), Numpy2Tensor()])
audio_transforms = Compose([QuantiseULaw(u = 255), FixLength1D(12 * 16000), Int2OneHot(256), Permute((1, 0)), Numpy2Tensor()])
joint_transforms = Lambda(lambda x: [(x[0], x[1][:, :-1]), x[1][:, 1:]])

# Create one dataset for everything and use PyTorch samplers to do the training/validation split
data_set = CSTRDataset(args.text_dir, args.audio_dir, text_transforms = text_transforms, audio_transforms = audio_transforms, joint_transforms = joint_transforms, sample_rate = 16000)

perm = np.random.permutation(len(data_set))
train_sampler = SubsetRandomSampler(perm[:np.round(args.train_split * len(data_set)).astype(int)].tolist())
validation_sampler = SubsetRandomSampler(perm[np.round(args.train_split * len(data_set)).astype(int):].tolist())
train_loader = AsynchronousLoader(data_set, device = torch.device('cuda:0'), batch_size = 2, shuffle = False, sampler = train_sampler, queue_size = 3)
validation_loader = AsynchronousLoader(data_set, device = torch.device('cuda:0'), batch_size = 2, shuffle = False, sampler = validation_sampler, queue_size = 3)

model = CSTRCharacterModel().cuda()

loss = [NLLLoss(one_hot = True).cuda()]
optim = Adam(model.parameters(), lr = 3e-4)
metrics = [(0, CategoricalAccuracy(one_hot = True).cuda())]

train_step = DefaultStep(model = model, losses = loss, optimizer = optim, metrics = metrics, train = True)
validation_step = DefaultStep(model = model, losses = loss, optimizer = optim, metrics = metrics, train = False)

train_callbacks = [ProgBarCallback(check_queue = True)]
validation_callback = train_callbacks + [CheckpointCallback('cstr.pth', monitor = 'CategoricalAccuracy_0', save_best = True, mode = 'max')]

trainer = Trainer(train_loader, train_step, train_callbacks)
validator = Trainer(validation_loader, validation_step, validation_callback)

x, y = data_set[0]

for i in range(1):
    print('Training Epoch {}'.format(i))
    next(trainer)
    print('Validating Epoch {}'.format(i))
    next(validator)

