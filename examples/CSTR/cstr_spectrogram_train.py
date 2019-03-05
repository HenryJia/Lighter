import time, os, sys, argparse
import numpy as np
import pandas as pd
import scipy
np.random.seed(94103)
from librosa.feature import melspectrogram

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.transforms import Compose, Lambda
from torch.utils.data import SubsetRandomSampler

from tqdm import tqdm

torch.backends.cudnn.deterministic = True
torch.manual_seed(94103)

from lighter.datasets.cstr import CSTRDataset
from lighter.datasets.transforms import Numpy2Tensor, Char2Vec, FixLength1D, Permute, QuantiseULaw, ExpandULaw, Normalize, SampleSequence1D

from lighter.train import Trainer, AsynchronousLoader, DefaultStep
from lighter.train.callbacks import ProgBarCallback, CheckpointCallback
from lighter.train.metrics import CategoricalAccuracy

from lighter.models.model_lib.cstr import CSTRMelModel

parser = argparse.ArgumentParser()
parser.add_argument('--text_dir', required = True, type = str, help = 'Directory of training text dir')
parser.add_argument('--audio_dir', required = True, type = str, help = 'Folder containing the training DICOM files')
parser.add_argument('--epochs', required = True, type = int, help = 'Number of epochs to train for')
parser.add_argument('--batch_size', default = 4, type = int, help = 'Batch size')
parser.add_argument('--model_name', required = True, type = str, help = 'Output model name')
parser.add_argument('--train_split', required = False, type = float, help = 'Proportion of the data used to train', default = 0.9)
parser.add_argument('--use_amp', required = False, action = 'store_true', help = 'Whether to use NVidia automatic mixed precision.')
parser.add_argument('--half', required = False, action = 'store_true', help = 'Whether to use half precision')
parser.add_argument('--device', default = 'cuda:0', type = str, help = 'Which CUDA device to use')
parser.add_argument('--depth', default = 10, type = int, help = 'Which CUDA device to use')
parser.add_argument('--stacks', default = 2, type = int, help = 'Which CUDA device to use')

args = parser.parse_args()

model = CSTRMelModel(wavenet_depth = args.depth, stacks = args.stacks).to(torch.device(args.device))

# Note, we could use torch.utils.rnn.pack_sequence instead of padding everything to a fixed length, but this is codewise easier for now
mel_transform = Lambda(lambda x: melspectrogram(x, sr = 16000, n_mels = 256, n_fft = 2048, hop_length = 128).astype(np.float32))
x_transforms = Compose([mel_transform, Numpy2Tensor()])
y_transforms = Compose([QuantiseULaw(u = 255), Lambda(lambda x: x.astype(np.long)), Numpy2Tensor()])

seq_sampler = SampleSequence1D(length = model.wavenet.get_receptive_field() + 4096, dim = 0)

def joint_transform_f(x):
    #if x[1].shape[0] < 2 ** 9 * 8:
        #np.append(np.zeros((2 ** 9 * 8 - x[1].shape[0],)), x[1], dim = 0) # In case we have really short audio
    x = seq_sampler(x[1])
    x1 = x_transforms(x)
    x2 = y_transforms(x)
    y = x2.clone()
    y[..., :model.wavenet.get_receptive_field() - 1] = -1
    x2 = F.pad(x2[..., :-1], (1, 0), mode = 'constant', value = 0)
    return [x1, x2], y

joint_transforms = Lambda(joint_transform_f)

# Create one dataset for everything and use PyTorch samplers to do the training/validation split
data_set = CSTRDataset(args.text_dir, args.audio_dir, text_transforms = None, audio_transforms = None, joint_transforms = joint_transforms, sample_rate = 16000)

perm = np.random.permutation(len(data_set))
train_sampler = SubsetRandomSampler(perm[:np.round(args.train_split * len(data_set)).astype(int)].tolist())
validation_sampler = SubsetRandomSampler(perm[np.round(args.train_split * len(data_set)).astype(int):].tolist())
train_loader = AsynchronousLoader(data_set, device = torch.device(args.device), batch_size = args.batch_size, shuffle = False, sampler = train_sampler)
validation_loader = AsynchronousLoader(data_set, device = torch.device(args.device), batch_size = args.batch_size, shuffle = False, sampler = validation_sampler)


if args.half:
    model = model.half()
    optim = Adam(model.parameters(), lr = 1e-3, eps = 1e-4)
else:
    optim = Adam(model.parameters(), lr = 1e-3, eps = 1e-8)

loss = [nn.NLLLoss(ignore_index = -1).cuda()]
metrics = [(0, CategoricalAccuracy(ignore_index = 0).cuda())]

train_step = DefaultStep(model = model, losses = loss, optimizer = optim, metrics = metrics, train = True, use_amp = args.use_amp)
validation_step = DefaultStep(model = model, losses = loss, optimizer = optim, metrics = metrics, train = False, use_amp = args.use_amp)

train_callbacks = [ProgBarCallback(check_queue = True)]
validation_callback = train_callbacks + [CheckpointCallback(args.model_name, monitor = 'CategoricalAccuracy_0', save_best = True, mode = 'max')]

trainer = Trainer(train_loader, train_step, train_callbacks)
validator = Trainer(validation_loader, validation_step, validation_callback)


for i in range(args.epochs):
    print('Training Epoch {}'.format(i))
    next(trainer)
    print('Validating Epoch {}'.format(i))
    next(validator)
