import time, os, sys, argparse
import numpy as np
import pandas as pd
import scipy
np.random.seed(94103)

import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
from torch.optim import Adam
from torchvision.transforms import Compose, Lambda
from torch.utils.data import SubsetRandomSampler

from tqdm import tqdm

torch.backends.cudnn.deterministic = True
torch.manual_seed(94103)

from lighter.datasets.cstr import CSTRDataset
from lighter.datasets.transforms import Numpy2Tensor, Char2Vec, FixLength1D, Permute, QuantiseULaw, ExpandULaw, Normalize

from lighter.train import Trainer, AsynchronousLoader, DefaultStep
from lighter.train.callbacks import ProgBarCallback, CheckpointCallback
from lighter.train.metrics import CategoricalAccuracy

from lighter.models.model_lib.cstr import CSTRWaveNetModel

parser = argparse.ArgumentParser()
parser.add_argument('--text_dir', required = True, type = str, help = 'Directory of training text dir')
parser.add_argument('--audio_dir', required = True, type = str, help = 'Folder containing the training DICOM files')
parser.add_argument('--epochs', required = True, type = int, help = 'Number of epochs to train for')
parser.add_argument('--batch_size', default = 4, type = int, help = 'Batch size')
parser.add_argument('--model_name', required = True, type = str, help = 'Output model name')
parser.add_argument('--train_split', required = False, type = float, help = 'Proportion of the data used to train', default = 0.9)
parser.add_argument('--use_amp', required = False, action = 'store_true', help = 'Whether to use NVidia automatic mixed precision.')
parser.add_argument('--half', required = False, action = 'store_true', help = 'Whether to use half precision')
parser.add_argument('--device', required = False, type = str, default = 'cuda:0', help = 'Which CUDA device to use')

args = parser.parse_args()

# Note, we could use torch.utils.rnn.pack_sequence instead of padding everything to a fixed length, but this is codewise easier for now
text_transforms = Compose([Char2Vec(), FixLength1D(256, pad = 0), Numpy2Tensor()])
audio_transforms = Compose([QuantiseULaw(u = 255), Normalize(1, 1), Lambda(lambda x: x.astype(np.long)), FixLength1D(4 * 16000, pad = -1, stop = 0), Numpy2Tensor()])
joint_transforms = Lambda(lambda x: [(x[0][0], x[0][1], x[1][:-1].clamp(0, 256)), x[1]])

# Create one dataset for everything and use PyTorch samplers to do the training/validation split
data_set = CSTRDataset(args.text_dir, args.audio_dir, text_transforms = text_transforms, audio_transforms = audio_transforms, joint_transforms = joint_transforms, sample_rate = 16000)

perm = np.random.permutation(len(data_set))
train_sampler = SubsetRandomSampler(perm[:np.round(args.train_split * len(data_set)).astype(int)].tolist())
validation_sampler = SubsetRandomSampler(perm[np.round(args.train_split * len(data_set)).astype(int):].tolist())
train_loader = AsynchronousLoader(data_set, device = torch.device(args.device), batch_size = args.batch_size, shuffle = False, sampler = train_sampler)
validation_loader = AsynchronousLoader(data_set, device = torch.device(args.device), batch_size = args.batch_size, shuffle = False, sampler = validation_sampler)

model = CSTRWaveNetModel().to(torch.device(args.device))

if args.half:
    model = model.half()
    optim = Adam(model.parameters(), lr = 3e-4, eps = 1e-4)
else:
    optim = Adam(model.parameters(), lr = 3e-4, eps = 1e-8)

#[x, y], _ = data_set[1100]
#x = x[None].cuda()
#y = y[None].cuda()
#model(x, y)
#with torch.autograd.profiler.profile(use_cuda = True) as prof:
    #model(x, y)

#print(prof.key_averages().table(sort_by = 'cuda_time_total'))
#exit()

loss = [nn.NLLLoss(ignore_index = -1).cuda()]
metrics = [(0, CategoricalAccuracy(ignore_index = 0).cuda())]

train_step = DefaultStep(model = model, losses = loss, optimizer = optim, metrics = metrics, train = True, use_amp = args.use_amp)
validation_step = DefaultStep(model = model, losses = loss, optimizer = optim, metrics = metrics, train = False, use_amp = args.use_amp)

train_callbacks = [ProgBarCallback(check_queue = True)]
validation_callback = train_callbacks + [CheckpointCallback(args.model_name, monitor = 'CategoricalAccuracy_0', save_best = True, mode = 'max')]

trainer = Trainer(train_loader, train_step, train_callbacks)
validator = Trainer(validation_loader, validation_step, validation_callback)

#lengths = []
#for x, y in tqdm(data_set):
    #lengths += [y.shape[0]]
#lengths = np.array(lengths)
#print(np.mean(lengths) / 16000)
#print(np.std(lengths) / 16000)
#print(np.min(lengths) / 16000, np.max(lengths) / 16000)
#print(np.mean((lengths / 16000 < 4).astype(np.float32)))
#print(np.mean((lengths / 16000 < 5).astype(np.float32)))
#print(np.mean((lengths / 16000 < 8).astype(np.float32)))
#print(np.mean((lengths / 16000 < 10).astype(np.float32)))
#exit()

#x, y = data_set[1100]
#print(data_set.audio_list[0])
#y = y.numpy() - 1
#print(np.max(y), np.min(y))
#expand = ExpandULaw()
#y = expand(y)
#print(np.max(y), np.min(y))
#y = (y + 1.0) / 2.0 # rescale to [0.0, 1.0]
#y = y * (np.iinfo(np.int16).max - np.iinfo(np.int16).min) + np.iinfo(np.int16).min
#y = y.astype(np.int16)
#print(np.max(y), np.min(y))
#scipy.io.wavfile.write('test.wav', 16000, y)
#exit()

for i in range(args.epochs):
    print('Training Epoch {}'.format(i))
    next(trainer)
    print('Validating Epoch {}'.format(i))
    next(validator)

