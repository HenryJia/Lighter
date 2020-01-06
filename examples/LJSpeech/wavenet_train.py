import time, os, sys, argparse, json
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

import apex
from apex import amp

from tqdm import tqdm

torch.backends.cudnn.deterministic = True
torch.manual_seed(94103)

from lighter.datasets.ljspeech import LJSpeechDataset
from lighter.datasets.transforms import Numpy2Tensor, Char2Vec, FixLength1D, Permute, QuantiseULaw, ExpandULaw, Normalize, SampleSequence1D

from lighter.train import SupervisedTrainer, AsynchronousLoader, SupervisedStep
from lighter.train.callbacks import ProgBarCallback, CheckpointCallback
from lighter.train.metrics import CategoricalAccuracy

from lighter.modules.model_lib.wavenet import MelWaveNetModel

parser = argparse.ArgumentParser()
parser.add_argument('--json', type=str, help='JSON file to specify arguments and will overwrite any command line arguments')

parser.add_argument('--csv_dir', type=str, help='Location of the metadata csv')
parser.add_argument('--audio_dir', type=str, help='Folder containing the training audio files')
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate of data')
parser.add_argument('--segment_length', default=16000, type=int, help='Segment length of each piece of training data')
parser.add_argument('--epochs', type=int, help='Number of epochs to train for')
parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
parser.add_argument('--model_name', type=str, help='Output model name')
parser.add_argument('--train_split', type=float, help='Proportion of the data used to train', default=0.9)
parser.add_argument('--use_amp', action='store_true', help='Whether to use NVidia automatic mixed precision.')
parser.add_argument('--half', action='store_true', help='Whether to use half precision')
parser.add_argument('--device', default='cuda:0', type=str, help='Which CUDA device to use')

parser.add_argument('--n_mels', default=80, type=int, help='Number of mel filterbanks to use')
parser.add_argument('--n_fft', default=800, type=int, help='Size of fft window to use')
parser.add_argument('--hops', default=200, type=int, help='Size of mel spectrogram hops to use')
parser.add_argument('--depth', default=8, type=int, help='Depth of a single WaveNet stack')
parser.add_argument('--stacks', default=2, type=int, help='Number of WaveNet stacks to use')
parser.add_argument('--res_channels', default=64, type=int, help='Number of WaveNet residual channel to use')
parser.add_argument('--skip_channels', default=256, type=int, help='Number of WaveNet skip channels to use')

args, unknown_args = parser.parse_known_args()

args = vars(args)
if args['json'] is not None:
    with open(args['json']) as f:
        args_json = json.load(f)
    args = {**args, **args_json}

model = MelWaveNetModel(n_mels=args['n_mels'], n_fft=args['n_fft'], hops=args['hops'], depth=args['depth'], stacks=args['stacks'], res_channels=args['res_channels'], skip_channels=args['skip_channels']).to(torch.device(args['device']))

# Note, we could use torch.utils.rnn.pack_sequence instead of padding everything to a fixed length, but this is codewise easier for now
mel_transform = Lambda(lambda x: melspectrogram(x, sr=args['sample_rate'], n_mels=args['n_mels'], n_fft=args['n_fft'], hop_length=args['hops']).astype(np.float32))
log_transform = Lambda(lambda x: np.log(np.clip(x, a_min=1e-5, a_max=None))) # Compress into Log-scale
h_transforms = Compose([mel_transform, log_transform, Numpy2Tensor()])
y_transforms = Compose([QuantiseULaw(u=255), Lambda(lambda x: x.astype(np.long)), Numpy2Tensor()])
seq_sampler = SampleSequence1D(length=args['segment_length'], dim=0)

def joint_transform_f(x):
    x = seq_sampler(x[1])
    h = h_transforms(x)
    x = y_transforms(x)
    y = x.clone()
    x = F.pad(x[..., :-1], (1, 0), mode = 'constant', value = 0)
    return [x, h], y

joint_transforms = Lambda(joint_transform_f)

# Create one dataset for everything and use PyTorch samplers to do the training/validation split
data_set = LJSpeechDataset(args['csv_dir'], args['audio_dir'], text_transforms=None, audio_transforms=None, joint_transforms=joint_transforms, sample_rate=args['sample_rate'])

perm = np.random.permutation(len(data_set))
np.save('data_permutations.npy', perm)
train_sampler = SubsetRandomSampler(perm[:np.round(args['train_split'] * len(data_set)).astype(int)].tolist())
validation_sampler = SubsetRandomSampler(perm[np.round(args['train_split'] * len(data_set)).astype(int):].tolist())
train_loader = AsynchronousLoader(data_set, device=torch.device(args['device']), batch_size=args['batch_size'], shuffle=False, sampler=train_sampler)
validation_loader = AsynchronousLoader(data_set, device=torch.device(args['device']), batch_size=args['batch_size'], shuffle=False, sampler=validation_sampler)

if args['half']:
    model = model.half()
    optim = Adam(model.parameters(), lr=1e-3, eps=1e-4)
elif args['use_amp']:
    optim = Adam(model.parameters(), lr=1e-3, eps=1e-8)
    model, optim = amp.initialize(model, optim, opt_level='O2')
else:
    optim = Adam(model.parameters(), lr=1e-3, eps=1e-8)

class SoftmaxRMSESTD(nn.Module): # Compute the RMSE as a fraction of the standard deviation
    def forward(self, out, target):
        return torch.sqrt(torch.mean((torch.argmax(out, dim=1) - target).float() ** 2)) / torch.std(target.float())

loss = nn.CrossEntropyLoss().cuda()
metrics = [CategoricalAccuracy().to(device=torch.device(args['device'])), SoftmaxRMSESTD().to(device=torch.device(args['device']))]

train_step = SupervisedStep(model=model, loss=loss, optimizer=optim, metrics=metrics, use_amp=args['use_amp'])
validation_step = SupervisedStep(model=model, loss=loss, optimizer=optim, metrics=metrics, use_amp=args['use_amp'])

train_callbacks = [ProgBarCallback(check_queue=True)]
validation_callback = train_callbacks + [CheckpointCallback(args['model_name'], monitor='CategoricalAccuracy', save_best=True, mode='max')]

trainer = SupervisedTrainer(train_loader, train_step, train_callbacks)
validator = SupervisedTrainer(validation_loader, validation_step, validation_callback)


for i in range(args['epochs']):
    model.train()
    print('Training Epoch {}'.format(i))
    next(trainer)
    model.eval()
    print('Validating Epoch {}'.format(i))
    next(validator)
