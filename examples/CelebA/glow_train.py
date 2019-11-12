import time, os, sys, argparse, math
import numpy as np # linear algebra 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
np.random.seed(94103)

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms

from lighter.datasets.celeba import CelebADataset

import apex
from apex import amp

from lighter.modules.model_lib.glow import Glow

from lighter.train import Trainer, AsynchronousLoader, DefaultStep
from lighter.train.callbacks import ProgBarCallback, CheckpointCallback
from lighter.train.metrics import CategoricalAccuracy, IOUMetric, IOULoss, F1Metric, F1Loss

parser = argparse.ArgumentParser()
parser.add_argument('--json', type=str, help='JSON file to specify arguments and will overwrite any command line arguments')

parser.add_argument('--root_dir', type=str, help='Root directory containing the folder with the CelebA dataset')
parser.add_argument('--img_size', type=int, help='Image ize to set our dataset to', default=128)

parser.add_argument('--epochs', type=int, help='Number of epochs to train for')
parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
parser.add_argument('--learning_rate', default=1e-4, type=float, help='Learning rate')
parser.add_argument('--model_name', type=str, help='Output model name')
#parser.add_argument('--train_split', type=float, help='Proportion of the data used to train', default = 0.9)
parser.add_argument('--use_amp', action='store_true', help='Whether to use NVidia automatic mixed precision.')
parser.add_argument('--half', action='store_true', help='Whether to use half precision')
parser.add_argument('--device', default='cuda:0', type=str, help='Which CUDA device to use')

parser.add_argument('--n_bits', type=int, help='Number of bits we quantize to per pixel', default=8)
parser.add_argument('--n_flows', type=int, help='Number of flows in each Glow block', default=32)
parser.add_argument('--n_blocks', type=int, help='Number of Glow blocks in our model', default=4)

args, unknown_args = parser.parse_known_args()

args = vars(args)
if args['json'] is not None:
    with open(args['json']) as f:
        args_json = json.load(f)
    args = {**args, **args_json}

n_bins = 2. ** args['n_bits']
n_pixels = args['img_size'] * args['img_size'] * 3

transforms = transforms.Compose([transforms.CenterCrop((178, 178)),
                                 transforms.Resize(args['img_size']),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Lambda(lambda x: x + torch.rand(*x.shape) / n_bins)])
                                 #transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))])

train_set = CelebADataset(args['root_dir'], split='train', download=True, transform=transforms)
validation_set = CelebADataset(args['root_dir'], split='valid', download=True, transform=transforms)

model = Glow(in_channels=3, n_flows=args['n_flows'], n_blocks=args['n_blocks']).to(device=torch.device(args['device']))

class GlowLoss(nn.Module):
    def forward(self, output, targets):
        log_pz, log_det, z = output

        loss = math.log(n_bins) * n_pixels # Constant from quantisation
        loss = loss - log_det - log_pz # Probability of x
        return loss.mean() / (math.log(2) * n_pixels) # Get the loss in terms of bits per pixel (dimension)

loss = GlowLoss().to(device = torch.device(args['device']))
if args['half']:
    model = model.half()
    optim = Adam(model.parameters(), lr=args['learning_rate'], eps=1e-4)
elif args['use_amp']:
    optim = Adam(model.parameters(), lr=args['learning_rate'], eps=1e-8)
    model, optim = amp.initialize(model, optim, opt_level='O2')
else:
    optim = Adam(model.parameters(), lr=args['learning_rate'], eps=1e-8)

metrics = [] # Don't have any metrics for GLOW yet

train_step = DefaultStep(model=model, loss=loss, optimizer=optim, metrics=metrics, train=True, use_amp=args['use_amp'])
validation_step = DefaultStep(model=model, loss=loss, optimizer=optim, metrics=metrics, train=False, use_amp=args['use_amp'])

train_loader = AsynchronousLoader(train_set, device=torch.device('cuda:0'), batch_size=args['batch_size'], shuffle=True)
validation_loader = AsynchronousLoader(validation_set, device=torch.device('cuda:0'), batch_size=args['batch_size'], shuffle=True)

train_callbacks = [ProgBarCallback(check_queue=True)]
validation_callback = train_callbacks + [CheckpointCallback('model.pth', monitor='loss', save_best=False, mode='min')]

trainer = Trainer(train_loader, train_step, train_callbacks)
validator = Trainer(validation_loader, validation_step, validation_callback)

for i in range(20):
    print('Training Epoch {}'.format(i))
    next(trainer)
    print('Validating Epoch {}'.format(i))
    next(validator)

