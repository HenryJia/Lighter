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
parser.add_argument('--model_name', required = True, type = str, help = 'Output model name')
parser.add_argument('--train_split', required = False, type = float, help = 'Proportion of the data used to train', default = 0.9)
parser.add_argument('--use_amp', required = False, action = 'store_true', help = 'Whether to use NVidia automatic mixed precision.')
parser.add_argument('--half', required = False, action = 'store_true', help = 'Whether to use half precision')
parser.add_argument('--device', required = False, type = str, default = 'cuda:0', help = 'Which CUDA device to use')

args = parser.parse_args()

# Note, we could use torch.utils.rnn.pack_sequence instead of padding everything to a fixed length, but this is codewise easier for now
mel_transform = Lambda(lambda x: melspectrogram(x, sr = 16000, n_mels = 256, n_fft = 2048, hop_length = 128).astype(np.float32))
x_transforms = Compose([mel_transform, Numpy2Tensor()])
y_transforms = Compose([QuantiseULaw(u = 255), Lambda(lambda x: x.astype(np.long)), Numpy2Tensor()])

def joint_transform_f(x):
    x = x[1]
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
train_loader = AsynchronousLoader(data_set, device = torch.device(args.device), batch_size = 1, shuffle = False, sampler = train_sampler)
validation_loader = AsynchronousLoader(data_set, device = torch.device(args.device), batch_size = 1, shuffle = False, sampler = validation_sampler)

model = CSTRMelModel().to(torch.device(args.device))
model.load_state_dict(torch.load(args.model_name))

for i, (x, y) in enumerate(validation_loader):
    with torch.no_grad():
        x = x[0]
        out = torch.zeros((1, 1)).to(device = torch.device(args.device), dtype = torch.long)
        bridge = model.bridge(x)

        pb = tqdm(total = bridge.shape[2])

        for i in range(bridge.shape[2]):
            if out.shape[1] < model.wavenet.get_receptive_field():
                inp = out
            else:
                inp = out[:, -model.wavenet.get_receptive_field():]

            out_probs = model.wavenet(bridge[..., i:i + 1], inp)[:, :, -1]
            out_probs = torch.exp(out_probs).view(-1)

            out_rv = torch.distributions.Categorical(out_probs)
            out_current = out_rv.sample().view(1, 1)

            out_max = torch.argmax(out_probs).view((1, 1))

            pb.set_postfix(current = out_current.item(), out_max = out_max.item())
            pb.update(1)

            if out is not None:
                out = torch.cat([out, out_current], dim = 1)
            else:
                out = out_current

        pb.close()
        out = out[0].cpu().numpy()
    break

expand = ExpandULaw(u = 255)
out = out.astype(np.float32)
out = expand(out - 1)
out = (out + 1.0) / 2.0 # rescale to [0.0, 1.0]
out = out * (np.iinfo(np.int16).max - np.iinfo(np.int16).min) + np.iinfo(np.int16).min
out = out.astype(np.int16)
scipy.io.wavfile.write('test.wav', 16000, out)

out = y[0].cpu().numpy().astype(np.float32)
out = expand(out - 1)
out = (out + 1.0) / 2.0 # rescale to [0.0, 1.0]
out = out * (np.iinfo(np.int16).max - np.iinfo(np.int16).min) + np.iinfo(np.int16).min
out = out.astype(np.int16)
scipy.io.wavfile.write('test_control.wav', 16000, out)
