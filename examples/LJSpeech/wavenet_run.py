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

from tqdm import tqdm

torch.backends.cudnn.deterministic = True
torch.manual_seed(94103)

from lighter.datasets.ljspeech import LJSpeechDataset
from lighter.datasets.transforms import Numpy2Tensor, Char2Vec, FixLength1D, Permute, QuantiseULaw, ExpandULaw, Normalize, SampleSequence1D

from lighter.train import Trainer, AsynchronousLoader, DefaultStep
from lighter.train.callbacks import ProgBarCallback, CheckpointCallback
from lighter.train.metrics import CategoricalAccuracy

from lighter.models.model_lib.ljspeech import MelModel

parser = argparse.ArgumentParser()
parser.add_argument('--json', type = str, help = 'JSON file to specify arguments and will overwrite any command line arguments')

parser.add_argument('--csv_dir', type = str, help = 'Directory of training text dir')
parser.add_argument('--audio_dir', type = str, help = 'Folder containing the training DICOM files')
parser.add_argument('--sample_rate', default = 16000, type = int, help = 'Sample rate of data')
parser.add_argument('--model_name', type = str, help = 'Output model name')
parser.add_argument('--train_split', type = float, help = 'Proportion of the data used to train', default = 0.9)
parser.add_argument('--use_amp', action = 'store_true', help = 'Whether to use NVidia automatic mixed precision.')
parser.add_argument('--half', action = 'store_true', help = 'Whether to use half precision')
parser.add_argument('--device', type = str, default = 'cuda:0', help = 'Which CUDA device to use')

parser.add_argument('--n_mels', default = 80, type = int, help = 'Number of mel filterbanks to use')
parser.add_argument('--n_fft', default = 800, type = int, help = 'Size of fft window to use')
parser.add_argument('--hops', default = 200, type = int, help = 'Size of mel spectrogram hops to use')
parser.add_argument('--depth', default = 8, type = int, help = 'Depth of a single WaveNet stack')
parser.add_argument('--stacks', default = 2, type = int, help = 'Number of WaveNet stacks to use')
parser.add_argument('--res_channels', default = 64, type = int, help = 'Number of WaveNet residual channel to use')
parser.add_argument('--skip_channels', default = 256, type = int, help = 'Number of WaveNet skip channels to use')
parser.add_argument('--softmax_temperature', default = 0.25, type = int, help = 'Temperature of softmax for generation')
parser.add_argument('--samples', default = 10, type = int, help = 'Number of samples to generate')

args, unknown_args = parser.parse_known_args()

args = vars(args)
if args['json'] is not None:
    with open(args['json']) as f:
        args_json = json.load(f)
    args = {**args, **args_json}

model = MelModel(n_mels = args['n_mels'], n_fft = args['n_fft'], hops = args['hops'], depth = args['depth'], stacks = args['stacks'], res_channels = args['res_channels'], skip_channels = args['skip_channels']).to(torch.device(args['device']))
model.load_state_dict(torch.load(args['model_name'], map_location = torch.device(args['device'])))
model.eval()

# Note, we could use torch.utils.rnn.pack_sequence instead of padding everything to a fixed length, but this is codewise easier for now
mel_transform = Lambda(lambda x: melspectrogram(x, sr = args['sample_rate'], n_mels = args['n_mels'], n_fft = args['n_fft'], hop_length = args['hops']).astype(np.float32))
log_transform = Lambda(lambda x: np.log(np.clip(x, a_min = 1e-5, a_max = None))) # Compress into Log-scale
h_transforms = Compose([mel_transform, log_transform, Numpy2Tensor()])
y_transforms = Compose([QuantiseULaw(u = 255), Lambda(lambda x: x.astype(np.long)), Numpy2Tensor()])

def joint_transform_f(x):
    x = x[1]
    h = h_transforms(x)
    x = y_transforms(x)
    y = x.clone()
    # Shift by 1 so our targets are the predictions
    x = F.pad(x[..., :-1], (1, 0), mode = 'constant', value = 0)
    return [x, h], y

joint_transforms = Lambda(joint_transform_f)

# Create one dataset for everything and use PyTorch samplers to do the training/validation split
data_set = LJSpeechDataset(args['csv_dir'], args['audio_dir'], text_transforms = None, audio_transforms = None, joint_transforms = joint_transforms, sample_rate = args['sample_rate'])

#perm = np.random.permutation(len(data_set))
perm = np.load('data_permutations.npy') # Use the same permutations as training so we get the same train and validation sets
train_sampler = SubsetRandomSampler(perm[:np.round(args['train_split'] * len(data_set)).astype(int)].tolist())
validation_sampler = SubsetRandomSampler(perm[np.round(args['train_split'] * len(data_set)).astype(int):].tolist())
train_loader = AsynchronousLoader(data_set, device = torch.device(args['device']), batch_size = 1, shuffle = False, sampler = train_sampler)
validation_loader = AsynchronousLoader(data_set, device = torch.device(args['device']), batch_size = 1, shuffle = False, sampler = validation_sampler)

if not os.path.exists('./samples'):
    os.makedirs('./samples')

for i, (x, y) in enumerate(validation_loader):
    if i >= args['samples']:
        break

    print('Generating sample number {} out of {}'.format(i + 1, args['samples']))
    with torch.no_grad():
        x = x[1]
        out = torch.zeros((1, 1)).to(device = torch.device(args['device']), dtype = torch.long)
        bridge = model.bridge(x)

        pb = tqdm(total = bridge.shape[2])

        for j in range(bridge.shape[2]):
            if out.shape[1] < model.wavenet.get_receptive_field():
                inp = out
            else:
                inp = out[:, -model.wavenet.get_receptive_field():]

            out_probs = model.wavenet(inp, bridge[..., max(j + 1 - model.wavenet.get_receptive_field(), 0):j + 1])[:, :, -1]
            #out_probs = torch.exp(out_probs).view(-1)
            out_probs = torch.softmax(out_probs / args['softmax_temperature'], dim = 1).view(-1)

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

        expand = ExpandULaw(u = 255)
        out = out.astype(np.float32)
        out = expand(out)
        out = (out + 1.0) / 2.0 # rescale to [0.0, 1.0]
        out = out * (np.iinfo(np.int16).max - np.iinfo(np.int16).min) + np.iinfo(np.int16).min
        out = out.astype(np.int16)
        scipy.io.wavfile.write('./samples/test' + str(i) + '.wav', args['sample_rate'], out)

        y_wav = y[0].cpu().numpy().astype(np.float32)
        y_wav = expand(y_wav)
        y_wav = (y_wav + 1.0) / 2.0 # rescale to [0.0, 1.0]
        y_wav = y_wav * (np.iinfo(np.int16).max - np.iinfo(np.int16).min) + np.iinfo(np.int16).min
        y_wav = y_wav.astype(np.int16)
        scipy.io.wavfile.write('./samples/test_groundtruth' + str(i) + '.wav', args['sample_rate'], y_wav)
