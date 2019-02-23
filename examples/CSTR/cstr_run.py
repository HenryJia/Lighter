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
torch.manual_seed(94103 + 1)

from lighter.datasets.cstr import CSTRDataset
from lighter.datasets.transforms import Numpy2Tensor, Char2Vec, FixLength1D, ExpandULaw

from lighter.train import Trainer, AsynchronousLoader, DefaultStep
from lighter.train.callbacks import ProgBarCallback, CheckpointCallback
from lighter.train.metrics import CategoricalAccuracy, NLLLoss

from lighter.models.model_lib.cstr import CSTRWaveNetModel

parser = argparse.ArgumentParser()
parser.add_argument('-t', required = True, type = str, help = 'Sentence to do text to speech on up to 256 characters')
parser.add_argument('--device', required = False, type = str, default = 'cuda:0', help = 'Which CUDA device to use')

args = parser.parse_args()

# Note, we could use torch.utils.rnn.pack_sequence instead of padding everything to a fixed length, but this is codewise easier for now
text_transforms = Compose([Char2Vec(), FixLength1D(256), Numpy2Tensor()])
expand = ExpandULaw(u = 255)

model = CSTRWaveNetModel().to(torch.device(args.device))
model.load_state_dict(torch.load('./cstr_model.pth'))

with torch.no_grad():
    x = text_transforms(args.t)[None].to(torch.device(args.device))
    out = None
    h0 = None
    speaker = torch.zeros((1,)).to(device = torch.device(args.device), dtype = torch.long)
    pb = tqdm(total = 4 * 16000)
    encoding = model.encoder(x)
    for i in range(4 * 16000):
        out_probs, h0 = model.decoder(encoding, speaker, out, h0 = h0, return_state = True, return_seq = False)
        out_probs = torch.exp(out_probs).view(-1)
        out_rv = torch.distributions.Categorical(out_probs)
        out_current = out_rv.sample().view(1, 1)
        #out_current = torch.argmax(out_probs).view((1, 1))

        pb.set_postfix(current = out_current.item())
        pb.update(1)

        if out_current.item() < 1: # Stop code
            break

        if out is not None:
            out = torch.cat([out, out_current], dim = 1)
        else:
            out = out_current

    pb.close()
    out = out[0].cpu().numpy()

#text_transforms = Compose([Char2Vec(), FixLength1D(256, pad = 0), Numpy2Tensor()])
#audio_transforms = Compose([QuantiseULaw(u = 255), Normalize(1, 1), Lambda(lambda x: x.astype(np.long)), FixLength1D(4 * 16000, pad = -1, stop = 0), Numpy2Tensor()])
#joint_transforms = Lambda(lambda x: [(x[0][0], x[0][1], x[1][:-1].clamp(0, 256)), x[1]])

## Create one dataset for everything and use PyTorch samplers to do the training/validation split
#data_set = CSTRDataset(args.text_dir, args.audio_dir, text_transforms = text_transforms, audio_transforms = audio_transforms, joint_transforms = joint_transforms, sample_rate = 16000)

#x, y = data_set[1100]
#out = y

out = out.astype(np.float32)
out = expand(out - 1)
out = (out + 1.0) / 2.0 # rescale to [0.0, 1.0]
out = out * (np.iinfo(np.int16).max - np.iinfo(np.int16).min) + np.iinfo(np.int16).min
out = out.astype(np.int16)
scipy.io.wavfile.write('test.wav', 16000, out)
