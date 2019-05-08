import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ..utils import Permute
from ..wavenet import WaveNetBlock



class WaveNetModel(nn.Module):
    """
    WaveNet model for generating audio from Mel Spectrograms

    """
    def __init__(self, depth = 10, stacks = 2, res_channels = 64, skip_channels = 256, out_channels = 256, gate_channels = None,
                 embed = True, bias = False, scale = None):
        super(WaveNetModel, self).__init__()

        self.depth = depth
        self.stacks = stacks
        self.skip_channels = skip_channels
        self.scale = scale

        if embed:
            self.embed = nn.Sequential(nn.Embedding(num_embeddings = 256, embedding_dim = res_channels),
                                       Permute((0, 2, 1)))
        else:
            self.embed = nn.Conv1d(1, res_channels, kernel_size = 1)

        self.wavenet_blocks = nn.ModuleList()
        for j in range(stacks):
            for i in range(depth):
                res = False if j == stacks - 1 and i == depth - 1 else True
                self.wavenet_blocks.append(WaveNetBlock(in_channels = res_channels, skip_channels = skip_channels, gate_channels = gate_channels,
                                                        dilation = 2**i, return_residual = res))

        self.out = nn.Sequential(nn.ReLU(),
                                 nn.Conv1d(skip_channels, out_channels, kernel_size = 1, bias = bias),
                                 nn.ReLU(),
                                 nn.Conv1d(out_channels, out_channels, kernel_size = 1, bias = bias))
        nn.init.xavier_uniform_(self.out[1].weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.out[3].weight, gain = nn.init.calculate_gain('relu'))


    def forward(self, x, h):
        # Note: We expect x and y in the format of (num_batches, time_steps)
        output = self.embed(x)

        h = torch.chunk(h, self.depth * self.stacks, dim = 1)
        for i in range(len(self.wavenet_blocks) - 1):
            output, s = self.wavenet_blocks[i](output, h[i])
            if self.scale:
                output = output * self.scale
            skip = skip + s if i > 0 else s

        return self.out(skip + self.wavenet_blocks[-1](output, h[-1]))


    def get_receptive_field(self):
        return self.stacks * 2 ** self.depth - (self.stacks - 1)



class MelWaveNetModel(nn.Module):
    """
    Model for generating audio from Mel Spectrograms

    """
    def __init__(self, n_mels = 80, n_fft = 800, hops = 200, depth = 8, stacks = 2,
                 res_channels = 64, skip_channels = 128, out_channels = 256 , gate_channels = None, **kwargs):
        super(MelWaveNetModel, self).__init__()

        if gate_channels is None:
            gate_channels = res_channels

        self.bridge = nn.Sequential(nn.BatchNorm1d(n_mels, affine = True),
                                    nn.ConvTranspose1d(n_mels, n_mels, kernel_size = n_fft, stride = hops),
                                    nn.Conv1d(n_mels, stacks * depth * 2 * gate_channels, kernel_size = 1))
        nn.init.xavier_uniform_(self.bridge[1].weight, gain = nn.init.calculate_gain('tanh'))

        self.wavenet = WaveNetModel(depth = depth, stacks = stacks, res_channels = res_channels, skip_channels = skip_channels,
                                    out_channels = out_channels, gate_channels = gate_channels, **kwargs)


    def forward(self, x, h):
        output = self.bridge(h)
        if x.shape[1] < output.shape[2]:
            output = output[..., -x.shape[1]:]
        return self.wavenet(x, output)
