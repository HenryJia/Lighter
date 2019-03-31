import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F



class WaveNetBlock(nn.Module):
    """
    Simple Causal Convolution block from WaveNet

    Parameters
    ----------
    in_channels: Integer
        Number of input channels
    skip_channels: Integer
        Number of channels for skip connection output
        If none, then no skip output will be used
    dilation: Integer
        Dilation of the dilated causal convolution
    """
    def __init__(self, in_channels, skip_channels, return_residual = True, dilation = 1, kernel_size = 2):
        super(WaveNetBlock, self).__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.return_residual = return_residual

        # dilated convolutions has twice the channels, one for the sigmoid gate, one for the tanh
        self.dilate_conv = nn.Conv1d(in_channels, 2 * in_channels, kernel_size = kernel_size, stride = 1, dilation = dilation, padding = 0)
        nn.init.xavier_uniform_(self.dilate_conv.weight, gain = nn.init.calculate_gain('tanh'))

        if return_residual:
            self.res_conv = nn.Conv1d(in_channels, in_channels, kernel_size = 1)
            nn.init.xavier_uniform_(self.res_conv.weight, gain = nn.init.calculate_gain('linear'))

        self.skip_channels = skip_channels
        self.skip_conv = nn.Conv1d(in_channels, skip_channels, kernel_size = 1)
        nn.init.xavier_uniform_(self.skip_conv.weight, gain = nn.init.calculate_gain('relu'))


    def forward(self, x, h = None): # x is our input, h is our extra conditional variable
        inp = F.pad(x, ((self.kernel_size - 1) * self.dilation, 0), mode = 'constant', value = 0)

        output = self.dilate_conv(inp)

        if h is not None:
            output = output + h
        else:
            output = output

        t, s = torch.chunk(output, chunks = 2, dim = 1)
        gated = torch.tanh(t) * torch.sigmoid(s)

        skip = self.skip_conv(gated)

        if self.return_residual:
            return self.res_conv(gated) + x, skip
        else:
            return skip
