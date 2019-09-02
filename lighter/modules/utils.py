import numpy as np

from torch import nn
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F



class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)



class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape


    def forward(self, x):
        return x.view(*self.shape)



class Permute(nn.Module):
    def __init__(self, *order):
        super(Permute, self).__init__()
        self.order = order


    def forward(self, x):
        return x.permute(*self.order).contiguous()



class PositionEncoding(nn.Module):
    """
    Simple Positional Encoding layer from Attention Is All You Need

    Note dimensionality of input (last dimension) must be even
    """
    def __init__(self, max_len = 10000):
        super(PositionEncoding, self).__init__()
        self.max_len = max_len

    def forward(self, x, start = 0):
        length = x.shape[1]
        dim = x.shape[2]

        with torch.no_grad():
            encoding = torch.zeros((length, dim)).to(device = x.device, dtype = x.dtype)
            pos = torch.arange(start, start + length).view(-1, 1).float() / (self.max_len ** (2 * torch.arange(dim // 2).view(1, -1) / dim)).float()
            encoding[:, ::2] = torch.sin(pos)
            encoding[:, 1::2] = torch.cos(pos)

        return x + encoding
