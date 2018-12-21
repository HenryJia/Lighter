import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Sequential):
    """
    Basic convolutional block as a derived class of nn.Sequential

    Parameters
    ----------
    dimensions: List/tuple of integers
        List of filter dimensions of the layers
    kernel_size: Integer
        kernel size of all the convolutional layers
    input_dimension: Integer
        Channel dimension of the input tensor
    block_num: Integer
        The block number of the current block for naming modules
    """
    def __init__(self, dimensions, kernel_size, input_dimension, block_num = 1):
        super(Block, self).__init__()
        self.dimensions = dimensions
        self.kernel_size = kernel_size
        self.input_dimension = input_dimension

        pad = kernel_size // 2
        self.add_module('Conv{}_{}'.format(block_num, 1), nn.Conv2d(input_dimension, dimensions[0], kernel_size, padding = pad))
        self.add_module('LReLU{}_{}'.format(block_num, 1), nn.LeakyReLU())
        for i in range(1, len(dimensions)):
            self.add_module('Conv{}_{}'.format(block_num, i + 1), nn.Conv2d(dimensions[i - 1], dimensions[i], kernel_size, padding = pad))
            self.add_module('LReLU{}_{}'.format(block_num, i + 1), nn.LeakyReLU())



class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Blocks going down the dimensions
        self.down = nn.ModuleList([Block([8, 8], 3, 1, 1), # downsampled x2
                                   Block([16, 16], 3, 8, 2), # downsampled x4
                                   Block([32, 32], 3, 16, 3), # downsampled x8
                                   Block([64, 64], 3, 32, 4)]) # downsampled x16

        # Blocks going up
        self.up = nn.ModuleList([Block([64, 64], 3, 64, 5), # downsampled x8
                                 Block([32, 32], 3, 64 + 32, 6), # downsampled x4
                                 Block([16, 16], 3, 32 + 16, 7), # downsampled x2
                                 Block([8, 8], 3, 16 + 8, 8)]) # original size

        self.output = nn.Sequential(nn.Conv2d(8, 1, 3, padding = 1), nn.Sigmoid())

    def forward(self, x):
        features = []
        for d in self.down:
            x = F.avg_pool2d(d(x), 2)
            features += [x]

        x = F.interpolate(self.up[0](x), scale_factor = 2, mode = 'nearest')
        for u in self.up[1:]:
            features = features[:-1]
            x = F.interpolate(u(torch.cat([x, features[-1]], dim = 1)), scale_factor = 2, mode = 'nearest')

        return self.output(x)
