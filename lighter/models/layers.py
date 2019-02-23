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



class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()


    def forward(self, x):
        return F.avg_pool2d(x, x.size()[2:]).squeeze()
