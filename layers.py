import numpy as np

from torch import nn
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F

class Conv2d(nn.Conv2d):
    def reset_parameters(self):
        xavier_uniform_(self.weight.data)
        if self.bias is not None:
            self.bias.data.zero_()



class ConvTranspose2d(nn.ConvTranspose2d):
    def reset_parameters(self):
        xavier_uniform_(self.weight.data)
        if self.bias is not None:
            self.bias.data.zero_()



class Linear(nn.Linear):
    def reset_parameters(self):
        xavier_uniform_(self.weight.data)
        if self.bias is not None:
            self.bias.data.zero_()



class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)



class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape


    def forward(self, x):
        return x.view(x.size()[0], *self.shape)



class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()


    def forward(self, x):
        return F.avg_pool2d(x, x.size()[2:]).squeeze()
