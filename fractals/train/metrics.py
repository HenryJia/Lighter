import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class Accuracy(nn.Module):
    def __init__(self, dim = 1):
        super(Accuracy, self).__init__()
        self.dim = dim

    def forward(self, out, target):
        return torch.mean((torch.max(out, dim = self.dim)[1] == target).float())
