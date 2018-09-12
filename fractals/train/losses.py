import torch
from torch import nn
import torch.nn.functional as F

class RMSELoss(nn.MSELoss):
    def __init__(self, **kwargs):
        super(RMSELoss, self).__init__(**kwargs)

    def forward(self, input, target):
        return torch.sqrt(super(RMSELoss, self).forward(input, target))
