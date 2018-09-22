import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F



class RMSELoss(nn.MSELoss):
    """
    Simple root mean squared error loss function based on MSELoss
    """
    def __init__(self, **kwargs):
        super(RMSELoss, self).__init__(**kwargs)


    def forward(self, out, target):
        return torch.sqrt(super(RMSELoss, self).forward(out, target))



class F1Metric(nn.Module):
    """
    Differentiable F1 Metric (also known as dice coefficient)

    Note this is only for binary classification

    Parameters
    ----------
    smooth: float
        Parameter for smoothing the loss function and keeping it differentiable everywhere
    """
    def __init__(self, smooth = 1):
        super(F1Metric, self).__init__()
        self.smooth = smooth


    def forward(self, out, target):
        dims = list(range(1, len(out.shape))) # Dimensions to sum over. We use all but the batch dimension

        intersection = torch.sum(target * out, dim = dims)
        return torch.mean((2. * intersection + self.smooth) / (torch.sum(target, dim = dims) + torch.sum(out, dim = dims) + self.smooth))



class F1Loss(F1Metric):
    """
    Differentiable F1 Loss

    This is exactly identical to F1Metric but we negate the result in order to optimise for the minimum

    To keep things positive we use 1 - x rather than - x
    """
    def __init__(self, **kwargs):
        super(F1Loss, self).__init__(**kwargs)


    def forward(self, out, target):
        return 1 - super(F1Loss, self).forward(out, target)



class IOUMetric(nn.Module):
    """
    Differentiable intersection over union metric

    Note this is only for binary classification

    Parameters
    ----------
    smooth: float
        Parameter for smoothing the loss function and keeping it differentiable everywhere
    """
    def __init__(self, smooth = 1):
        super(IOUMetric, self).__init__()
        self.smooth = smooth


    def forward(self, out, target):
        dims = list(range(1, len(out.shape))) # Dimensions to sum over. We use all but the batch dimension

        intersection = torch.sum(target * out, dim = dims)
        union = (torch.sum(target, dim = dims) + torch.sum(out, dim = dims) - intersection) # Inclusion exclusion formula
        return torch.mean((intersection + self.smooth) / (union + self.smooth))



class IOULoss(IOUMetric):
    """
    Differentiable intersection over union loss

    This is exactly identical to IOUMetric but we negate the result in order to optimise for the minimum

    To keep things positive we use 1 - x rather than - x
    """
    def __init__(self, **kwargs):
        super(IOULoss, self).__init__(**kwargs)


    def forward(self, out, target):
        return 1 - super(IOULoss, self).forward(out, target)



class BinaryAccuracy(nn.Module):
    """
    Binary accuracy metric (not differentiable)
    """
    def __init__(self):
        super(BinaryAccuracy, self).__init__()

    def forward(self, out, target):
        return torch.mean((torch.round(out) == target).float())



class CategoricalAccuracy(nn.Module):
    """
    Categorical accuracy metric (not differentiable)

    Parameters
    ----------
    dim: integer
        The dimension to calculate the accuracy across
    """
    def __init__(self, dim = 1):
        super(CategoricalAccuracy, self).__init__()
        self.dim = dim

    def forward(self, out, target):
        return torch.mean((torch.max(out, dim = self.dim)[1] == target).float())
