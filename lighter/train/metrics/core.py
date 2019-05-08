import math

import torch
from torch import Tensor
import torch.nn as nn



class CombineLinear(nn.Module):
    """
    Simple class for combining multiple losses together into one linear

    Paramters
    ---------
    losses: list of loss functions
        List of PyTorch Loss functions to combine
    weightings: list of floats or 1D NumPy array of floats
        List of weightings to apply to each loss function
    """
    def __init__(self, losses, weights):
        super(CombineLinear, self).__init__()
        self.losses = losses
        self.weights = weights


    def forward(self, out, target):
        return sum([w * l(out, target) for w, l in zip(self.weights, self.losses)]) / sum(self.weights)



class RMSELoss(nn.MSELoss):
    """
    Simple root mean squared error loss function based on MSELoss
    """
    def __init__(self, **kwargs):
        super(RMSELoss, self).__init__(**kwargs)


    def forward(self, out, target):
        return torch.sqrt(super(RMSELoss, self).forward(out, target))



class GaussianNLLLoss(nn.Module):
    """
    Similar to MSELoss but requires a variance output as well

    Parameters
    ----------
    dim: Integer
        Dimension to split across to get mu and sigma
    is_log: Boolean
        Whether the input is log of standard deviation or the raw standard deviation
    min_simga: Float
        The minimum value to clip sigma or log_sigma to (depending on the previous parameter)
    """
    def __init__(self, dim = 1, is_log = True, min_sigma = -9.0):
        super(GaussianNLLLoss, self).__init__()
        self.dim = dim
        self.is_log = is_log
        self.min_sigma = min_sigma


    def forward(self, out, target):
        mu, sigma = torch.chunk(out, 2, dim = self.dim)
        sigma = torch.clamp(sigma, min = self.min_sigma)
        if self.is_log:
            log_sigma = sigma
            sigma = torch.exp(sigma)
        else:
            log_sigma = torch.log(sigma)

        return torch.mean(((target - mu) ** 2) / (2 * sigma ** 2) + log_sigma + 0.5 * math.log(2 * math.pi))



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
        intersection = torch.sum(target * out)
        return (2. * intersection + self.smooth) / (torch.sum(target) + torch.sum(out) + self.smooth)



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
        intersection = torch.sum(target * out)
        union = (torch.sum(target) + torch.sum(out) - intersection) # Inclusion exclusion formula
        return (intersection + self.smooth) / (union + self.smooth)



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
    one_hot: Bool
        Whether the data is in one hot format or not
    ignore_index: Integer
        Index value to ignore
    """
    def __init__(self, dim = 1, one_hot = False, ignore_index = -100):
        super(CategoricalAccuracy, self).__init__()
        self.dim = dim
        self.one_hot = one_hot
        self.ignore_index = ignore_index


    def forward(self, out, target):
        t = target if not self.one_hot else torch.argmax(target, dim = 1)
        mask =  1 - (t == self.ignore_index).float()
        return torch.sum((torch.argmax(out, dim = self.dim) == t).float() * mask) / torch.sum(mask)



class NLLLoss(nn.NLLLoss):
    """
    Identical to PyTorch's NLLLoss but can handle one hot format

    Parameters
    ----------
    one_hot: Bool
        Whether the data is in one hot format or not
    """
    def __init__(self, one_hot = False, **kwargs):
        super(NLLLoss, self).__init__(**kwargs)
        self.one_hot = one_hot

    def forward(self, out, target):
        t = target if not self.one_hot else torch.argmax(target, dim = 1)
        return super(NLLLoss, self).forward(out, t)
