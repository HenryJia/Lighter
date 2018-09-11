from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda
from torch.optim import Adam

@dataclass(frozen = True)
class ClosureReport(object):
    outputs: dict
    losses: dict
    metrics: dict



class DefaultClosure(object):
    """
    The default closure class that runs basic supervised training

    Parameters
    ----------
    model: PyTorch model
        The PyTorch model we want to optimize
        Note the model should return either a PyTorch Tensor of a list of Tensors
    losses: List of PyTorch loss functions
        A list of PyTorch loss functions
        We expect 1 loss function per output of the model
        If multiple loss functions are required for an output, they can be composed into a single loss function beforehand
    optimizer: PyTorch optimizer
        The PyTorch optimizer we're using
    metrics: List of (idx, PyTorch metric)
        A list of tuples of output indexes and the PyTorch metrics we're applying to them
    train: Boolean
        Whether we are training or evaluating
    """
    def __init__(self, model, losses, optimizer, metrics, train = True):
        self.model = model
        self.losses = losses
        self.optimizer = optimizer
        self.metrics = metrics
        self.train = train


    def __call__(self, sample):
        data, targets = sample

        out = self.model(*data)

        if type(out) is torch.Tensor: # If we just have a single output
            out = [out]
        losses = [(('{}_{}'.format(c.__name__, idx), c(o, t)) for idx, (c, o, t) in enumerate(zip(self.losses, out, targets))]

        metrics = [('{}_{}'.format(m.__name__, idx), m(out[idx], targets[idx])) for (idx, m) in self.metrics]
        total_loss = sum(zip(*losses)[1]) # use the zip transposition trick to avoid having to loop manually

        if self.train:
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        out = [('output_{}'.format(idx), o) for idx, o in enumerate(out)]
        return ClosureReport(outputs = dict(out), losses = dict(losses), metrics = dict(metrics))
