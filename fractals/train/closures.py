from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
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
        We use a list of tuple pairs rather than a dictionary as to allow multiple metrics to be applied to the same output
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

        data = [data] if torch.is_tensor(data) else data
        targets = [targets] if torch.is_tensor(targets) else targets

        out = self.model(*data)

        out = [out] if torch.is_tensor(out) else out

        losses = [('{}_{}'.format(l.__class__.__name__, idx), l(o, t)) for idx, (l, o, t) in enumerate(zip(self.losses, out, targets))]
        total_loss = sum(list(zip(*losses))[1]) # use the zip transposition trick to avoid having to loop manually
        losses += [('total_loss', total_loss)]

        if self.train:
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        metrics = [('{}_{}'.format(m.__class__.__name__, idx), m(out[idx], targets[idx]).detach().cpu().numpy()) for (idx, m) in self.metrics]
        out = [('output_{}'.format(idx), o.detach().cpu().numpy()) for idx, o in enumerate(out)]
        losses = [(name, loss.detach().cpu().numpy()) for (name, loss) in losses]

        return ClosureReport(outputs = dict(out), losses = dict(losses), metrics = dict(metrics))
