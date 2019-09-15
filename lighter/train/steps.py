from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from apex import amp



@dataclass(frozen = True)
class StepReport(object):
    outputs: dict
    losses: dict
    metrics: dict

class DefaultStep(object):
    """
    The default step class that runs basic supervised training

    This is pretty much the most basic possible step. It just handles runnign the model and applying the loss and metrics

    Returns a StepReport containing the outputs of the model, the loss and the metrics

    Parameters
    ----------
    model: PyTorch model
        The PyTorch model we want to optimize
    loss: PyTorch loss function
        A PyTorch loss function to apply
    optimizer: PyTorch optimizer
        The PyTorch optimizer we're using
    metrics: List of PyTorch metrics
        A list of PyTorch metrics to apply
    train: Boolean
        Whether we are training or evaluating
    use_amp: Boolean
        Whether to use NVidia's automatic mixed precision training
    """
    def __init__(self, model, loss, optimizer, metrics = [], train = True, use_amp = False):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.train = train
        self.use_amp = use_amp


    def __call__(self, sample):
        data, targets = sample

        data = [data] if torch.is_tensor(data) else data

        out = self.model(*data)
        loss = self.loss(out, targets)

        if self.train:
            self.optimizer.zero_grad()
            if self.use_amp:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            # Compute the metrics
            metrics = [(m.__class__.__name__, m(out, targets).item()) for m in self.metrics]

        return StepReport(outputs = {'out': out.detach()}, losses = {'loss': loss.item()}, metrics = dict(metrics))
