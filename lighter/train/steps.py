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

    Returns a StepReport containing the outputs of the model, the losses and the metrics

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
    use_amp: Boolean
        Whether to use NVidia's automatic mixed precision training
    """
    def __init__(self, model, losses, optimizer, metrics = [], train = True, use_amp = False):
        self.model = model
        self.losses = losses
        self.optimizer = optimizer
        self.metrics = metrics
        self.train = train
        self.use_amp = use_amp


    def unload_instance(self, sample): # Recursive unloading for each instance based on torch.utils.data.default_collate
        if torch.is_tensor(sample):
            # TODO: Refactor to remove this, since unloading from GPU cannot be done asynchronously in PyTorch
            return sample.detach().cpu().numpy()
        else:
            return [self.load_instance(s) for s in sample]


    def __call__(self, sample):
        data, targets = sample

        data = [data] if torch.is_tensor(data) else data
        targets = [targets] if torch.is_tensor(targets) else targets

        out = self.model(*data)

        out = [out] if torch.is_tensor(out) else out

        out = [o.float() if o.dtype == torch.float16 else o for o in out]

        losses = [('{}_{}'.format(l.__class__.__name__, idx), l(o, t)) for idx, (l, o, t) in enumerate(zip(self.losses, out, targets))]
        total_loss = sum(list(zip(*losses))[1]) # use the zip transposition trick to avoid having to loop manually
        losses += [('total_loss', total_loss)]

        if self.train:
            self.optimizer.zero_grad()
            if self.use_amp:
                with amp.scale_loss(total_loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                total_loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            # Compute the metrics
            metrics = [('{}_{}'.format(m.__class__.__name__, idx), m(out[idx], targets[idx]).item()) for (idx, m) in self.metrics]

            # In case the output has some nested structure, we unload to NumPy recursively
            out = [('output_{}'.format(idx), self.unload_instance(o)) for idx, o in enumerate(out)]

            # Use .items to get just the number instead of converting to numpy
            losses = [(name, loss.item()) for (name, loss) in losses]

        return StepReport(outputs = dict(out), losses = dict(losses), metrics = dict(metrics))
