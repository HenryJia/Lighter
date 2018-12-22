from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class DefaultStep(object):
    """
    The default closure class that runs basic supervised training

    When called, it will return a dictionary of {outputs, losses

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
    train: Boolean
        Whether we are training or evaluating
    """
    def __init__(self, model, losses, optimizer, train = True):
        self.model = model
        self.losses = losses
        self.optimizer = optimizer
        self.train = train


    #def unload_instance(self, sample): # Recursive unloading for each instance based on torch.utils.data.default_collate
        #if torch.is_tensor(sample):
            #return sample.detach().cpu().numpy()
        #else:
            #return [self.load_instance(s) for s in sample]


    def detach_instance(self, sample): # Recursive detaching gradients for each instance based on torch.utils.data.default_collate
        if torch.is_tensor(sample):
            return sample.detach()
        else:
            return [self.detach_instance(s) for s in sample]


    def __call__(self, engine, sample):
        data, targets = sample

        data = [data] if torch.is_tensor(data) else data
        targets = [targets] if torch.is_tensor(targets) else targets

        self.model.train(self.train) # Set the training mode
        out = self.model(*data)

        out = [out] if torch.is_tensor(out) else out

        losses = [('{}_{}'.format(l.__class__.__name__, idx), l(o, t)) for idx, (l, o, t) in enumerate(zip(self.losses, out, targets))]
        total_loss = sum(list(zip(*losses))[1]) # use the zip transposition trick to avoid having to loop manually
        losses += [('total_loss', total_loss)]

        if self.train:
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        # In case the output has some nested structure, we unload to NumPy recursively
        #out = [('output_{}'.format(idx), self.unload_instance(o)) for idx, o in enumerate(out)]

        # In case the output has some nested structure, we detach gradients recursively
        # We output the prediction and target pairs so we can pass it on to ignite metrics
        out_pairs = [('output_{}'.format(idx), self.detach_instance([o, t])) for idx, (o, t) in enumerate(zip(out, targets))]
        losses = [(name, loss.item()) for (name, loss) in losses]

        return {'out_pairs' : dict(out_pairs), 'losses' : dict(losses)}
