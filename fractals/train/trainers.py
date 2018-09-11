import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda
from torch.optim import Adam

import tqdm
from tqdm import tqdm
tqdm.monitor_interval = 0

class Trainer(object):
    """
    The class that holds everything together

    Parameters
    ----------
    loader: Finite iterator
        The loader is a finite iterator which must return a sample at each iteration
    closure: callable
        closure is a callable which executes one iteration of optimisation which called
        closure should return the statistics which should be tracked by callbacks in the form of a namedtuple
        Other formats may be used but a namedtuple should be used if possible for consistency
        closure should accept the sample as argument when called
    callbacks: list of callables
        callbacks should be a list of callable classes which handle statistics tracking and outputs of training/evaluation/prediction
        callbacks should take the output from closure, the instance of the Trainer which has it and the batch number as a member as arguments
    """
    def __init__(self, loader, closure, callbacks):
        self.loader = loader
        self.closure = closure
        self.callbacks = callbacks


    def __next__(self):
        for c in callbacks:
            c.epoch_begin()
        for i, sample in enumerate(loader):
            out = closure(sample)
            for c in callbacks:
                c(out, self, i)
        for c in callbacks:
            c.epoch_end()
