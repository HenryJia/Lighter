import random

import numpy as np

import torch
from torch import nn
from torch import distributions
import torch.nn.functional as F

class EpsilonGreedyPolicy(nn.Module):
    """
    Epsilon greedy policy with exponential decay

    start: Float
        Initial value for epsilon
    end: Float or None
        End value for epsilon
        Set to None for no decay
    t: Float or None
        Time constant for exponential decay
        Set to None for no decay
    """
    def __init__(self, start=0.9, end=None, t=None):
        self.start = start
        self.end = end
        self.t = t

        self.k = self.start - self.end


    def __call__(self, outputs):
        if self.t is not None and self.end is not None:
            epsilon = self.end + self.k
            self.k -= (self.k / self.t)
        else:
            epsilon = self.start

        if random.random() < epsilon:
            return random.randrange(outputs.shape[1])
        else:
            return torch.argmax(outputs, dim=1).item()



class SoftmaxPolicy(nn.Module):
    """
    Softmax policy

    This policy assumes that the softmax has already been applied
    """
    def __call__(self, outputs):
        return distributions.Categorical(outputs).sample().item()
