import torch
from torch import nn
import torch.nn.functional as F

from .callbacks import Callback



class UpdateDQNCallback(Callback):
    """
    Basic callback to update our DQN actor critic model

    Parameters
    ----------
    agent: DQNAgent
        The DQN agent we want to update
    interval: Integer
        The update interval of our critic
    """
    def __init__(self, agent, interval):
        self.interval = interval
        self.agent = agent
        self.n = 0


    def epoch_end(self, cls):
        self.n += 1
        if self.n % self.interval == 0:
            self.agent.update_critic()
