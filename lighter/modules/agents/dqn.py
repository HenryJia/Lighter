import copy, random, math

import numpy as np
import pandas as pd
import scipy
random.seed(94103)
np.random.seed(94103)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class DQNAgent(nn.Module):
    """
    Creates a basic actor critic DQN agent from a model

    https://arxiv.org/abs/1312.5602

    Handles the forwards and backwards passes of the model

    Parameters
    ----------
    model: PyTorch model
        The PyTorch model we want to use as our agent
    gamma: Float
        Discount factor for Q Learning
    """

    def __init__(self, model, gamma=0.99):
        super(DQNAgent, self).__init__()
        self.actor = model
        self.critic = copy.deepcopy(model)
        self.gamma = gamma


    def forward(self, state):
        return self.actor(state)


    def backward(self, experience):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        for s, a, r, n_s in experience:
            state_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_state_batch.append(n_s)

        state_batch = torch.stack(state_batch, dim=0)
        action_batch = torch.stack(action_batch, dim=0)
        reward_batch = torch.stack(reward_batch, dim=0)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_state_batch)), device=state_batch.device, dtype=torch.bool)
        next_state_batch = torch.stack([s for s in next_state_batch if s is not None], dim=0)

        q_expected = torch.zeros(state_batch.shape[0], device=state_batch.device)
        q_expected[non_final_mask] = self.critic(next_state_batch).max(dim=1)[0].detach()
        q_expected = q_expected * self.gamma + reward_batch

        q_current = self.actor(state_batch).gather(1, action_batch.view(-1, 1)).squeeze()

        loss = F.mse_loss(q_current, q_expected)
        return loss


    def parameters(self):
        return self.actor.parameters()


    def update_critic(self):
        self.critic.load_state_dict(self.actor.state_dict())
