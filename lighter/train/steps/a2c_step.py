import time

import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions
from torch.optim import Adam

from apex import amp

from .steps import StepReport



class A2CStep(object):
    """
    The Advantage Actor Critic training step class

    Returns a StepReport containing the outputs of the model, the loss and the metrics

    Parameters
    ----------
    env: List of Gym environments
        This is the environment we will sample from to train
        Ideally, this should be an OpenAI Gym environment but anything which has the same API works
        We pass a list to allow concurrent running, the batch size will be this multiplied by the update interval
    agent: PyTorch model
        The A2C model we want to optimize.
        Must output 2 separate tensors, 1 being the policy, and one being the state value function
        Note, the policy should be a PyTorch distribution
    optimizer: PyTorch optimizer
        The PyTorch optimizer we're using
    update_interval: Integer
        How many steps to take before we run an update. Set to 0 (default) for update only at the end of an episode
    entropy_weight:
        Weight assigned to the entropy loss function to ensure adequate exploration
    epsilon:
        Small epsilon fuzz parameter to be used for certain ops
    gamma: Float
        Discount factor for computing the loss, default is 0.9
    metrics: List of PyTorch metrics
        A list of PyTorch metrics to apply
        These currently don't work yet
    use_amp: Boolean
        Whether to use NVidia's automatic mixed precision training
    """
    def __init__(self, env, actor, critic, optimizer_policy, optimizer_value, update_interval=0, gamma=0.9, entropy_weight=1e-4, epsilon=1e-5, metrics=[], use_amp=False):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.update_interval = update_interval
        self.optimizer_policy = optimizer_policy
        self.optimizer_value = optimizer_value
        self.gamma = gamma
        self.entropy_weight = entropy_weight
        self.epsilon = epsilon
        self.metrics = metrics
        self.use_amp = use_amp

        self.policy_history = []
        self.action_history = []
        self.value_history = []
        self.reward_history = []


    def reset(self): # Reset our environment back to the beginning
        device = next(self.actor.parameters()).device
        state = self.env.reset()
        self.state = torch.tensor(state).pin_memory().to(device=device, dtype=torch.float32, non_blocking=True).view(1, -1)


    def __call__(self): # We don't actually need any data since the environment is a member variable
        device = next(self.actor.parameters()).device

        policy = self.actor(self.state)
        action = policy.sample()

        next_state, reward, done, info = self.env.step(action.cpu().numpy()[0])

        if done:
            value = torch.zeros((1, ), device=device)
        else:
            value = self.critic(self.state)[:, 0]

        # Push this first since we want the input state to the actor to get here not the next one
        # Then update self.state
        self.policy_history.append(policy)
        self.action_history.append(action)
        self.reward_history.append(reward) # Keep rewards out of the GPU
        self.value_history.append(value)

        self.state = torch.tensor(next_state).pin_memory().to(device=device, dtype=torch.float32, non_blocking=True).view(1, -1)

        metrics = [('reward', reward)]


        if done or len(self.action_history) == self.update_interval:
            returns = []
            advantage = []
            R = 0
            # Compute the return at each time step with discount factor
            # Fastest way to do this is to loop backwards through the rewards
            for i, r in enumerate(self.reward_history[::-1]):
                R = r + self.gamma * R
                returns.insert(0, R)

            # Time dimension across dimension 0, concurrency/batch dimension across dimension 1
            returns = torch.tensor(returns).to(device=device, non_blocking=True)

            # We have to manually compute mean and std since we need to mask the env which are done
            returns = (returns - torch.mean(returns)) / (torch.std(returns) + self.epsilon)

            value = torch.cat(self.value_history, dim=0)

            advantage = (returns + value[-1] - value).detach()

            log_probs = torch.cat([p.log_prob(a) for p, a in zip(self.policy_history, self.action_history)], dim=0)

            policy_loss = -torch.mean(log_probs * advantage)
            value_loss = F.mse_loss(value, returns, reduction='mean')
            entropy_loss = torch.mean(torch.cat([p.entropy() for p in self.policy_history], dim=0))

            policy_loss = policy_loss - self.entropy_weight * entropy_loss

            self.optimizer_policy.zero_grad()
            self.optimizer_value.zero_grad()

            if self.use_amp:
                with amp.scale_loss(policy_loss, self.optimizer) as scaled_policy_loss:
                    scaled_policy_loss.backward()
                with amp.scale_loss(value_loss, self.optimizer) as scaled_value_loss:
                    scaled_value_loss.backward()
            else:
                policy_loss.backward()
                value_loss.backward()

            self.optimizer_policy.step()
            self.optimizer_value.step()

            if done:
                self.policy_history = []
                self.action_history = []
                self.value_history = []
                self.reward_history = []

            loss = {'policy_loss': policy_loss.item(), 'value_loss': value_loss.item()}
            return StepReport(outputs = {'out': policy, 'action': action}, losses=loss, metrics=dict(metrics)), done

        loss = {'policy_loss': 0, 'value_loss': 0}
        return StepReport(outputs = {'out': policy, 'action': action}, losses=loss, metrics=dict(metrics)), done

