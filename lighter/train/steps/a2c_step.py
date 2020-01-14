import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions
from torch.optim import Adam

from apex import amp

from .steps import StepReport
from ...modules.agents.policies import SoftmaxPolicy



class A2CStep(object):
    """
    The Advantage Actor Critic step class

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
    update_interval: Integer
        How many steps to take before we run an update. Set to 0 for update only at the end of an episode
    optimizer: PyTorch optimizer
        The PyTorch optimizer we're using
    gamma: Float
        Discount factor for computing the loss, default is 0.9
    metrics: List of PyTorch metrics
        A list of PyTorch metrics to apply
    train: Boolean
        Whether we are training
    use_amp: Boolean
        Whether to use NVidia's automatic mixed precision training
    """
    def __init__(self, env, agent, optimizer, update_interval=0, gamma=0.9, epsilon=1e-5, metrics=[], train=True, use_amp=False):
        self.env = env if type(env) is list else [env]
        self.agent = agent
        self.update_interval = update_interval
        self.optimizer = optimizer
        self.gamma = gamma
        self.epsilon = epsilon
        self.metrics = metrics
        self.train = train
        self.use_amp = use_amp

        self.probs = []
        self.log_probs = [] # Note this is the log likelihood of the actions, couldn't think of a better name
        self.value_history = []
        self.reward_history = []


    def reset(self): # Reset our environment back to the beginning
        self.states = [e.reset() for e in self.env]


    def __call__(self): # We don't actually need any data since the environment is a member variable
        device = next(self.agent.parameters()).device

        states = [torch.tensor(s) for s in self.states]
        states = torch.stack(states, dim=0).pin_memory().to(device=device, dtype=torch.float32, non_blocking=True)

        out, value = self.agent(states)
        actions = distributions.Categorical(out).sample().tolist()

        env_out = [e.step(a) for a, e in zip(actions, self.env)]
        next_states, rewards, done, info = list(zip(*env_out)) # Use the zip transposition trick
        self.states = next_states

        metrics = [('reward', np.mean(rewards))]
        done = any(done) # We're done if any of the environments are done

        if self.train:
            self.probs.append(out)
            self.log_probs.append(torch.log(out[torch.arange(len(actions)), actions]))
            self.value_history.append(value[:, 0])

            #reward = torch.tensor(reward).to(device=device, dtype=torch.float32, non_blocking=True)
            # Keep rewards out of the GPU, we have to loop through and CPU ops should be a little faster for now
            self.reward_history.append(torch.tensor(rewards))

            if done or len(self.probs) == self.update_interval:
                returns = []
                R = 0
                # Compute the return at each time step with discount factor
                # Fastest way to do this is to loop backwards through the rewards
                for r in self.reward_history[::-1]:
                    R = r + self.gamma * R
                    returns.insert(0, R)

                # Time dimension across dimension 0, concurrency/batch dimension across dimension 1
                returns = torch.stack(returns, dim=0).to(device=device, dtype=torch.float32, non_blocking=True)
                returns = (returns - returns.mean(dim=0, keepdims=True))
                if returns.shape[0] > 1: # PyTorch returns NaN if you try taking std of an array of size 1
                    returns /= (returns.std(dim=0, keepdims=True) + self.epsilon)

                value = torch.stack(self.value_history, dim=0).detach()

                advantage = returns - value

                log_probs = torch.stack(self.log_probs, dim=0)
                probs = torch.stack(self.probs, dim=0)

                policy_loss = -torch.sum(log_probs * advantage)
                value_loss = F.smooth_l1_loss(value, returns)
                entropy_loss = -torch.mean(torch.sum(probs * torch.log(probs), dim=-1))
                loss = policy_loss + value_loss - 1e-4 * entropy_loss
                self.optimizer.zero_grad()
                if self.use_amp:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                self.optimizer.step()

                with torch.no_grad():
                    # Compute the metrics
                    metrics += [(m.__class__.__name__, m(out, targets).item()) for m in self.metrics]

                self.probs = []
                self.log_probs = []
                self.value_history = []
                self.reward_history = []

                return StepReport(outputs = {'out': out.detach(), 'action': actions}, losses={'loss': loss.item()}, metrics=dict(metrics)), done

        return StepReport(outputs = {'out': out.detach(), 'action': actions}, losses={'loss': 0}, metrics=dict(metrics)), done

