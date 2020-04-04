import copy
from multiprocessing.pool import ThreadPool

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

from ...utils.mpi_utils import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs, setup_pytorch_for_mpi, sync_params, mpi_avg_grads



class PPOBuffer(object):
    def __init__(self, state_shape, action_shape, size, device, gamma=0.99, lam=0.97):
        self.size = size
        self.device = device
        self.gamma = gamma
        self.lam = lam
        self.state_history = torch.zeros(size, *state_shape).pin_memory().to(device=device, dtype=torch.float32, non_blocking=True)
        self.policy_history = [None] * size
        self.action_history = torch.zeros(size, *action_shape).pin_memory().to(device=device, dtype=torch.float32, non_blocking=True)
        self.value_history = torch.zeros(size).pin_memory().to(device=device, dtype=torch.float32, non_blocking=True)
        self.reward_history = torch.zeros(size).pin_memory().to(device=device, dtype=torch.float32, non_blocking=True)

        self.returns = torch.zeros(size).pin_memory().to(device=device, dtype=torch.float32, non_blocking=True)
        self.advantage = torch.zeros(size).pin_memory().to(device=device, dtype=torch.float32, non_blocking=True)
        self.log_prob_history = torch.zeros(size).pin_memory().to(device=device, dtype=torch.float32, non_blocking=True)

        self.path_start_idx = 0
        self.idx = 0


    def push(self, state, policy, action, reward, value):
        with torch.no_grad():
            self.state_history[self.idx] = state
            self.policy_history[self.idx] = policy
            self.action_history[self.idx] = action
            self.reward_history[self.idx] = reward
            self.value_history[self.idx] = value
        self.idx += 1


    def finish_episode(self):
        path_slice = slice(self.path_start_idx, self.idx)

        with torch.no_grad():
            policy_history = self.policy_history[path_slice]
            action_history = self.action_history[path_slice]
            reward_history = self.reward_history[path_slice]
            value_history = self.value_history[path_slice]

            R = 0
            # Compute the return at each time step with discount factor
            # Fastest way to do this is to loop backwards through the rewards
            returns = torch.zeros_like(reward_history)
            for i in range(returns.shape[0] - 1, -1, - 1):
                R = reward_history[i] + self.gamma * R
                returns[i] = R

            # Do generalised advantage estimation (GAE)
            if self.lam is not None:
                delta = reward_history[:-1] + self.gamma * value_history[1:] - value_history[:-1]

                A = reward_history[-1] - value_history[-1]
                advantage = torch.zeros_like(value_history)
                advantage[-1] = A
                if reward_history.shape[0] > 1:
                    for i in range(delta.shape[0] - 1, -1, -1):
                        A = delta[i] + self.gamma * self.lam * A
                        advantage[i] = A
            else:
                advantage = (returns - value_history)

            self.returns[path_slice] = returns
            self.advantage[path_slice] = advantage
            self.log_prob_history[path_slice] = torch.cat([p.log_prob(a) for p, a in zip(policy_history, action_history)], dim=0)

            self.path_start_idx = self.idx


    def get(self):
        with torch.no_grad():
            advantage = self.advantage[:self.idx]
            # Apply the advantage normalisation
            # In OpenAI's spinning up, they compute the mean across different processes using MPI, we will leave that out for now
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)
            #returns = (returns - returns.mean()) / (returns.std() + 1e-6)

            out = [self.state_history[:self.idx],
                   self.policy_history[:self.idx],
                   self.action_history[:self.idx],
                   self.reward_history[:self.idx],
                   self.value_history[:self.idx],
                   advantage,
                   self.returns[:self.idx],
                   self.log_prob_history[:self.idx]]

        self.path_start_idx = 0
        self.idx = 0

        return out


    def full(self):
        return self.idx == self.size



class PPOStep(object):
    """
    The Proximal Policy Optimisation step class

    Note: This still uses the actor critic setup like A2C

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
    optimizer: PyTorch optimizer
        The PyTorch optimizer we're using
    num_steps: Integer
        How many steps to take before we run an update. Set to 0 for update only at the end of an episode
    batch_size: Integer
        Batch size for the updates after we're done running the model
    epochs:
        Number of epochs to update for after we're done running the model
    gamma: Float
        Discount factor for computing the loss, default is 0.9
    lam: Float or None
        Lambda discount factor for generalised advantage estimation
        Set to None to not use generalised advantage esitmation
    clip: Float
        The clipping parameter of the PPO policy loss function
    value_weight:
        Weight assigned to the value loss. Default is 1
    entropy_weight
        Weight assigned to the entropy loss to ensure adequate exploration. Default is 1e-4
    epsilon:
        Small epsilon fuzz parameter to be used for certain ops
    metrics: List of PyTorch metrics
        A list of PyTorch metrics to apply
        This currently does not work yet
    use_amp: Boolean
        Whether to use NVidia's automatic mixed precision training
    """
    def __init__(self, env, actor, critic, optimizer_policy, optimizer_value, num_steps=1000, batch_size=64, epochs=1, gamma=0.99, clip=0.2, target_kl=0.01, lam=0.97, value_weight=1, entropy_weight=1e-4, epsilon=1e-5, metrics=[], update_when_done=False, use_amp=False):
        #self.env = env if type(env) is list else [env]
        self.env = env
        self.actor = actor
        self.critic = critic
        self.num_steps = num_steps
        self.batch_size=batch_size
        self.epochs=epochs
        self.optimizer_policy = optimizer_policy
        self.optimizer_value = optimizer_value
        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.target_kl = target_kl
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.epsilon = epsilon
        self.metrics = metrics
        self.update_when_done = update_when_done
        self.use_amp = use_amp

        self.buf = PPOBuffer(env.observation_space.shape, env.action_space.shape, num_steps, device=next(self.actor.parameters()).device, gamma=gamma, lam=lam)

        #setup_pytorch_for_mpi()

        #if num_procs() > 1: # Ensure every process gets a different random seed if we are using more than 1 processes
            #seed = np.random.get_state()
            #seed += 10000 * proc_id()
            #torch.manual_seed(seed)
            #np.random.seed(seed)

        #sync_params(self.agent)


    def reset(self): # Reset our environment back to the beginning
        state = self.env.reset()
        self.state = torch.tensor(state).pin_memory().to(device=self.buf.device, dtype=torch.float32, non_blocking=True).view(1, -1)


    def __call__(self): # We don't actually need any data since the environment is a member variable
        device = self.buf.device

        with torch.no_grad():
            policy = self.actor(self.state)
            action = policy.sample()

            next_state, reward, done, info = self.env.step(action.cpu().numpy()[0])

            if done:
                value = 0
            else:
                value = self.critic(self.state)

            # Push this first since we want the input state to the actor to get here not the next one
            # Then update self.state
            self.buf.push(self.state, policy, action, reward, value)
            self.state = torch.tensor(next_state).pin_memory().to(device=device, dtype=torch.float32, non_blocking=True).view(1, -1)

            metrics = [('reward', reward)]

        if self.buf.full(): # If our buffer is full then we'll finish the episode here
            done = True

        if done:
            self.buf.finish_episode()

        if not self.update_when_done: # Only update when the buffer is full
            if not self.buf.full():
                loss = {'policy_loss': 0, 'value_loss': 0}
                return StepReport(outputs={'out': policy, 'action': action}, losses=loss, metrics=dict(metrics)), done
            else:
                print('Running PPO Update')
        else: # Update when buffer is full or the episode is complete
            if not done:
                loss = {'policy_loss': 0, 'value_loss': 0}
                return StepReport(outputs={'out': policy, 'action': action}, losses=loss, metrics=dict(metrics)), done


        state_history, policy_history, action_history, reward_history, value_history, advantage, returns, log_prob_history = self.buf.get()

        permutations = torch.randperm(action_history.shape[0])
        for i in range(self.epochs):
            j = 0
            while j < action_history.shape[0]:
                batch_size = min(action_history.shape[0] - j, self.batch_size)
                perm = permutations[j:j + batch_size]

                policy = self.actor(state_history[perm])

                # This might be a tad slower, but much more numerically stable
                ratio = torch.exp(policy.log_prob(action_history[perm]) - log_prob_history[perm])

                approx_kl = (log_prob_history[perm] - policy.log_prob(action_history[perm])).mean().item()
                if self.target_kl and approx_kl > 1.5 * self.target_kl:
                    print('Early stopping policy updates at approx_kl = {}, epoch {}, batch {}'.format(approx_kl, i, j))
                    break

                policy_loss = torch.mean(torch.min(ratio * advantage[perm], torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantage[perm]))
                entropy_loss = torch.mean(policy.entropy())

                policy_loss = -policy_loss - self.entropy_weight * entropy_loss

                self.optimizer_policy.zero_grad()
                if self.use_amp:
                    with amp.scale_loss(policy_loss, self.optimizer) as scaled_policy_loss:
                        scaled_policy_loss.backward()
                else:
                    policy_loss.backward()
                self.optimizer_policy.step()

                j+=batch_size

            if self.target_kl and approx_kl > 1.5 * self.target_kl:
                break

        permutations = torch.randperm(action_history.shape[0])
        for i in range(self.epochs):
            j = 0
            while j < action_history.shape[0]:
                batch_size = min(action_history.shape[0] - j, self.batch_size)
                perm = permutations[j:j + batch_size]

                value = self.critic(state_history[perm])[:, 0]

                value_loss = F.mse_loss(returns[perm], value, reduction='mean')

                self.optimizer_value.zero_grad()
                if self.use_amp:
                    with amp.scale_loss(value_loss, self.optimizer) as scaled_value_loss:
                        scaled_value_loss.backward()
                else:
                    value_loss.backward()
                self.optimizer_value.step()

                j+=batch_size

        #with torch.no_grad():
            # Compute the metrics, this is disabled for now
            #metrics += [(m.__class__.__name__, m(out, targets).item()) for m in self.metrics]

        loss = {'policy_loss': policy_loss.item(), 'value_loss': value_loss.item()}
        return StepReport(outputs={'out': policy, 'action': action}, losses=loss, metrics=dict(metrics)), done

