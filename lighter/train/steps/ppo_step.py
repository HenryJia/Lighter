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
    update_interval: Integer
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
    def __init__(self, env, actor, critic, optimizer_policy, optimizer_value, update_interval=0, batch_size=64, epochs=1, gamma=0.99, clip=0.2, target_kl=0.01, lam=0.97, value_weight=1, entropy_weight=1e-4, epsilon=1e-5, metrics=[], use_amp=False):
        #self.env = env if type(env) is list else [env]
        self.env = env
        self.actor = actor
        self.critic = critic
        self.update_interval = update_interval
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
        self.use_amp = use_amp

        self.state_history = []
        self.policy_history = []
        self.action_history = []
        self.value_history = []
        self.reward_history = []

        #setup_pytorch_for_mpi()

        #if num_procs() > 1: # Ensure every process gets a different random seed if we are using more than 1 processes
            #seed = np.random.get_state()
            #seed += 10000 * proc_id()
            #torch.manual_seed(seed)
            #np.random.seed(seed)

        #sync_params(self.agent)


    def reset(self): # Reset our environment back to the beginning
        self.state = self.env.reset()

        self.state_history = []
        self.policy_history = []
        self.action_history = []
        self.reward_history = []
        self.value_history = []


    def __call__(self): # We don't actually need any data since the environment is a member variable
        device = next(self.actor.parameters()).device

        with torch.no_grad():
            #states = [torch.tensor(s).float() for s in self.states]
            #states = torch.stack(states, dim=0).pin_memory().to(device=device, non_blocking=True)

            ## We won't actually need the value, but dividing actor and critic would be unnecessarily complex here
            #out_distribution, value = self.agent_old(states)
            #actions = out_distribution.sample()

            #env_out = []
            #for a, e, d in zip(actions.tolist(), self.env, self.done_history[-1]):
                #if not d: # Continue the envs that are not done yet
                    #env_out += [e.step(np.array(a))]
                #else: # Otherwise stop and fill the entries with appropriate values
                    #env_out += [[np.zeros(states[0].shape), 0, True, None]]

            #next_states, rewards, done, info = list(zip(*env_out)) # Use the zip transposition trick

            state = torch.tensor(self.state).pin_memory().to(device=device, dtype=torch.float32, non_blocking=True).view(1, -1)
            out_distribution = self.actor(state)
            action = out_distribution.sample()

            next_state, reward, done, info = self.env.step(action.cpu().numpy()[0])
            self.state = next_state

            if done:
                value = torch.zeros(1, 1).pin_memory().to(device=device, dtype=torch.float32, non_blocking=True)
            else:
                value = self.critic(state)

            metrics = [('reward', reward)]

            self.state_history.append(state)
            self.policy_history.append(out_distribution)
            self.action_history.append(action)
            self.value_history.append(value)

            self.reward_history.append(reward)

        if done or len(self.action_history) == self.update_interval:
            with torch.no_grad():
                returns = []
                R = 0
                # Compute the return at each time step with discount factor
                # Fastest way to do this is to loop backwards through the rewards
                for r in self.reward_history[::-1]:
                    R = r + self.gamma * R
                    returns.insert(0, R)

                returns = torch.tensor(returns).pin_memory().to(device=device, dtype=torch.float32, non_blocking=True)
                reward_history = torch.tensor(self.reward_history).pin_memory().to(device=device, dtype=torch.float32, non_blocking=True)

                # Do generalised advantage estimation (GAE)
                if self.lam:
                    value_history = torch.cat(self.value_history, dim=0)[:, 0]
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

                # Apply the advantage normalisation
                # In OpenAI's spinning up, they compute the mean across different processes using MPI, we will leave that out for now
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)
                #returns = (returns - returns.mean()) / (returns.std() + 1e-6)

                advantage = advantage.detach()

                states = torch.cat(self.state_history, dim=0)
                actions = torch.cat(self.action_history, dim=0)
                log_prob_history = torch.cat([p.log_prob(a) for p, a in zip(self.policy_history, self.action_history)], dim=0)

            permutations = np.random.permutation(actions.shape[0])
            for i in range(self.epochs):
                j = 0
                while j < actions.shape[0]:
                    batch_size = min(actions.shape[0] - j, self.batch_size)
                    perm = permutations[j:j + batch_size]

                    out_distribution = self.actor(states[perm])

                    # This might be a tad slower, but much more numerically stable
                    ratio = torch.exp(out_distribution.log_prob(actions[perm]) - log_prob_history[perm])

                    approx_kl = (log_prob_history[perm] - out_distribution.log_prob(actions[perm])).mean().item()
                    if self.target_kl and approx_kl > 1.5 * self.target_kl:
                        print('Early stopping policy updates at approx_kl = {}, epoch {}, batch {}'.format(approx_kl, i, j))
                        break

                    policy_loss = torch.mean(torch.min(ratio * advantage[perm], torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantage[perm]))
                    entropy_loss = torch.mean(out_distribution.entropy())

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

            permutations = np.random.permutation(actions.shape[0])
            for i in range(self.epochs):
                j = 0
                while j < actions.shape[0]:
                    batch_size = min(actions.shape[0] - j, self.batch_size)
                    perm = permutations[j:j + batch_size]

                    value = self.critic(states[perm])[:, 0]

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

            self.state_history = []
            self.policy_history = []
            self.action_history = []
            self.value_history = []
            self.reward_history = []

            loss = {'policy_loss': policy_loss.item(), 'value_loss': value_loss.item()}
            return StepReport(outputs = {'out': out_distribution, 'action': action}, losses=loss, metrics=dict(metrics)), done

        loss = {'policy_loss': 0, 'value_loss': 0}
        return StepReport(outputs = {'out': out_distribution, 'action': action}, losses=loss, metrics=dict(metrics)), done

