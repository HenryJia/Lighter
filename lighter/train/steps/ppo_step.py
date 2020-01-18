import copy

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
    train: Boolean
        Whether we are training
    use_amp: Boolean
        Whether to use NVidia's automatic mixed precision training
    """
    def __init__(self, env, agent, optimizer, update_interval=0, batch_size=128, epochs=1, gamma=0.99, clip=0.2, value_weight=1, entropy_weight=1e-4, epsilon=1e-5, metrics=[], train=True, use_amp=False):
        self.env = env if type(env) is list else [env]
        self.agent = agent
        self.update_interval = update_interval
        self.batch_size=batch_size
        self.epochs=epochs
        self.optimizer = optimizer
        self.gamma = gamma
        self.clip = clip
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.epsilon = epsilon
        self.metrics = metrics
        self.train = train
        self.use_amp = use_amp

        self.state_history = []
        self.policy_history = []
        self.action_history = []
        self.done_history = [[False] * len(self.env)]
        self.value_history = []
        self.reward_history = []


    def reset(self): # Reset our environment back to the beginning
        self.states = [e.reset() for e in self.env]

        # Create a new instance of our agent to pre-allocate the memory
        # This is slightly faster than using deepcopy every time
        self.agent_old = copy.deepcopy(self.agent)

        self.state_history = []
        self.policy_history = []
        self.action_history = []
        self.done_history = [[False] * len(self.env)]
        self.reward_history = []


    def __call__(self): # We don't actually need any data since the environment is a member variable
        device = next(self.agent.parameters()).device
        assert next(self.agent_old.parameters()).device == device, 'self.agent and self.agent_old must be on the same device, make sure reset is run straight before __call__'

        states = [torch.tensor(s) for s in self.states]
        states = torch.stack(states, dim=0).pin_memory().to(device=device, dtype=torch.float32, non_blocking=True)

        # We won't actually need the value, but dividing actor and critic would be unnecessarily complex here
        out_distribution, value = self.agent_old(states)
        actions = out_distribution.sample()

        env_out = []
        for a, e, d in zip(actions.tolist(), self.env, self.done_history[-1]):
            if not d: # Continue the envs that are not done yet
                env_out += [e.step(a)]
            else: # Otherwise stop and fill the entries with appropriate values
                env_out += [[np.zeros(states[0].shape), 0, True, None]]

        next_states, rewards, done, info = list(zip(*env_out)) # Use the zip transposition trick
        self.states = next_states

        metrics = [('reward', np.mean(rewards))]

        if self.train:
            self.state_history.append(states)
            self.policy_history.append(out_distribution)
            self.action_history.append(actions)

            self.done_history.append(done)
            # Keep rewards out of the GPU, we have to loop through and CPU ops should be a little faster
            self.reward_history.append(torch.tensor(rewards))

            if all(done) or len(self.action_history) == self.update_interval:
                returns = []
                R = 0
                # Compute the return at each time step with discount factor
                # Fastest way to do this is to loop backwards through the rewards
                for r in self.reward_history[::-1]:
                    R = r + self.gamma * R
                    returns.insert(0, R)

                # Time dimension across dimension 0, concurrency/batch dimension across dimension 1
                returns = torch.stack(returns, dim=0).pin_memory().to(device=device, dtype=torch.float32, non_blocking=True)

                # Note: the indexing shifts it by one so we're not masking out the final state as the gym env will return done on the final state
                done_mask = 1 - torch.tensor(self.done_history[:-1]).pin_memory().to(device=device, dtype=torch.float32, non_blocking=True)

                # We have to manually compute mean and std since we need to mask the envs which are done
                returns = (returns - torch.sum(returns * done_mask, dim=0) / (torch.sum(done_mask, dim=0) + self.epsilon))
                if returns.shape[0] > 1: # PyTorch returns NaN if you try taking std of an array of size 1
                    returns /= (torch.sum(returns ** 2 * done_mask, dim=0) / (torch.sum(done_mask, dim=0) - 1 + self.epsilon) + self.epsilon)

                states = torch.cat(self.state_history, dim=0)
                actions = torch.cat(self.action_history, dim=0)
                log_prob_history = torch.cat([p.log_prob(a) for p, a in zip(self.policy_history, self.action_history)], dim=0).detach()

                # Note: flatten costs no time as no memory actually gets copied
                done_mask = done_mask.flatten()
                returns = returns.flatten()

                for i in range(self.epochs):
                    j = 0
                    while j < actions.shape[0]:
                        batch_size = min(actions.shape[0] - j, self.batch_size)

                        out_distribution, value = self.agent(states[j:j + batch_size])
                        value = value.squeeze()

                        advantage = (returns[j:j + batch_size] - value).detach()

                        # This is a tad slower, but much more numerically stable
                        ratio = torch.exp(out_distribution.log_prob(actions[j:j + batch_size]) - log_prob_history[j:j + batch_size])

                        policy_loss = torch.min(ratio * advantage, torch.clamp(ratio, 1 - self.clip, 1 + self.clip))
                        value_loss = F.smooth_l1_loss(returns[j:j + batch_size], value, reduction='none')
                        entropy_loss = out_distribution.entropy()

                        loss = torch.mean(done_mask[j:j + batch_size] * (-policy_loss + self.value_weight * value_loss - self.entropy_weight * entropy_loss))

                        self.optimizer.zero_grad()
                        if self.use_amp:
                            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()

                        self.optimizer.step()

                        j+=batch_size

                #with torch.no_grad():
                    # Compute the metrics, this is disabled for now
                    #metrics += [(m.__class__.__name__, m(out, targets).item()) for m in self.metrics]

                self.agent_old.load_state_dict(self.agent.state_dict())

                self.state_history = []
                self.policy_history = []
                self.action_history = []
                self.done_history = [self.done_history[-1]]
                self.value_history = []
                self.reward_history = []

                return StepReport(outputs = {'out': out_distribution, 'action': actions}, losses={'loss': loss.item()}, metrics=dict(metrics)), all(done)

        return StepReport(outputs = {'out': out_distribution, 'action': actions}, losses={'loss': 0}, metrics=dict(metrics)), all(done)

