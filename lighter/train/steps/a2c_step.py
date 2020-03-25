from multiprocessing import Process, Lock, Pipe
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
from ...modules.agents.policies import SoftmaxPolicy

class EnvPool(object):
    def __init__(self, envs):
        self.envs = envs

        # I would like to use threading instead of multiprocessing but Python's GIL is a bitch
        self.pipes = [Pipe() for i in range(len(envs))]
        self.workers = [Process(target=self.run_env, args=(i, e, self.pipes[i][1])) for i, e in enumerate(envs)]
        self.start = [Lock() for i in range(len(envs))]
        self.finish = [Lock() for i in range(len(envs))]

        self.done = [False] * len(self.envs)

        # Lock it intially so we don't start running
        for s, f, w in zip(self.start, self.finish, self.workers):
            s.acquire()
            f.acquire()
            w.start()


    def run(self, actions, shape): # We need the shape so we can fill done environments with 0s
        self.actions = actions
        # Release our lock to run all our workers in parallel

        for i, p in enumerate(self.pipes):
            if not self.done[i]:
                p[0].send(self.actions[i])

        for i, s in enumerate(self.start):
            if not self.done[i]:
                s.release()

        # Wait for workers to finish the iteration
        for i, f in enumerate(self.finish):
            if not self.done[i]:
                f.acquire()

        out = []
        for i, p in enumerate(self.pipes):
            if not self.done[i]:
                out += [p[0].recv()]
            else:
                out += [[np.zeros(shape, dtype=np.float32), 0, True, None]]

        self.done = [o[2] for o in out]

        return out


    def run_env(self, idx, env, pipe):
        done = False
        while not done:
            self.start[idx].acquire() # Wait for our main thread to be ready
            action = pipe.recv()
            state, rewards, done, info = env.step(action)
            pipe.send([state, rewards, done, info])
            self.finish[idx].release() # Signal to main thread we're done for the iteration


    def __del__(self):
        for w in self.workers:
            w.join() # If we destroy the pool wait for all workers to join
            w.close()


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
    def __init__(self, envs, agent, optimizer, update_interval=0, gamma=0.9, entropy_weight=1e-4, epsilon=1e-5, metrics=[], use_amp=False):
        self.envs = envs if type(envs) is list else [envs]
        self.agent = agent
        self.update_interval = update_interval
        self.optimizer = optimizer
        self.gamma = gamma
        self.entropy_weight = entropy_weight
        self.epsilon = epsilon
        self.metrics = metrics
        self.use_amp = use_amp

        self.done_history = [[False] * len(self.envs)] # We can't be done on the initial state
        self.value_history = []
        self.reward_history = []


    def reset(self): # Reset our environment back to the beginning
        self.states = [e.reset() for e in self.envs]

        self.policy_history = []
        self.action_history = []
        self.done_history = [[False] * len(self.envs)]
        self.value_history = []
        self.reward_history = []

        if hasattr(self, 'pool'):
            del self.pool
        self.pool = EnvPool(self.envs)


    def __call__(self): # We don't actually need any data since the environment is a member variable
        device = next(self.agent.parameters()).device

        #t0 = time.time()
        states = [torch.tensor(s).float() for s in self.states]
        states = torch.stack(states, dim=0).pin_memory().to(device=device, non_blocking=True)
        #print('states2gpu', time.time() - t0)

        #t0 = time.time()
        out_distribution, value = self.agent(states)
        actions = out_distribution.sample()
        #print('agent', time.time() - t0)

        #t0 = time.time()
        #env_out = []        print('no update', all(done))

        #for a, e, d in zip(actions.tolist(), self.envs, self.done_history[-1]):
            #if not d: # Continue the envs that are not done yet
                #env_out += [e.step(a)]
            #else: # Otherwise stop and fill the entries with appropriate values
                #env_out += [[np.zeros(states[0].shape), 0, True, None]]
        env_out = self.pool.run([a[0] for a in np.split(actions.cpu().numpy(), len(self.envs))], states[0].shape)
        #print('env', time.time() - t0)


        #t0 = time.time()
        next_states, rewards, done, info = list(zip(*env_out)) # Use the zip transposition trick
        self.states = next_states
        #print('reorg', time.time() - t0)

        #t0 = time.time()
        metrics = [('reward', np.mean(np.array(rewards)[~np.array(self.done_history[-1])]))]

        self.policy_history.append(out_distribution)
        self.action_history.append(actions)
        self.value_history.append(value[:, 0])

        self.done_history.append(done)
        # Keep rewards out of the GPU, we have to loop through and CPU ops should be a little faster
        self.reward_history.append(torch.tensor(rewards).float())
        #print('post', time.time() - t0)

        if all(done) or len(self.action_history) == self.update_interval:
            returns = []
            R = 0
            # Compute the return at each time step with discount factor
            # Fastest way to do this is to loop backwards through the rewards
            for r in self.reward_history[::-1]:
                R = r + self.gamma * R
                returns.insert(0, R)

            # Time dimension across dimension 0, concurrency/batch dimension across dimension 1
            returns = torch.stack(returns, dim=0).to(device=device, non_blocking=True)

            # Note: the indexing shifts it by one so we're not masking out the final state as the gym env will return done on the final state
            done_mask = 1 - torch.tensor(self.done_history[:-1]).to(device=device, dtype=torch.float32, non_blocking=True)

            # We have to manually compute mean and std since we need to mask the envs which are done
            returns = (returns - torch.sum(returns * done_mask, dim=0) / (torch.sum(done_mask, dim=0) + self.epsilon))
            if returns.shape[0] > 1: # PyTorch returns NaN if you try taking std of an array of size 1
                returns /= (torch.sum(returns ** 2 * done_mask, dim=0) / (torch.sum(done_mask, dim=0) - 1 + self.epsilon) + self.epsilon)

            value = torch.stack(self.value_history, dim=0)

            advantage = done_mask * (returns - value).detach()

            log_probs = torch.stack([p.log_prob(a) for p, a in zip(self.policy_history, self.action_history)], dim=0)

            policy_loss = -log_probs * advantage
            value_loss = F.smooth_l1_loss(value, returns, reduction='none')
            entropy_loss = torch.stack([p.entropy() for p in self.policy_history], dim=0)
            loss = torch.mean(done_mask * (policy_loss + value_loss - self.entropy_weight * entropy_loss))

            self.optimizer.zero_grad()
            if self.use_amp:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward(retain_graph=True)

            self.optimizer.step()

            #with torch.no_grad():
                # Compute the metrics # This currently does nothing
                #metrics += [(m.__class__.__name__, m(out, targets).item()) for m in self.metrics]

            if all(done):
                self.policy_history = []
                self.action_history = []
                self.done_history = [self.done_history[-1]] # Keep the last bit so we still know if the environemtn is done
                self.value_history = []
                self.reward_history = []

            return StepReport(outputs = {'out': out_distribution, 'action': actions}, losses={'loss': loss.item()}, metrics=dict(metrics)), all(done)

        return StepReport(outputs = {'out': out_distribution, 'action': actions}, losses={'loss': 0}, metrics=dict(metrics)), all(done)

