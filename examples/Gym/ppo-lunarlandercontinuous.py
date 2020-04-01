import time, os, sys, argparse, json, copy, random, math
from queue import Queue
from collections import deque
import numpy as np
import pandas as pd
import scipy
random.seed(94103)
np.random.seed(94103)

import gym
from gym import wrappers
from gym.wrappers.monitoring.video_recorder import VideoRecorder

import torch
torch.manual_seed(94103)
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.optim import Adam

from lighter.utils.rl_utils import RingBuffer
from lighter.train.steps import PPOStep
from lighter.train.trainers import RLTrainer
from lighter.train.callbacks import ProgBarCallback

from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, default=600, help='Number of episodes to train for')
parser.add_argument('--envs', type=int, default=2, help='Number of environments to concurrently train on')
parser.add_argument('--episode_len', type=int, default=1000, help='Maximum length of an episode')
parser.add_argument('--gamma', type=float, default=0.99, help='Gamma discount factor')
parser.add_argument('--entropy_weight', type=float, default=1e-4, help='Gamma discount factor')
parser.add_argument('--policy_learning_rate', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--value_learning_rate', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--device', default='cuda:0', type=str, help='Which CUDA device to use')
args, unknown_args = parser.parse_known_args()



class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()

        self.policy_out = nn.Sequential(nn.Linear(8, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 4))


    def forward(self, x):
        mean, log_std = self.policy_out(x).chunk(2, dim=1)
        std = torch.exp(torch.clamp(log_std, -5, 2))
        return MultivariateNormal(loc=mean, scale_tril=torch.diag_embed(std))



actor = Actor()
actor = actor.to(torch.device(args.device))
critic = nn.Sequential(nn.Linear(8, 256),
                       nn.ReLU(),
                       nn.Linear(256, 256),
                       nn.ReLU(),
                       nn.Linear(256, 1))
critic = critic.to(torch.device(args.device))

optim_policy = Adam(actor.parameters(), lr=args.policy_learning_rate)
optim_value = Adam(critic.parameters(), lr=args.value_learning_rate)

env = gym.make('LunarLanderContinuous-v2')

train_step = PPOStep(env, actor, critic, optim_policy, optim_value, update_interval=0, batch_size=64, epochs=10, target_kl=None, gamma=args.gamma, entropy_weight=args.entropy_weight, clip=0.2, use_amp=False)

callbacks = [ProgBarCallback(total=args.episode_len, stateful_metrics=['policy_loss', 'value_loss', 'reward'])]

trainer = RLTrainer(train_step, callbacks, max_len=args.episode_len)

for i in range(args.episodes):
    print('Episode', i)
    next(trainer)
    if i > 500:
        train_step.clip = 0.05

total_reward_avg = 0
for i in range(100):
    state = env.reset()
    state = torch.from_numpy(state.astype(np.float32)).to(torch.device(args.device))
    total_reward = 0
    for j in range(args.episode_len):
        with torch.no_grad():
            out_distribution = actor(state.view(1, -1))
            action = out_distribution.mean.squeeze().cpu().numpy()

            next_state, reward, done, info = env.step(action)

            state = torch.from_numpy(next_state.astype(np.float32)).to(torch.device(args.device))

            total_reward += reward
            if done:
                break;

    print('Total reward', total_reward)
    total_reward_avg += total_reward

print('Total reward averaged over 100 consecutive trials', total_reward_avg / 100)

env.close()

torch.save(actor.state_dict(), 'ppo-actor.pth')
torch.save(critic.state_dict(), 'ppo-critic.pth')

