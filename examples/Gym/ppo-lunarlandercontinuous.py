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
from torch.optim import Adam, RMSprop
from torchvision.transforms import Compose, Lambda
from torch.utils.data import SubsetRandomSampler

from lighter.utils.rl_utils import RingBuffer
from lighter.train.steps import PPOStep
from lighter.train.trainers import RLTrainer
from lighter.train.callbacks import ProgBarCallback

from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, default=1024, help='Number of episodes to train for')
parser.add_argument('--envs', type=int, default=2, help='Number of environments to concurrently train on')
parser.add_argument('--episode_len', type=int, default=1000, help='Maximum length of an episode')
parser.add_argument('--gamma', type=float, default=0.995, help='Gamma discount factor')
parser.add_argument('--entropy_weight', type=float, default=1e-4, help='Gamma discount factor')
parser.add_argument('--policy_learning_rate', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--value_learning_rate', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--device', default='cuda:0', type=str, help='Which CUDA device to use')
args, unknown_args = parser.parse_known_args()



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.policy_out = nn.Sequential(nn.Linear(8, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 4))
        self.value_out = nn.Sequential(nn.Linear(8, 256),
                                      nn.ReLU(),
                                      nn.Linear(256, 256),
                                      nn.ReLU(),
                                      nn.Linear(256, 1))


    def forward(self, x):
        mean, log_std = self.policy_out(x).chunk(2, dim=1)
        std = torch.exp(torch.clamp(log_std, -5, 2))
        return MultivariateNormal(loc=mean, scale_tril=torch.diag_embed(std)), self.value_out(x)



agent = Model()
agent = agent.to(torch.device(args.device))

optim_policy = Adam(agent.policy_out.parameters(), lr=args.policy_learning_rate)
optim_value = Adam(agent.value_out.parameters(), lr=args.value_learning_rate)

env = gym.make('LunarLanderContinuous-v2')

train_step = PPOStep(env, agent, optim_policy, optim_value, update_interval=0, batch_size=64, epochs=10, gamma=args.gamma, entropy_weight=args.entropy_weight, use_amp=False)

callbacks = [ProgBarCallback(total=args.episode_len, stateful_metrics=['policy_loss', 'value_loss', 'reward'])]

trainer = RLTrainer(train_step, callbacks)

for i in range(args.episodes):
    print('Episode', i)
    next(trainer)

#env = gym.make('LunarLanderContinuous-v2')
recorder = VideoRecorder(env, path='./ppo-lunarlandercontinuous.mp4')

for i in range(100):
    state = env.reset()
    state = torch.from_numpy(state.astype(np.float32)).to(torch.device(args.device))
    for j in range(args.episode_len):
        with torch.no_grad():
            recorder.capture_frame()
            out_distribution, value = agent(state.view(1, -1))
            action = out_distribution.mean.squeeze().cpu().numpy()

            next_state, reward, done, info = env.step(action)

            state = torch.from_numpy(next_state.astype(np.float32)).to(torch.device(args.device))

            if done:
                break;

    print('Total reward', reward)

recorder.close()
env.close()

