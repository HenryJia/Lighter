import time, os, sys, argparse, json, copy, random, math
from queue import Queue
from collections import deque
import numpy as np
import pandas as pd
import scipy
random.seed(94103)
np.random.seed(94103)

import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder

import torch
torch.manual_seed(94103)
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam, RMSprop
from torchvision.transforms import Compose, Lambda
from torch.utils.data import SubsetRandomSampler

from lighter.utils.rl_utils import RingBuffer
from lighter.modules.agents import DQNAgent
from lighter.modules.agents.policies import SoftmaxPolicy
from lighter.train.steps import VPGStep
from lighter.train.trainers import RLTrainer
from lighter.train.callbacks import ProgBarCallback

from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, default=200, help='Number of episodes to train for')
parser.add_argument('--episode_len', type=int, default=500, help='Maximum length of an episode')
parser.add_argument('--gamma', type=float, default=0.9, help='Gamma discount factor')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--device', default='cuda:0', type=str, help='Which CUDA device to use')
args, unknown_args = parser.parse_known_args()


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential(nn.Linear(4, 256),
                                 nn.Tanh(),
                                 nn.Linear(256, 2),
                                 nn.Softmax(dim=1))

    def forward(self, x):
        return Categorical(self.net(x))

agent = Model().to(torch.device(args.device))

optim = Adam(agent.parameters(), lr=args.learning_rate)

env = gym.make('CartPole-v1')
recorder = VideoRecorder(env, path='./vpg-cartpole.mp4')

train_step = VPGStep(env, agent, optim, args.gamma, use_amp=False)

callbacks = [ProgBarCallback(total=args.episode_len, stateful_metrics=['loss', 'reward'])]

trainer = RLTrainer(train_step, callbacks)

for i in range(args.episodes):
    print('Episode', i)
    next(trainer)

state = env.reset()
state = torch.from_numpy(state.astype(np.float32)).to(torch.device(args.device))

total_reward = 0
for j in range(args.episode_len):
    with torch.no_grad():
        recorder.capture_frame()
        state = state
        out_distribution = agent(state.view(1, -1))
        action = torch.argmax(out_distribution.probs, dim=1).item()

        next_state, reward, done, info = env.step(action)
        total_reward += reward

        state = torch.from_numpy(next_state.astype(np.float32)).to(torch.device(args.device))

    if done:
        break

print('Total reward', total_reward)

recorder.close()
env.close()
