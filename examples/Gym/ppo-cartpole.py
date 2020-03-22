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
from lighter.train.steps import PPOStep
from lighter.train.trainers import RLTrainer
from lighter.train.callbacks import ProgBarCallback

from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, default=64, help='Number of episodes to train for')
parser.add_argument('--envs', type=int, default=2, help='Number of environments to concurrently train on')
parser.add_argument('--episode_len', type=int, default=500, help='Maximum length of an episode')
parser.add_argument('--gamma', type=float, default=0.99, help='Gamma discount factor')
parser.add_argument('--entropy_weight', type=float, default=1e-4, help='Gamma discount factor')
parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--device', default='cuda:0', type=str, help='Which CUDA device to use')
args, unknown_args = parser.parse_known_args()

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = nn.Sequential(nn.Linear(4, 256),
                                      nn.Tanh())
        self.policy_out = nn.Sequential(nn.Linear(256, 2),
                                        nn.Softmax(dim=1))
        self.value_out = nn.Linear(256, 1)


    def forward(self, x):
        features = self.features(x)
        return Categorical(self.policy_out(features)), self.value_out(features)



agent = Model()
agent = agent.to(torch.device(args.device))

optim = Adam(agent.parameters(), lr=args.learning_rate)

env = [gym.make('CartPole-v1') for i in range(args.envs)] # Do it concurrently
recorder = VideoRecorder(env[0], path='./dqn-cartpole.mp4')
#env = gym.make('CartPole-v1')
#recorder = VideoRecorder(env, path='./dqn-cartpole.mp4')

train_step = PPOStep(env, agent, optim, update_interval=0, batch_size=128, epochs=100, gamma=args.gamma, entropy_weight=args.entropy_weight, use_amp=False)

callbacks = [ProgBarCallback(total=args.episode_len, stateful_metrics=['loss', 'reward'])]

trainer = RLTrainer(train_step, callbacks, args.episode_len)

for i in range(args.episodes):
    print('Episode', i)
    next(trainer)

env = env[0]
state = env.reset()
state = torch.from_numpy(state.astype(np.float32)).to(torch.device(args.device))

total_reward = 0
for j in range(args.episode_len):
    with torch.no_grad():
        recorder.capture_frame()
        out_distribution, value = agent(state.view(1, -1))
        action = torch.argmax(out_distribution.probs, dim=1).item()

        next_state, reward, done, info = env.step(action)
        total_reward += reward

        state = torch.from_numpy(next_state.astype(np.float32)).to(torch.device(args.device))

    if done:
        break

print('Total reward', total_reward)

recorder.close()
env.close()

