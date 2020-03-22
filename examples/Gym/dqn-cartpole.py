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
from lighter.modules.agents.policies import EpsilonGreedyPolicy
from lighter.train.steps import DQNStep
from lighter.train.trainers import RLTrainer
from lighter.train.callbacks import ProgBarCallback, UpdateDQNCallback

from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, default=200, help='Number of episodes to train for')
parser.add_argument('--episode_len', type=int, default=500, help='Maximum length of an episode')
parser.add_argument('--replay_memory', type=int, default=10000, help='Maximum length of replay memory')
parser.add_argument('--batch_size', type=int, default=128, help='Model batch size during training')
parser.add_argument('--gamma', type=float, default=0.9999, help='Gamma discount factor')
parser.add_argument('--epsilon_start', type=float, default=0.9, help='Initial probability of selecting a random action')
parser.add_argument('--epsilon_decay', type=int, default=200, help='time period of decay of epsilon')
parser.add_argument('--epsilon_end', type=float, default=0.05, help='Final probability of selecting a random action')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--update_critic', type=int, default=10, help='Numbero f steps to wait before updating the critic')
parser.add_argument('--device', default='cuda:0', type=str, help='Which CUDA device to use')
args, unknown_args = parser.parse_known_args()


experiences = RingBuffer(args.replay_memory)

agent = DQNAgent(nn.Sequential(nn.Linear(4, 256),
                               nn.Tanh(),
                               nn.Linear(256, 2)),
                 gamma=args.gamma)
agent = agent.to(torch.device(args.device))
policy = EpsilonGreedyPolicy(start=args.epsilon_start, end=args.epsilon_end, t=args.epsilon_decay)

optim = Adam(agent.parameters(), lr=args.learning_rate)

env = gym.make('CartPole-v1')
recorder = VideoRecorder(env, path='./dqn-cartpole.mp4')

train_step = DQNStep(env, agent, policy, experiences, optim, args.batch_size, use_amp=False)

callbacks = [ProgBarCallback(total=args.episode_len, stateful_metrics=['reward']), UpdateDQNCallback(agent, args.update_critic)]

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
        outputs = agent(state.view(1, -1))
        action = torch.argmax(outputs, dim=1).item()

        next_state, reward, done, info = env.step(action)
        total_reward += reward

        state = torch.from_numpy(next_state.astype(np.float32)).to(torch.device(args.device))

    if done:
        break

print('Total reward', total_reward)

recorder.close()
env.close()

