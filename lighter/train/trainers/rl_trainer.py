from threading import Thread
from queue import Queue

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda
from torch.optim import Adam

import tqdm
from tqdm import tqdm
tqdm.monitor_interval = 0



class RLTrainer(object):
    """
    RL trainer using step and a (gym) environment

    Parameters
    ----------
    step: Callable
        step is a class which has a forward and backward member function for evaluating and training
        step.forward should return an action for the environment when called with a state as input
        step.backward should return the statistics which should be tracked by callbacks in the form of a StepReport
        step.backward should take the state, reward
        Other formats may be used but a namedtuple should be used if possible for consistency
        Note: For this RL trainer, step must also have a reset function to reset it back to the beginning of an episode
    callbacks: List of callables
        callbacks should be a list of callable classes which handle statistics tracking and outputs of training/evaluation/prediction
        callbacks should take the output from step, the instance of the SynchronousRLTrainer which has it and the batch number as a member as arguments
    episode_len: Integer
        Maximum length of an episode of training
    queue_size: Integer
        Size of the queue for the outputs to feed to callbacks
        Set to 0 for infinite length queue
        Infinite queue would mean that training is never slowed by callbacks
        But it would eat up more memory as increasing number of outputs are stored
    """
    def __init__(self, step, callbacks, episode_len=1024, queue_size=10):
        self.step = step
        self.callbacks = callbacks
        self.episode_len = episode_len
        self.queue_size = queue_size


    def train_loop(self):
        self.step.reset()
        for i in range(self.episode_len):
            report, done = self.step()
            #self.queue.put_nowait(report)
            self.queue.put(report)
            if done:
                break


    def __next__(self):
        for c in self.callbacks:
            c.epoch_begin(self)

        # Use threading so our training isn't waiting on callbacks
        self.queue = Queue(maxsize = self.queue_size)
        self.worker = Thread(target = self.train_loop)
        self.worker.setDaemon(True)
        self.worker.start()

        i = 0
        while (self.worker.is_alive() or not self.queue.empty()) and i < self.episode_len:
            out = self.queue.get()
            for c in self.callbacks:
                c(out, self, i)
            self.queue.task_done()
            i += 1
        self.queue.join()
        self.worker.join()
        for c in self.callbacks:
            c.epoch_end(self)
