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



class SupervisedTrainer(object):
    """
    Performs supervised training and callbacks using step and loader

    Parameters
    ----------
    loader: Finite iterator
        The loader is a finite iterator which must return a sample at each iteration
    step: callable
        step is a callable which executes one iteration of optimisation when called
        step should return the statistics which should be tracked by callbacks in the form of a StepReport
        Other formats may be used but a namedtuple should be used if possible for consistency
        step should accept the sample as argument when called
    callbacks: list of callables
        callbacks should be a list of callable classes which handle statistics tracking and outputs of training/evaluation/prediction
        callbacks should take the output from step, the instance of the SupervisedTrainer which has it and the batch number as a member as arguments
    queue_size: Integer
        Size of the queue for the outputs to feed to callbacks
        Set to 0 for infinite length queue
        Infinite queue would mean that training is never slowed by callbacks
        But it would eat up more memory as increasing number of outputs are stored
    """
    def __init__(self, loader, step, callbacks, queue_size=10):
        self.loader = loader
        self.step = step
        self.callbacks = callbacks
        self.queue_size = queue_size


    def train_loop(self):
        for i, sample in enumerate(self.loader):
            self.queue.put_nowait(self.step(sample))


    def __next__(self):
        for c in self.callbacks:
            c.epoch_begin(self)

        # Use threading so our training isn't waiting on callbacks
        self.queue = Queue(maxsize = self.queue_size)
        self.worker = Thread(target = self.train_loop)
        self.worker.setDaemon(True)
        self.worker.start()

        i = 0
        while (self.worker.is_alive() or not self.queue.empty()) and i < len(self.loader):
            out = self.queue.get()
            for c in self.callbacks:
                c(out, self, i)
            self.queue.task_done()
            i += 1
        self.queue.join()
        self.worker.join()
        for c in self.callbacks:
            c.epoch_end(self)
