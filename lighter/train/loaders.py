from threading import Thread
from queue import Queue

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader



class AsynchronousLoader(DataLoader):
    """
    Class for asynchronously loading from CPU memory to device memory

    Parameters
    ----------
    dataset: PyTorch Dataset
        The PyTorch dataset we're loading
    device: PyTorch Device
        The PyTorch device we are loading to
    pin_memory: Boolean
        Whether to use CUDA pinned memory
        Note that this should *always* be set to True for asynchronous loading to CUDA devices
        This option is only here for debugging purposes
    workers: Integer
        Number of worker processes to use for loading from storage and collating the batches in CPU memory
    queue_size: Integer
        Size of the que used to store the data loaded to the device
    """
    def __init__(self, dataset, device, queue_size = 10, pin_memory = True, workers = 10, **kwargs):
        self.dataset = dataset
        self.device = device
        self.workers = workers
        self.pin_memory = pin_memory
        self.queue_size = queue_size

        # Use PyTorch's DataLoader for collating samples and stuff since it's nicely written and parallelrised
        super(AsynchronousLoader, self).__init__(dataset, pin_memory = pin_memory, num_workers = workers, **kwargs)

        self.idx = 0


    def load_loop(self): # The loop that will load into the queue in the background
        for sample in self.cpu_iterator:
            self.queue.put(self.load_instance(sample))


    def load_instance(self, sample): # Recursive loading for each instance based on torch.utils.data.default_collate
        if torch.is_tensor(sample):
            return sample.to(self.device, non_blocking = True)
        else:
            return [self.load_instance(s) for s in sample]


    def __iter__(self):
        self.cpu_iterator = super(AsynchronousLoader, self).__iter__()
        self.idx = 0
        self.queue = Queue(maxsize = self.queue_size)
        self.worker = Thread(target = self.load_loop)
        self.worker.setDaemon(True)
        self.worker.start()
        return self


    def __next__(self):
        # If we've reached the number of batches to return or the queue is empty and the worker is dead then exit
        if (not self.worker.is_alive() and self.queue.empty()) or self.idx >= self.__len__():
            self.idx = 0
            self.queue.join()
            self.worker.join()
            raise StopIteration
        else: # Otherwise return the next batch
            out = self.queue.get()
            self.queue.task_done()
            self.idx += 1
        return out
