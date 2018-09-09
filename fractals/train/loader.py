import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader



class AsynchronousLoader(object):
    def __init__(self, dataset, device, batch_size = 1, shuffle = False, pin_memory = True, workers = 10, queue_size = 10):
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.workers = workers
        self.pin_memory = pin_memory
        self.queue_size = queue_size

        # Use PyTorch's DataLoader for collating samples and stuff since it's nicely written and parallelrised
        self.dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, pin_memory = pin_memory, num_workers = workers)

        self.running = False


    def load_loop(self): # The loop that will load into the queue in the background
        for sample in self.loader:
            q.put(self.load_instance(sample))


    def load_instance(self, sample): # Recursive loading for each instance based on torch.utils.data.default_collate
        if torch.is_tensor(sample):
            return sample.to(self.device, non_blocking = True)
        else:
            return [self.load_instance(s) for s in sample]


    def __iter__(self):
        assert self.idx == 0, 'An instance of AsynchronousLoader can only be run one at a time'
        self.idx = 0
        self.queue = Queue(maxsize = self.queue_size)
        self.worker = Thread(target = self.load_loop)
        self.worker.setDaemon(True)
        self.worker.start()
        return self


    def __next__(self):
        # If we've reached the number of batches to return or the queue is empty and the worker is dead then exit
        if (not self.worker.worker.is_alive() and queue.empty()) or self.idx >= len(self.dataloader):
            self.idx = 0
            self.queue.join()
            self.worker.join()
            raise StopIteration
        else: # Otherwise return the next batch
            out = self.queue.get()
            self.queue.task_done()
            self.idx += 1
        return out
