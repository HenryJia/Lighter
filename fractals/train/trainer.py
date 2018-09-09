import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda
from torch.optim import Adam

class Trainer(object):
    def __init__(self, loader, closure, model, callbacks_iter, callbacks_epoch):
        self.loader = loader
        self.closure = closure
        self.model = model
        self.callbacks_iter = callbacks_iter
        self.callbacks_epoch = callbacks_epoch


    def __next__(self):
        
