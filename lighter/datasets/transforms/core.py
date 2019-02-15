from time import time
from pathlib import Path
import random, os, warnings
from collections import OrderedDict

import numpy as np
import scipy.io.wavfile
import scipy.signal

import torch



class Transform(object):
    """
        Base class for all transformation
        Neither PyTorch, nor torchvision actually has a base class so we will define our own
    """
    def __repr__(self):
        return self.__class__.__name__ + '()'



class Binary2Tensor(Transform):
    """
    Convert binary to float e.g. 'M' -> 1, 'F' -> 0
    """
    def __init__(self, x_true):
        self.x_true = x_true


    def __call__(self, x):
        return torch.tensor([float(x == self.x_true)])


    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'x_true={0}'.format(self.x_true)
        format_string += ')'
        return format_string



class String2Float(Transform):
    """
    Simple class to convert a string of float to float e.g. '32' -> 32.0
    """
    def __call__(self, x):
        return torch.tensor([float(x)])



class Numpy2Tensor(Transform):
    """
    Simple class to convert a NumPy to Torch
    """
    def __call__(self, x):
        return torch.from_numpy(x)



class Reshape(Transform):
    """
    Reshapes PyTorch tensors and NumPy arrays as required

    Parameters
    ----------
    shape: tuple of integers
        Shape to reshape all input to
    """
    def __init__(self, shape):
        self.shape = shape


    def __call__(self, x):
        if isinstance(x, np.ndarray):
            return x.reshape(self.shape)
        elif torch.is_tensor(x):
            return x.view(*self.shape)


    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'shape={0}'.format(self.shape)
        format_string += ')'
        return format_string



class Permute(Transform):
    """
    Permute NumPy or PyTorch arrays/tensors as required

    Useful for changing image dimension order

    Parameters
    ----------
    dims: Tuple of integers
        Dimension order to transpose to
    """
    def __init__(self, dims):
        self.dims = dims


    def __call__(self, x):
        if torch.is_tensor(x):
            return x.permute(*self.dims)
        elif type(x) is np.ndarray:
            return np.transpose(x, self.dims)


    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'dims={0}'.format(self.dims)
        format_string += ')'
        return format_string



class Normalize(Transform):
    """
    Normalize PyTorch tensors as required

    Note: This can also be used to shift and rescale tensors as the operation is same as Gaussian normalization

    FAQ: Why don't we just use PyTorch's Normalize class?
    A: PyTorch's Normalize class officially only supports tensors of images in the format C x H x W and normalizes with a (mean, std) per channel

    Parameters
    ----------
    mean: PyTorch FloatTensor or Float
        Normalization mean
        Note that this must work with PyTorch's broadcasting rules
    std: PyTorch FloatTensor or Float
        Normalisation standard deviation
        Note that this must work with PyTorch's broadcasting rules
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std


    def __call__(self, x):
        return (x - self.mean) / self.std


    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'mean={0}, '.format(self.mean)
        format_string += 'std={0}'.format(self.std)
        format_string += ')'
        return format_string



class Int2OneHot(Transform):
    """
    Converts integer classes to one hot format

    Parameters
    ----------
    num_classes: Integer
        Number of classes
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes


    def __call__(self, x):
        output = np.zeros((np.prod(x.shape), self.num_classes), dtype = np.uint8)
        output[np.arange(np.prod(x.shape)), x.flatten()] = 1
        if x.shape[-1] == 1:
            shape = x.shape[:-1]
        else:
            shape = x.shape
        return output.reshape(shape + (self.num_classes,))


    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'num_classes={0}, '.format(self.num_classes)
        format_string += ')'
        return format_string



class FixLength1D(Transform):
    """
    Transform for padding a sequences a fixed length

    Parameters
    ----------
    length: Integer
        Length to pad everything to
    left: Bool
        Whether to pad on the left or the right side
    """
    def __init__(self, length, left = False):
        self.length = length
        self.left = left


    def __call__(self, x): # We expect inputs in the format of (timesteps, dims)
        if x.shape[0] > self.length: # If we exceed the length, then crop it
            if self.left:
                return x[-self.length:]
            else:
                return x[:self.length]
        elif x.shape[0] == self.length: # Nothing to do here, we're already at the right length
            return x
        else:
            if self.left:
                return np.append(np.zeros((self.length - x.shape[0], x.shape[1]), dtype = x.dtype), x, axis = 0)
            else:
                return np.append(x, np.zeros((self.length - x.shape[0], x.shape[1]), dtype = x.dtype), axis = 0)


    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'length={0}'.format(self.length)
        format_string += ')'
        return format_string



class JointRandomHFlip(Transform):
    """
    Simple class to do random horizontal flipping, but jointly for the target mask and the image

    Note this can be used both on images in the format of NumPy arrays or PyTorch tensors

    Parameters
    ----------
    p: Float
        Probability of flipping horizontally
    """
    def __init__(self, p):
        self.p = p


    def flip(self, data):
        if torch.is_tensor(data):
            return data.flip(2)
        elif type(data) is np.ndarray:
            return data[:, :, ::-1]
        else:
            return [self.flip(d) for d in data]


    def __call__(self, data):
        if np.random.rand() < self.p:
            return self.flip(data)
        else:
            return data


    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'p={0}'.format(self.p)
        format_string += ')'
        return format_string
