import numpy as np

import cv2

import torch

class Transform(object):
    """
        Base class for all transformation
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



class Resize(Transform):
    """
    Resizes NumPy images as required

    Parameters
    ----------
    size: tuple of integers
        Size to resize all input to
    interpolation: OpenCV Interpolation flag
        Which type of interpolation to use
        The default is bilinear (cv2.INTER_LINEAR)
    """
    def __init__(self, size, interpolation = cv2.INTER_LINEAR):
        self.size = size
        self.interpolation = interpolation


    def __call__(self, x):
        return cv2.resize(x, self.size, interpolation = self.interpolation)


    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'size={0}, '.format(self.size)
        format_string += 'interpolation={0}'.format(self.interpolation)
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



class Bbox2Binary(Transform):
    """
    Convert NumPy or PyTorch bounding boxes to binary mask
    Note bounding boxes must be normalized between 0 and 1

    Parameters
    ----------
    size: tuple of integers
        Size of the output
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, bbox):
        out = torch.zeros((1, ) + self.size).long()
        if bbox is None or (bbox < 0).any() or (bbox > 1).any():
            return out
        bbox[:, ::2] *= self.size[0]
        bbox[:, 1::2] *= self.size[1]
        bbox = bbox.round().long()
        for i in range(bbox.shape[0]):
            x1, y1, x2, y2 = bbox[i, 0], bbox[i, 1], bbox[i, 2], bbox[i, 3]
            out[:, y1:y2, x1:x2] = 1
        return out

