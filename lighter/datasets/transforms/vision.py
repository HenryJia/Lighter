from time import time
from pathlib import Path
import random, os, warnings
from collections import OrderedDict

import numpy as np

import cv2

import torch

from .core import Transform


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
            x, y, w, h = bbox[i, 0], bbox[i, 1], bbox[i, 2], bbox[i, 3]
            out[:, y:y + h, x:x + w] = 1
        return out


    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'size={0}'.format(self.size)
        format_string += ')'
        return format_string
