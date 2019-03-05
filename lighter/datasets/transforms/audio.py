from time import time
from pathlib import Path
import random, os, warnings
from collections import OrderedDict

import numpy as np
import scipy.io.wavfile
import scipy.signal

import torch

from tqdm import tqdm

from .core import Transform



class QuantiseULaw(Transform):
    """
    Apply the quantisation mechanism in the WaveNet paper

    Parameters
    ----------
    u: Integer
        u parameter in the WaveNet quantisation trick. Default is 255 like the paper
    """
    def __init__(self, u = 255):
        self.u = u


    def __call__(self, x):
        out = np.sign(x) * (np.log(1 + self.u * np.abs(x)) / np.log(1 + self.u)) # Apply the formula
        out += 1. # Rescale to 0 to 255 and cast to uint8
        out /= 2.
        out *= np.iinfo(np.uint8).max
        out = out.astype(np.uint8)

        return out



class ExpandULaw(Transform):
    """
    Apply the inverse transformation of the WaveNet quantisation for decompressing

    Example use:

    x, y = data_set[0]
    print(data_set.audio_list[0])
    y = y.numpy()
    print(np.max(y), np.min(y))
    expand = ExpandULaw()
    y = expand(y)
    print(np.max(y), np.min(y))
    y = (y + 1.0) / 2.0 # rescale to [0.0, 1.0]
    y = y * (np.iinfo(np.int16).max - np.iinfo(np.int16).min) + np.iinfo(np.int16).min
    y = y.astype(np.int16)
    scipy.io.wavfile.write('test.wav', 48000, y)
    exit()

    Parameters
    ----------
    u: Integer
        u parameter in the WaveNet quantisation trick. Default is 255 like the paper

    """
    def __init__(self, u = 255):
        self.u = u


    def __call__(self, x):
        out = x.astype(np.float32) # Cast back to uint8 and rescale to (-1, 1)
        out /= (np.iinfo('uint8').max / 2) # This takes it to (0, 2)
        out -= 1.0 # (This takes it to (-1, 1)
        return np.sign(out) * ((1 + self.u) ** np.abs(out) - 1) / self.u
