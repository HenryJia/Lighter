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
        out *= np.iinfo('uint8').max
        out = out.astype('uint8')

        return out



def ExpandULaw(Transform):
    """
    Apply the inverse transformation of the WaveNet quantisation for decompressing

    Parameters
    ----------
    u: Integer
        u parameter in the WaveNet quantisation trick. Default is 255 like the paper
    """
    def __init__(self, u = 255):
        self.u = u


    def __call__(self, x):
        out = x.astype('np.float32') # Cast back to uint8 and rescale to (-1, 1)
        out /= (np.iinfo('uint8').max / 2)
        out -= 1.0
        return np.sign(x) * ((1 + self.u) ** np.abs(x) - 1) / self.u
