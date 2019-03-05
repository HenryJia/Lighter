from time import time
from pathlib import Path
import random, os, warnings

import numpy as np
import scipy.io.wavfile
import scipy.signal

import torch
from torch.utils.data import Dataset

from tqdm import tqdm


def ensure_sample_rate(desired_sample_rate, file_sample_rate, mono_audio):
    """
    Resample mono audio to the sample rate we want
    """
    if file_sample_rate != desired_sample_rate:
        mono_audio = scipy.signal.resample_poly(mono_audio, desired_sample_rate, file_sample_rate)
    return mono_audio



def wav_to_float(x):
    try:
        max_value = np.iinfo(x.dtype).max
        min_value = np.iinfo(x.dtype).min
    except:
        max_value = np.finfo(x.dtype).max
        min_value = np.iinfo(x.dtype).min
    x = x.astype('float32', casting = 'safe')
    x -= min_value
    x /= ((max_value - min_value) / 2.)
    x -= 1.
    return x



def load_wav(filename, desired_sample_rate = None):
    """
    Utility function to load a wav file at the sample rate we want with WaveNet quantisation
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        channels = scipy.io.wavfile.read(filename)
    file_sample_rate, audio = channels
    if audio.ndim == 2: # Make it mono
        audio = audio[:, 0]
    if desired_sample_rate:
        audio = ensure_sample_rate(desired_sample_rate, file_sample_rate, audio).astype(audio.dtype)
    audio = wav_to_float(audio)
    return audio
