from time import time
from pathlib import Path
import random, os, warnings
from collections import OrderedDict

import numpy as np
import scipy.io.wavfile
import scipy.signal

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec

import torch
from torch.utils.data import Dataset

from tqdm import tqdm


def float_to_uint8(x):

    return x



def ensure_mono(raw_audio):
    """
    Just use first channel.
    """
    if raw_audio.ndim == 2:
        raw_audio = raw_audio[:, 0]
    return raw_audio



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
    x = x.astype('float32', casting='safe')
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
    audio = ensure_mono(audio)
    if desired_sample_rate:
        audio = ensure_sample_rate(desired_sample_rate, file_sample_rate, audio).astype(audio.dtype)
    audio = wav_to_float(audio)
    return audio



class CSTRDataset(Dataset):
    """
    Dataset class for the CSTR Voice Cloning Toolkit

    Note this returns (text, audio) not (audio, text) when used

    Parameters
    ----------
    text_dir: String
        Directory containing the text files
    audio_dir: String
        Directory containin the audio files
    text_transforms: callable
        Transforms to apply to the text
        All transformations should return a NumPy array
    audio_transforms:
        Transforms to apply to the audio
        All transformations should return a NumPy array
    joint_transforms: Callable
        transforms to apply to both the text and the audio simultaneously
        All joint transformations should return a list of NumPy arrays
    """

    def __init__(self, text_dir, audio_dir, text_transforms = None, audio_transforms = None, joint_transforms = None, sample_rate = None):
        super(CSTRDataset, self).__init__()
        self.text_dir = text_dir
        self.audio_dir = audio_dir

        self.text_transforms = text_transforms
        self.audio_transforms = audio_transforms
        self.joint_transforms = joint_transforms

        self.sample_rate = sample_rate

        # Sort them both so they should now match each other
        self.text_list = sorted([s for s in list(Path(text_dir).rglob("*.txt"))])
        self.audio_list = sorted([s for s in list(Path(audio_dir).rglob("*.wav"))])


    def __getitem__(self, idx):
        text = Path(self.text_list[idx]).read_text()
        if self.text_transforms:
            text = self.text_transforms(text)

        audio = load_wav(self.audio_list[idx], desired_sample_rate = self.sample_rate)
        if self.audio_transforms:
            audio = self.audio_transforms(audio)

        if self.joint_transforms:
            x, y = self.joint_transforms((text, audio))

        return x, y


    def __len__(self):
        return len(self.text_list)
