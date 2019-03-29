from time import time
from pathlib import Path
import random, os

import numpy as np
import scipy.io.wavfile
import scipy.signal

import torch
from torch.utils.data import Dataset

from .utils.audio import load_wav

from tqdm import tqdm



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
        self.speaker_dict = set([os.path.basename(os.path.dirname(t)) for t in self.text_list])
        self.speaker_dict = dict([(s, i) for i, s in enumerate(self.speaker_dict)])


    def __getitem__(self, idx):
        text = Path(self.text_list[idx]).read_text()
        speaker = self.speaker_dict[os.path.basename(os.path.dirname(self.text_list[idx]))]
        if self.text_transforms:
            text = self.text_transforms(text)

        audio = load_wav(self.audio_list[idx], desired_sample_rate = self.sample_rate)
        if self.audio_transforms:
            audio = self.audio_transforms(audio)

        if self.joint_transforms:
            x, y = self.joint_transforms(((text, speaker), audio))
        else:
            x, y = (text, speaker), audio

        return x, y


    def __len__(self):
        return len(self.text_list)
