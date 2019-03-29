import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from .utils.audio import load_wav



class LJSpeechDataset(Dataset):
    """
    Dataset class for the LJSpeech Dataset

    Note this returns (text, audio) not (audio, text) when used

    Parameters
    ----------
    csv_dir: String
        Directory containing of the metadata csv
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

    def __init__(self, csv_dir, audio_dir, text_transforms = None, audio_transforms = None, joint_transforms = None, sample_rate = None):
        super(LJSpeechDataset, self).__init__()
        self.csv_dir = csv_dir
        self.audio_dir = audio_dir
        self.data_df = pd.read_csv(csv_dir, sep = '|', header = None, usecols = [0, 2], names = ['wav', 'txt'])

        self.text_transforms = text_transforms
        self.audio_transforms = audio_transforms
        self.joint_transforms = joint_transforms

        self.sample_rate = sample_rate


    def __getitem__(self, idx):
        text = self.data_df.iloc[idx]['txt']

        if self.text_transforms:
            text = self.text_transforms(text)

        fn = os.path.join(self.audio_dir, self.data_df.iloc[idx]['wav'] + '.wav')
        audio = load_wav(fn, desired_sample_rate = self.sample_rate)
        if self.audio_transforms:
            audio = self.audio_transforms(audio)

        if self.joint_transforms:
            x, y = self.joint_transforms((text, audio))
        else:
            x, y = text, audio

        return x, y


    def __len__(self):
        return len(self.data_df)
