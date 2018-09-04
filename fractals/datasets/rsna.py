from time import time
import random, os
from collections import OrderedDict

import numpy as np
import pandas as pd
import cv2
import pydicom

import torch
from torch.utils.data import Dataset

class RSNADataset(Dataset):
    """
    Dataset for the RSNA Pneumonia Kaggle Competition

    Parameters
    ----------
    data_df: Pandas Dataframe
        Pandas dataframe containing the label csv file
    details_df: Pandas Dataframe
        Pandas dataframe containing the detailed clas info csv file
        Currently this is unused by the class
    data_dir: String
        directory containing the dcm files
    features: List
        A list of tuples containing the keyword of the feature to extract from the DICOM file and the corresponding transformation to apply
        All transformations should return a NumPy array
    y_transforms:
        A function or callable transfrom to apply to the targets specified by the label dataframe entries
        The input to this function will be the raw dataframe rows corresponding to the selected entry
        All  transformations should return a NumPy array
    """

    def __init__(self, data_df, details_df, data_dir, features, y_transforms):
        super(RSNADataset, self).__init__()
        self.data_df = data_df
        self.details_df = details_df
        self.data_dir = data_dir
        self.features = features
        self.y_transforms = y_transforms

        self.id_list = data_df['patientId'].unique().tolist()

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        x_dcm = pydicom.read_file(os.path.join(directory, id_list[idx], '.dcm')

        x = []
        for keyword, transform in self.features:
            x += [torch.from_numpy(transform(x_dcm.get('keyword')))]

        y_df = data_df.loc[data_df['patientId'] == id_list[idx]]
        y = torch.from_numpy(y_transforms(y_df))
        return x
