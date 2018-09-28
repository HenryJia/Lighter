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

    def __init__(self, data_df, data_dir, features, y_transforms, joint_transforms = None):
        super(RSNADataset, self).__init__()
        self.data_df = data_df
        self.data_dir = data_dir
        self.features = features
        self.y_transforms = y_transforms
        self.joint_transforms = joint_transforms

        self.id_list = data_df['patientId'].unique().tolist()


    def __len__(self):
        return len(self.id_list)


    def __getitem__(self, idx):
        x_dcm = pydicom.read_file(os.path.join(self.data_dir, self.id_list[idx] + '.dcm'))

        x = []
        for keyword, transform in self.features:
            x += [transform(x_dcm.get(keyword))]

        y_df = self.data_df.loc[self.data_df['patientId'] == self.id_list[idx]]
        y = self.y_transforms(y_df)
        if torch.is_tensor(y):
            y = [y]
        if self.joint_transforms:
            x, y = self.joint_transforms([x, y])
        return x, y



def split_validation(data_df, proportion): # Simple function to split the Pandas Dataframe
    id_list = data_df['patientId'].unique()

    mask = np.random.rand(len(id_list)) < proportion
    train_ids = id_list[mask]
    validation_ids = id_list[~mask]

    train_df = data_df.loc[data_df['patientId'].isin(train_ids)]
    validation_df = data_df.loc[data_df['patientId'].isin(validation_ids)]

    return train_df, validation_df



class GetBbox(object):
    """
    Simple class to convert Pandas Dataframe to a NumPy array of coordinate pairs for the bounding boxes
    """
    def __call__(self, df):
        bbox_df = df[['x', 'y', 'width', 'height']]
        if bbox_df.isnull().values.any():
            return torch.ones(1) - 2 # To signal that there is no bounding box

        bbox = bbox_df.values
        bbox[:, 2:] += bbox[:, :2] # We use 2 coordinates instead of 1 coordinate and width/height
        return torch.from_numpy(bbox)
