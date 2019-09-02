import time, os, sys, argparse
import numpy as np
import pandas as pd
np.random.seed(94103)
np.seterr(all = 'raise')

import cv2
from skimage import measure

import torch
from torch import nn
from torchvision.transforms import Compose, Lambda

torch.backends.cudnn.deterministic = True
torch.manual_seed(94103)

from lighter.datasets.rsna import RSNADataset, split_validation, GetBbox
from lighter.datasets.transforms import Numpy2Tensor, Reshape, Resize, Bbox2Binary, Normalize

from lighter.modules.model_lib.rsna import UNet
from lighter.modules.densenet import DenseNet

from lighter.train import AsynchronousLoader

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type = str, help = 'Root directory containing the folder with the DICOM files and the csv files')
parser.add_argument('--test_dir', type = str, help = 'Folder containing the testing DICOM files')
parser.add_argument('--sample_csv', type = str, help = 'Filename of the csv sample submissions file')
parser.add_argument('--model', type = str, help = 'Directory of the trained model')
parser.add_argument('--out_csv', type = str, help = 'Directory of the output csv')

args = parser.parse_args()

root_dir = args.root_dir
dcm_dir = os.path.join(root_dir, args.test_dir)
sample_df_dir = os.path.join(root_dir, args.sample_csv)

sample_df = pd.read_csv(sample_df_dir)

data_df = pd.DataFrame(columns = ['patientId', 'x', 'y', 'width', 'height', 'Target'])
for idx, row in sample_df.iterrows():
    data_df.loc[idx] = [row['patientId']] + [np.nan] * 4 + [0]

print('Testing data dataframe lengths:\n', len(data_df))
print('Head of data dataframes:\n', data_df.head())

image_transforms = Compose([Resize((256, 256)), Numpy2Tensor(), Lambda(lambda x: x.float()), Reshape((1, 256, 256)), Normalize(0, 255)])
y_transforms = Compose([GetBbox(), Normalize(0, 1024), Bbox2Binary((256, 256)), Lambda(lambda x: x.float())])


data_set = RSNADataset(data_df, dcm_dir, [('pixel_array', image_transforms)], y_transforms)
data_loader = AsynchronousLoader(data_set, device = torch.device('cuda:0'), batch_size = 32, shuffle = False)

features = DenseNet(growth_rate = 8, block_config = (4, 8, 16, 32), activation = nn.LeakyReLU(inplace = True), input_channels = 1)
model = nn.Sequential(features,
                      nn.Conv2d(features.output_channels, 16, kernel_size = 1),
                      nn.LeakyReLU(inplace = True),
                      nn.Upsample(size = (256, 256), mode = 'nearest'),
                      nn.Conv2d(16, 1, kernel_size = 3, padding = 1),
                      nn.Sigmoid()).to(torch.device('cuda:0'))

model.load_state_dict(torch.load(args.model))
model.eval()

out_df = pd.DataFrame(columns = sample_df.columns)

with torch.no_grad():
    i = 0
    for data, targets in tqdm(data_loader):
        data = [data] if torch.is_tensor(data) else data

        out = model(*data).cpu().numpy()
        for j in range(out.shape[0]):
            out_mask = cv2.resize(out[j, 0], (1024, 1024), interpolation = cv2.INTER_LINEAR)

            labels = measure.label(np.round(out_mask))
            prediction_string = ''
            for region in measure.regionprops(labels):
                y1, x1, y2, x2 = region.bbox
                height = y2 - y1
                width = x2 - x1

                confidence = np.mean(out_mask[y1:y1 + height, x1:x1 + width])
                prediction_string += str(confidence) + ' ' + str(x1) + ' ' + str(y1) + ' ' + str(width) + ' ' + str(height) + ' '

            out_df.loc[i] = [data_set.id_list[i], prediction_string]
            i += 1

print('Head of output dataframes:\n', out_df.head())

out_df.to_csv(args.out_csv, index = False)
