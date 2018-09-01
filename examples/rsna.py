import os

import numpy as np # linear algebra 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Grayscale, ColorJitter
from torch.optim import Adam

import tqdm
from tqdm import tqdm
tqdm.monitor_interval = 0

from voc_annotations_parser import VocAnnotationsParser
from data_utils import VOCDataset
from train_utils import make_queue, Accuracy, RMSELoss
from models import TrackingNetwork


root_dir = '/home/data/PASCAL_VOC/VOCdevkit/VOC2012/'
img_dir = os.path.join(root_dir, 'JPEGImages')
ann_dir = os.path.join(root_dir, 'Annotations')
set_dir = os.path.join(root_dir, 'ImageSets', 'Main', 'trainval.txt')

parser = VocAnnotationsParser(img_dir, set_dir, ann_dir)
data_df = parser.get_annotation_dataframe()

# Split the dataframe into train and validation ourselves
mask = np.random.rand(len(data_df)) < 0.8
train_df = data_df[mask]
val_df = data_df[~mask]

# Set up the dataset
transforms = Compose([Grayscale(), ColorJitter(brightness = 0.1, contrast = 0.1, saturation = 0, hue = 0)])
train_set = VOCDataset(train_df, 0.5, (64, 64), transforms)
val_set = VOCDataset(val_df, 0.5, (64, 64), transforms)

train_loader = DataLoader(train_set, batch_size = 8, shuffle = True, pin_memory = True, num_workers = 6)
val_loader = DataLoader(val_set, batch_size = 8, shuffle = True, pin_memory = True, num_workers = 6)

# Set up the network for training
net = TrackingNetwork(input_shape = (1, 64, 64), num_classes = len(train_set.class_map)).cuda()
criterions = [nn.NLLLoss().cuda(), nn.MSELoss().cuda()]
metric_criterions = [Accuracy().cuda(), RMSELoss().cuda()]
optim = Adam(net.parameters(), lr = 3e-4)

loss_names = ['label_losses', 'position_losses']
metric_names = ['label_accuracy', 'position_rmse']

def run(queue, worker, length, train = False):
    pb = tqdm(total = length)

    loss_avgs = [[0, 0]] * len(criterions) # Keep an exponential running avg
    metric_avgs = [[0, 0]] * len(metric_criterions)
    total_loss_avg = 0
    while worker.is_alive() or not queue.empty():

        data, targets = queue.get()

        out = net(*data)

        if type(out) is torch.Tensor: # If we just have a single output
            out = [out]
        losses = [c(o, t) for c, o, t in zip(criterions, out, targets)]
        metrics = [m(o, t) for m, o, t in zip(metric_criterions, out, targets)]
        total_loss = sum(losses)

        if train:
            net.zero_grad()
            total_loss.backward()
            optim.step()

        total_loss_avg = 0.9 * total_loss_avg + 0.1 * total_loss.data.cpu().numpy()
        loss_avgs = [0.9 * la + 0.1 * l.data.cpu().numpy() for la, l in zip(loss_avgs, losses)]
        metric_avgs = [0.9 * ma + 0.1 * m.data.cpu().numpy() for ma, m in zip(metric_avgs, metrics)]

        losses_print = dict(zip(loss_names, loss_avgs))
        metrics_print = dict(zip(metric_names, metric_avgs))

        pb.update(data[0].size()[0])
        pb.set_postfix(loss = total_loss_avg, **losses_print, **metrics_print, queue_empty = queue.empty())

    queue.join()
    pb.close()

    result_str = '\nResults: loss = {:.3g} '.format(total_loss_avg)
    for name in losses_print.keys():
        result_str += name + ' = {:.3g} '.format(losses_print[name])
    for name in metrics_print.keys():
        result_str += name + ' = {:.3g} '.format(metrics_print[name])
    print(result_str + '\n')

print('Training')
for epoch in range(100):

    print('Epoch ', epoch + 1, ', Beginning Train')

    train_queue, train_worker = make_queue(train_loader)
    net.train()
    run(train_queue, train_worker, len(train_set), train = True)

    print('Epoch ', epoch + 1, ', Beginning Validation')
    val_queue, val_worker = make_queue(val_loader)
    net.eval()
    run(val_queue, val_worker, len(val_set), train = False)

    torch.save(net.cpu().state_dict(), 'network-epoch' + str(epoch) + '.nn')
    net.cuda()
