import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from ..utils.torch_utils import meshgrid

class YOLOClassifier(nn.Module):
    """
    Simple class for final convolutional classifier layers for YOLO

    Outputs a list of [(samples, num_classes) + output_gird

    Paramters
    ---------
    input_channels: Integer
        The input channels for the layer
    anchor_priors: PyTorch tensor
        PyTorch tensor (n_bbox, 2) of width and height priors for the bounding box predictions
    num_classes: Integer
        Number of classes
        Default is 20 from the YOLOv2 paper
    kernel_dim: Integer
        Dimension of the convolutional kernel
        Default is 3 from the YOLOv2 paper
    """
    def __init__(self, input_channels, anchor_priors, num_classes = 20, kernel_dim = 3):
        super(YOLOClassifier, self).__init__()

        self.input_channels = input_channels
        self.anchor_priors = anchor_priors
        self.num_classes = num_classes

        self.out_xy = nn.Conv2d(input_channels, len(anchor_priors) * 2, kernel_dim, padding = kernel_dim // 2)
        self.out_wh = nn.Conv2d(input_channels, len(anchor_priors) * 2, kernel_dim, padding = kernel_dim // 2)
        self.out_confidence = nn.Conv2d(input_channels, len(anchor_priors), kernel_dim, padding = kernel_dim // 2)
        self.out_class = nn.Conv2d(input_channels, len(anchor_priors) * num_classes, kernel_dim, padding = kernel_dim // 2)


    def forward(self, x):
        # Don't need gradients for the meshgrid
        with torch.no_grad()
            num_batches, _, height, width = x.shape
            grid = meshgrid(0, (x.shape[3] - 1) / x.shape[3], x.shape[3], 0, (x.shape[2] - 1) / x.shape[2], x.shape[2])
            grid = grid.repeat(len(self.anchor_priors), dim = 0)
            step = grid[:, 1, 1, None, None]

        out_xy = (F.sigmoid(self.out_xy(x)) + grid) / step
        out_wh = torch.exp(self.out_wh(x)) * self.anchor_priors.view(-1, 1, 1)
        out_confidence = F.sigmoid(self.out_confidence(x))
        # Note we use log_sigmoid instead of log_softmax incase our classes are not mutually exclusive
        out_class = F.log_sigmoid(self.out_class(x))

        # Do some reshaping for convenience later
        # Change to (num_batches, num_anchors, height, width, 2)
        out_xy = out_xy.view(num_batches, -1, 2, height, width).permute(0, 1, 3, 4, 2)
        out_wh = out_wh.view(num_batches, -1, 2, height, width).permute(0, 1, 3, 4, 2)

        # Change to (num_batches, num_anchors, height, width, num_classes)
        out_class = out_class.view(num_batches, -1, self.num_classes, height, width).permute(0, 1, 3, 4, 2)

        return [[out_xy, out_wh, out_confidence, out_class]] # Double list wrap so closure treats it as 1 output
