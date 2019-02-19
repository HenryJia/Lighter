import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ..utils.torch_utils import bbox_iou



class CombineLinear(nn.Module):
    """
    Simple class for combining multiple losses together into one linear

    Paramters
    ---------
    losses: list of loss functions
        List of PyTorch Loss functions to combine
    weightings: list of floats or 1D NumPy array of floats
        List of weightings to apply to each loss function
    """
    def __init__(self, losses, weights):
        super(CombineLinear, self).__init__()
        self.losses = losses
        self.weights = weights


    def forward(self, out, target):
        return sum([w * l(out, target) for w, l in zip(self.weights, self.losses)]) / sum(self.weights)



class RMSELoss(nn.MSELoss):
    """
    Simple root mean squared error loss function based on MSELoss
    """
    def __init__(self, **kwargs):
        super(RMSELoss, self).__init__(**kwargs)


    def forward(self, out, target):
        return torch.sqrt(super(RMSELoss, self).forward(out, target))



class F1Metric(nn.Module):
    """
    Differentiable F1 Metric (also known as dice coefficient)

    Note this is only for binary classification

    Parameters
    ----------
    smooth: float
        Parameter for smoothing the loss function and keeping it differentiable everywhere
    """
    def __init__(self, smooth = 1):
        super(F1Metric, self).__init__()
        self.smooth = smooth


    def forward(self, out, target):
        intersection = torch.sum(target * out)
        return (2. * intersection + self.smooth) / (torch.sum(target) + torch.sum(out) + self.smooth)



class F1Loss(F1Metric):
    """
    Differentiable F1 Loss

    This is exactly identical to F1Metric but we negate the result in order to optimise for the minimum

    To keep things positive we use 1 - x rather than - x
    """
    def __init__(self, **kwargs):
        super(F1Loss, self).__init__(**kwargs)


    def forward(self, out, target):
        return 1 - super(F1Loss, self).forward(out, target)



class IOUMetric(nn.Module):
    """
    Differentiable intersection over union metric

    Note this is only for binary classification

    Parameters
    ----------
    smooth: float
        Parameter for smoothing the loss function and keeping it differentiable everywhere
    """
    def __init__(self, smooth = 1):
        super(IOUMetric, self).__init__()
        self.smooth = smooth


    def forward(self, out, target):
        intersection = torch.sum(target * out)
        union = (torch.sum(target) + torch.sum(out) - intersection) # Inclusion exclusion formula
        return (intersection + self.smooth) / (union + self.smooth)



class IOULoss(IOUMetric):
    """
    Differentiable intersection over union loss

    This is exactly identical to IOUMetric but we negate the result in order to optimise for the minimum

    To keep things positive we use 1 - x rather than - x
    """
    def __init__(self, **kwargs):
        super(IOULoss, self).__init__(**kwargs)


    def forward(self, out, target):
        return 1 - super(IOULoss, self).forward(out, target)



class BinaryAccuracy(nn.Module):
    """
    Binary accuracy metric (not differentiable)
    """
    def __init__(self):
        super(BinaryAccuracy, self).__init__()

    def forward(self, out, target):
        return torch.mean((torch.round(out) == target).float())



class CategoricalAccuracy(nn.Module):
    """
    Categorical accuracy metric (not differentiable)

    Parameters
    ----------
    dim: integer
        The dimension to calculate the accuracy across
    one_hot: Bool
        Whether the data is in one hot format or not
    ignore_index: Integer
        Index value to ignore
    """
    def __init__(self, dim = 1, one_hot = False, ignore_index = -100):
        super(CategoricalAccuracy, self).__init__()
        self.dim = dim
        self.one_hot = one_hot
        self.ignore_index = ignore_index


    def forward(self, out, target):
        t = target if not self.one_hot else torch.argmax(target, dim = 1)
        mask =  1 - (t == self.ignore_index).float()
        return torch.sum((torch.argmax(out, dim = self.dim) == t).float() * mask) / torch.sum(mask)



class NLLLoss(nn.NLLLoss):
    """
    Identical to PyTorch's NLLLoss but can handle one hot format

    Parameters
    ----------
    one_hot: Bool
        Whether the data is in one hot format or not
    """
    def __init__(self, one_hot = False, **kwargs):
        super(NLLLoss, self).__init__(**kwargs)
        self.one_hot = one_hot

    def forward(self, out, target):
        t = target if not self.one_hot else torch.argmax(target, dim = 1)
        return super(NLLLoss, self).forward(out, t)



class YOLOLoss(nn.Module):
    """
    YOLO Loss function

    This Loss function expects the targets as a tensor of (samples, max_bbox, 5)
    where the last dimension is x, y, w, h, class

    The output to apply the loss on should be a list of PyTorch tensors of (out_xy, out_wh, out_confidence, out_class)

    Parameters
    ----------
    pos_weight: float
        Weight of the localisation loss
        Default is 5 from the YOLO papers
    noobj_weight: float
        Weight of the confidence loss where there is no object
        Default is 0.5 from the YOLO papers
    obj_weight: float
        Weight of the confidence loss where there is an object
        Default is 1 from the YOLO papers
    class_weight: float
        Weight of the classification loss
        Default is 1 from the YOLO papers
    """
    def __init__(self, pos_weight = 5.0, noobj_weight = 0.5, obj_weight = 1.0, class_weight = 1.0):
        super(YOLOLoss, self).__init__()
        self.pos_weight = pos_weight
        self.noobj_weight = noobj_weight
        self.obj_weight = obj_weight
        self.class_weight = class_weight

        self.bce = nn.BCELoss()


    def forward(self, out, target):
        # Note out is not a PyTorch tensor, it is a list of torch tensors
        # Note that out_xy and out_wh has dimensions (num_batches, num_anchors, height, width, 2)
        # Note that out_confidence has dimensions (num_batches, num_anchors, height, width)
        # Note that out_class has dimensions (num_batches, num_anchors, height, width, num_classes)
        out_xy, out_wh, out_confidence, out_class = out

        # We do not need gradients for generating the masks, so speed it up by disabling gradients
        with torch.no_grad():

            # Obtain various useful dimensions
            num_batches, num_anchors, height, width = out_xy.shape[:-1]

            target_xy, target_wh, target_class = target[:, :, :2], target[:, :, 2:4], target[:, :, 4:5]

            # Using multiply by 0 as a cheap way to do zeros_like without having to copy to device from CPU
            mask_xy = 0 * out_xy
            mask_wh = 0 * out_wh

            mask_confidence = 0 * out_confidence
            mask_class = 0 * out_class

            # Because PyTorch is procedural and loops don't slow it that much, to keep things simple we will simply use loops instead of vectorising
            for i in range(num_batches):
                o_xy = out_xy[i] # (num_anchors, height, width, 2)
                o_hw = out_hw[i] # (num_anchors, height, width, 2)

                t_xy = target_xy[i] # (num_bbox, 2)
                t_wh = target_wh[i] # (num_bbox, 2)
                t_class = target_class[i] # (num_bbox, 1)

                # First we compute obj_ij_mask, this is 1 if a object is in cell i, and has the lowest IOU with anchor j
                for j in range(num_bbox):
                    if t_xy[j, 0] < 0: # If there is no actual bounding box here, stop
                        break;

                    # Compute which cell is responsible for the bounding box
                    # We need to shift because we are rounding to the centre of bounding box (e.g. nearest to 1.5 or 2.5 instead of 1 or 2)
                    x = torch.round(t_xy[j, 0] * o_xy.shape[2] - 0.5)
                    y = torch.round(t_xy[j, 1] * o_xy.shape[1] - 0.5)

                    # Using multiply by 0 as a cheap way to do zeros like without having to copy to device from CPU
                    iou = bbox_iou(0 * t_xy[None, j], t_wh[None, j], 0 * o_xy[:, y, x], o_wh[:, x, y]) # (num_anchors, )
                    max_iou, anchor = torch.max(iou, dim = 0)

                    mask_xy[i, anchor, y, x] = t_xy[j]
                    mask_wh[i, anchor, y, x] = t_wh[j]

                    # Convert to 1 hot at the same time
                    mask_class[i, anchor, y, x, t_class[j]] = 1

                    mask_confidence[i, :, y, x] = iou

                mask_obj = (mask_wh > 0).float()
                mask_noobj = 1 - mask_obj


        # We could use PyTorch's MSELoss but those don't let us use a mask easily so we won't
        loss = 0
        loss += self.pos_weight * torch.mean(mask_obj[..., None] * (out_xy - mask_xy) ** 2)
        loss += self.pos_weight * torch.mean(mask_obj[..., None] * (torch.sqrt(out_wh) - torch.sqrt(mask_wh)) ** 2)

        loss += self.obj_weight * torch.mean(mask_obj * (out_confidence - mask_confidence) ** 2)
        loss += self.noobj_weight * torch.mean(mask_noobj * (out_confidence - mask_confidence) ** 2)

        # We will use PyTorch's BCELoss because we can multiply the mask with the outputs to make them the same
        loss += self.class_weight * self.bce(out_class * mask_obj[..., None], mask_class)

        return loss
