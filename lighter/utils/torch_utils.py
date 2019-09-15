import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo



'''
    Simple utility function to create a meshgrid for various vision functions

    Parameters:
    x1: Float
        Starting value for x grid (inclusive)
    x2: Float
        Ending value for x grid
    x_steps: Integer
        Number of steps for x
    y1: Float
        Starting value for y grid (inclusive)
    y2: Float
        Ending value for y grid
    y_steps: Integer
        Number of steps for y

    Outputs: (2, y_steps, x_steps) of [x_grid, y_grid]
'''
def meshgrid2d(x1, x2, x_steps, y1, y2, y_steps):
    grid_x = torch.linspace(x1, x2, x_steps).cuda().view(1, -1).expand(y_steps, x_steps)
    grid_y = torch.linspace(y1, y2, y_steps).cuda().view(-1, 1).expand(y_steps, x_steps)
    grid = torch.stack([grid_x, grid_y], dim = 0)
    return grid


'''
    Simple utility function for IOU for bounding boxes

    Note, (x1, y1), (x2, y2) are the centres and not the top left coordinates
'''
def bbox_iou(x1, y1, w1, h1, x2, y2, w2, h2):
    x_a = torch.max(x1 - w1 / 2.0, x2 - w2 / 2.0)
    y_a = torch.max(y1 - h1 / 2.0, y2 - h2 / 2.0)
    x_b = torch.min(x1 + h1 / 2.0, x2 + w2 / 2.0)
    y_b = torch.max(y1 + h1 / 2.0, y2 + h2 / 2.0)

    intersection = torch.clamp(x_b - x_a, min = 0) * torch.clamp(y_b - y_a, min = 0)
    union = w1 * h1 + w2 * h2 - intersection

    return intersection / (union + 1e-6)






