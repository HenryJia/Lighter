import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict


class DenseLayer(nn.Sequential):
    """
    Implementation of the DenseLayer from DenseNet

    Parameters
    ----------
    input_channels: Integer
        Number of input channels in to the DenseLayer
    growth_rate: Integer
        Growth rate of the DenseLayer.
        See the paper for more details
    activation: PyTorch activation
        PyTorch activation to use for convolutional layers
    bn_size: Integer
        Bottleneck size.
        The paper uses 4 for all layers so we are defaulting to this
    drop_rate: Float
        Dropout rate
        Default is 0
    """
    def __init__(self, input_channels, growth_rate, activation = nn.ReLU(inplace = True), bn_size = 4, drop_rate = 0):
        super(DenseLayer, self).__init__()
        self.add_module('conv1', nn.Conv2d(input_channels, bn_size * growth_rate, kernel_size = 1, bias = False))
        self.add_module('norm1', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu1', activation)
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size = 3, padding = 1, bias = False))
        self.add_module('norm2', nn.BatchNorm2d(growth_rate))
        self.add_module('relu2', activation)
        self.add_module('dropout', nn.Dropout2d(p = drop_rate))



class DenseBlock(nn.Module):
    """
    Simple implementation of a dense block for DenseNet.

    This is based on PyTorch code ut is more flexible and modular

    Paramters
    ---------
    layers: List of PyTorch Layers
        List of PyTorch layers to use as the layers in a dense block
        Each layer should take a single input and return a single output
    dim: Integer
        Dimension to concatenate features across
        Default is 1 for convolutional networks
    """
    def __init__(self, layers, dim = 1):
        super(DenseBlock, self).__init__()
        self.layers = layers
        self.dim = dim


    def forward(self, x):
        for l in self.layers:
            x = torch.cat([x, l(x)], dim = self.dim)
        return x


class TransitionLayer(nn.Sequential):
    """
    Implementation of the Transition Layer from DenseNet

    Parameters
    ----------
    input_channels: Integer
        Number of input channels in to the DenseLayer
    output_channels: Integer
        Growth rate of the DenseLayer.
        See the paper for more details
    activation: PyTorch activation
        PyTorch Activation to use for convolutional layers
    """
    def __init__(self, input_channels, output_channels, activation = nn.ReLU(inplace = True)):
        super(TransitionLayer, self).__init__()
        self.add_module('conv', nn.Conv2d(input_channels, output_channels, kernel_size = 1, bias = False))
        self.add_module('norm', nn.BatchNorm2d(output_channels))
        self.add_module('relu', activation)
        self.add_module('pool', nn.AvgPool2d(kernel_size = 2, stride = 2))


class DenseNet(nn.Sequential):
    """
    Densenet-BC model class, based on
    'Densely Connected Convolutional Networks' <https://arxiv.org/pdf/1608.06993.pdf>

    Note for the sake of flexibility of this framework, we only implement the feature extractor and not the final fully connected layers

    Parameters
    ----------
    growth_rate: Integer
        How many filters to add each layer ('k' in the paper)
        Default is 32 as that is what is used in the paper
    block_config: List of Integers
        How many layers in each DenseBlock
        Default here is the DenseNet121 model
    input_channels: Integer
        Number of channels in the input data
        Default is 3 for RGB data
    feature_channels: Integer
        The number of output channels in the first convolution layer
        Default is 64 as that is what is used in the paper
    bn_size: Integer
        Multiplicative factor for number of channels in bottle neck layers (i.e. bn_size * k channels in the bottleneck layer)
        Default is 4 as that is what is used in the paper
    compression: Float
        Factor to reduce channels to in transition layers ('theta' in the paper)
        Default is 0.5 as that is what is used in the paper
    activation: PyTorch activation function
        Dctivation function to use for the DenseNet
        Default is ReLU as that is what is used in the paper
    drop_rate: Float
        Dropout rate after each dense layer
        Default is 0
    """

    def __init__(self, growth_rate = 32, block_config = (6, 12, 24, 16), input_channels = 3, feature_channels = 64, bn_size = 4, compression = 0.5, activation = nn.ReLU(inplace = True), drop_rate = 0):
        super(DenseNet, self).__init__()

        # First convolution
        self.add_module('conv_0', nn.Conv2d(input_channels, feature_channels, kernel_size = 7, stride = 2, padding = 3, bias = False))
        self.add_module('norm_0', nn.BatchNorm2d(feature_channels))
        self.add_module('relu_0', activation)
        self.add_module('pool_0', nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1))

        # Each denseblock
        channels = feature_channels
        for i, num_layers in enumerate(block_config):
            layers = nn.ModuleList([DenseLayer(channels + i * growth_rate, growth_rate, activation, bn_size, drop_rate) for i in range(num_layers)])
            block = DenseBlock(layers)
            self.add_module('denseblock_%d' % (i + 1), block)

            channels = channels + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = TransitionLayer(input_channels = channels, output_channels = round(channels * compression))
                self.add_module('transition_%d' % (i + 1), trans)
                channels = round(channels * compression)

        self.output_channels = channels # This is useful when we're adding stuff on to the end

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight) # The default is kaiming_uniform_ with a = math.sqrt(5)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
