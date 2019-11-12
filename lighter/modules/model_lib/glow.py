import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ..glow_block import GlowBlock


class Glow(nn.Module):
    """
    Glow model from GLOW : Generative Flowwith Invertible 1×1 Convolutions paper
    ArXiv: 1807.03039

    Parameters
    ----------
    in_channels: Int
        Number of channels in the inpuy image
    n_flows: Int
        Number of flows per glow block
    n_blocks: Int
        Number of glow blocks in our model
    affine: Bool
        Whether to use affine coupling or additive couple
    conv_lu: Bool
        Whether to use LU decomposed 1x1 convolutions
    """
    def __init__(self, in_channels, n_flows, n_blocks, affine=True, conv_lu=True):
        super().__init__()

        self.blocks = nn.ModuleList()
        n_channels = in_channels
        for i in range(n_blocks - 1):
            self.blocks.append(GlowBlock(n_channels, n_flows, affine=affine, conv_lu=conv_lu))
            n_channels *= 2
        self.blocks.append(GlowBlock(n_channels, n_flows, split=False, affine=affine))


    def forward(self, x):
        log_pz_sum = 0
        log_det_sum = 0
        z1 = x
        z_out = []

        for i, block in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                z1, z2, log_det, log_pz = block(z1)
                z_out.append(z2)
            else:
                z1, log_det, log_pz = block(z1)
                z_out.append(z1)

            log_det_sum = log_det_sum + log_det
            log_pz_sum = log_pz_sum + log_pz

        return log_pz_sum, log_det_sum, z_out


    def reverse(self, z_list, reconstruct=False):
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                input = block.reverse(z_list[-1], z_list[-1], reconstruct=reconstruct)

            else:
                input = block.reverse(input, z_list[-(i + 1)], reconstruct=reconstruct)

        return input
