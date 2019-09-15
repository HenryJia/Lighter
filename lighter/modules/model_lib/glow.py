import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ..glow_block import GlowBlock


class Glow(nn.Module):
    def __init__(self, in_channel, n_flow, n_block, affine=True, conv_lu=True):
        super().__init__()

        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block - 1):
            self.blocks.append(GlowBlock(n_channel, n_flow, affine=affine, conv_lu=conv_lu))
            n_channel *= 2
        self.blocks.append(GlowBlock(n_channel, n_flow, split=False, affine=affine))

    def forward(self, x):
        log_pz_sum = 0
        logdet = 0
        out = x
        z_outs = []

        for block in self.blocks:
            out, det, log_pz, z_new = block(out)
            z_outs.append(z_new)
            logdet = logdet + det

            if log_pz is not None:
                log_pz_sum = log_pz + log_pz

        return log_pz_sum, logdet, z_outs

    def reverse(self, z_list, reconstruct=False):
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                input = block.reverse(z_list[-1], z_list[-1], reconstruct=reconstruct)

            else:
                input = block.reverse(input, z_list[-(i + 1)], reconstruct=reconstruct)

        return input
