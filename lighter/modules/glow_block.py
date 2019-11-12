import math

import numpy as np
import scipy
import scipy.linalg

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.distributions import Normal


class ActNorm(nn.Module):
    """
    ActNorm layer from GLOW : Generative Flowwith Invertible 1×1 Convolutions paper
    ArXiv: 1807.03039

    Parameters
    ----------
    in_channels: Int
        Number of input channels
    """
    def __init__(self, in_channels):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channels, 1, 1))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))


    def initialize(self, x):
        with torch.no_grad():
            flatten = x.permute(1, 0, 2, 3).contiguous().view(x.shape[1], -1)
            mean = flatten.mean(1)[:, None, None]
            std = flatten.std(1)[:, None, None]

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-4))


    def forward(self, x):
        _, _, height, width = x.shape

        if self.initialized.item() == 0:
            self.initialize(x)
            self.initialized.fill_(1)

        log_det = height * width * torch.sum(torch.log(torch.abs(self.scale)))

        return self.scale * (x + self.loc), log_det


    def reverse(self, output):
        return output / self.scale - self.loc



class InvertibleConv2d(nn.Module):
    """
    Invertible 1x1 Convolutions layer from GLOW : Generative Flowwith Invertible 1×1 Convolutions paper
    ArXiv: 1807.03039

    Parameters
    ----------
    in_channels: Int
        Number of input channels
    """
    def __init__(self, in_channels):
        super().__init__()

        weight = torch.randn(in_channels, in_channels)
        q, _ = torch.qr(weight)
        weight = q[..., None, None]
        self.weight = nn.Parameter(weight)


    def forward(self, x):
        _, _, height, width = x.shape

        out = F.conv2d(x, self.weight)
        log_det = height * width * torch.slogdet(self.weight.squeeze().double())[1].float()

        return out, log_det


    def reverse(self, output):
        return F.conv2d(output, self.weight.squeeze().inverse()[..., None, None])



class InvertibleConv2dLU(nn.Module):
    """
    LU Decomposed Invertible 1x1 Convolution layer from GLOW : Generative Flowwith Invertible 1×1 Convolutions paper
    ArXiv: 1807.03039

    Parameters
    ----------
    in_channels: Int
        Number of input channels
    """
    def __init__(self, in_channels):
        super().__init__()

        # Original code uses normal, let's use kaiming_uniform_ instead
        #weight = torch.randn(in_channels, in_channels)

        weight = np.random.randn(in_channels, in_channels)

        q, _ = scipy.linalg.qr(weight)
        w_p, w_l, w_u = scipy.linalg.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        # Because these aren't learnable parameters, but we still want them in our state_dict so that they are on the same device
        # As everything else when we call to()
        # we use register_buffer to add them
        self.register_buffer('w_p', w_p)
        self.register_buffer('u_mask', torch.from_numpy(u_mask))
        self.register_buffer('l_mask', torch.from_numpy(l_mask))
        self.register_buffer('s_sign', torch.sign(w_s))
        self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))

        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(torch.log(torch.abs(w_s)))
        self.w_u = nn.Parameter(w_u)


    def calc_weight(self):
        weight = self.w_p @ (self.w_l * self.l_mask + self.l_eye) @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))

        return weight


    def forward(self, x):
        _, _, height, width = x.shape

        weight = self.calc_weight()

        out = F.conv2d(x, weight[..., None, None])
        log_det = height * width * torch.sum(self.w_s)

        return out, log_det


    def reverse(self, output):
        weight = self.calc_weight()

        return F.conv2d(output, weight.inverse()[..., None, None])



class CouplingBlock(nn.Module):
    """
    Coupling layer from GLOW : Generative Flowwith Invertible 1×1 Convolutions paper
    ArXiv: 1807.03039

    Parameters
    ----------
    in_channels: Int
        Number of input channels
    hidden_channels: Int
        Number of channels in the hidden layer
        Default is 512 in the paper but we'll use 128 for testing
    affine: Bool
        Whether to use affine or additive coupling
        Default in the paper is True
    """
    def __init__(self, in_channels, hidden_channels=128, affine=True):
        super().__init__()

        self.affine = affine

        self.net = nn.Sequential(
            nn.Conv2d(in_channels // 2, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels if self.affine else in_channels // 2, kernel_size=3, padding=1),
        )

        with torch.no_grad():
            self.net[0].weight.data.normal_(0, 0.05)
            self.net[0].bias.zero_()

            self.net[2].weight.data.normal_(0, 0.05)
            self.net[2].bias.zero_()

            self.net[4].weight.zero_()
            self.net[4].bias.zero_()


    def forward(self, x):
        x_a, x_b = x.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(x_b).chunk(2, 1)
            s = torch.exp(log_s) # Switch this around and stick with the method in the paper
            y_a = s * x_a + t

            #s = F.sigmoid(log_s + 2)
            #y_a = (x_a + t) * s

            log_det = torch.sum(torch.log(torch.abs(s)).view(x.shape[0], -1), 1)

        else:
            out = self.net(x_b)
            y_a = x_a + out
            log_det = 0

        return torch.cat([y_a, x_b], 1), log_det


    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            s = torch.exp(log_s)
            in_a = (out_a - t) / s
            #s = F.sigmoid(log_s + 2)
            #in_b = out_b / s - t

        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out

        return torch.cat([out_a, in_b], 1)


class Flow(nn.Module):
    """
    One single block of flow from GLOW : Generative Flowwith Invertible 1×1 Convolutions paper
    ArXiv: 1807.03039

    Parameters
    ----------
    in_channels: Int
        Number of input channels
    affine: Bool
        Whether to use affine or additive coupling
        Default in the paper is True
    conv_lu: Bool
        Whether to use LU decomposed invertible 1x1 convolutions
        Default in the paper is True
    """
    def __init__(self, in_channels, affine=True, conv_lu=True):
        super().__init__()

        self.actnorm = ActNorm(in_channels)

        if conv_lu:
            self.invconv = InvertibleConv2dLU(in_channels)
        else:
            self.invconv = InvertibleConv2d(in_channels)

        self.coupling = CouplingBlock(in_channels, affine=affine)


    def forward(self, x):
        out, log_det1 = self.actnorm(x)
        out, log_det2 = self.invconv(out)
        out, log_det3 = self.coupling(out)

        log_det = log_det1 + log_det2 + log_det3

        return out, log_det

    def reverse(self, output):
        x = self.coupling.reverse(output)
        x = self.invconv.reverse(x)
        x = self.actnorm.reverse(x)

        return x



class GlowBlock(nn.Module):
    """
    Glow Block (comprised of many flows) from GLOW : Generative Flowwith Invertible 1×1 Convolutions paper
    ArXiv: 1807.03039

    Parameters
    ----------
    in_channels: Int
        Number of input channels
    n_flows: Int
        Number of flows to use
    split: Bool
        Whether to apply a split operation to the channels at the end
        This is slightly different to just splittign the output along the channels dimension
    affine: Bool
        Whether to use affine or additive coupling
        Default in the paper is True
    conv_lu: Bool
        Whether to use LU decomposed invertible 1x1 convolutions
        Default in the paper is True
    """
    def __init__(self, in_channels, n_flows, split=True, affine=True, conv_lu=True):
        super().__init__()

        squeeze_dim = in_channels * 4

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(Flow(squeeze_dim, affine=affine, conv_lu=conv_lu))

        self.split = split

        if split:
            self.prior = nn.Conv2d(squeeze_dim // 2, squeeze_dim, 1)
        else:
            self.prior = nn.Conv2d(squeeze_dim, squeeze_dim * 2, 1)

        with torch.no_grad():
            self.prior.weight.zero_()
            self.prior.bias.zero_()


    def forward(self, x):
        batch_size, n_channel, height, width = x.shape
        squeezed = x.view(batch_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(batch_size, n_channel * 4, height // 2, width // 2)

        log_det = 0

        for i, flow in enumerate(self.flows):
            out, det = flow(out)
            log_det = log_det + det

        if self.split:
            z1, z2 = out.chunk(2, dim=1)
            mean, log_sd = self.prior(z1).chunk(2, dim=1)
            log_pz = Normal(mean, torch.exp(log_sd)).log_prob(z2)
            log_pz = log_pz.view(batch_size, -1).sum(dim=1)
            return z1, z2, log_det, log_pz

        else:
            zero = torch.zeros_like(out)
            mean, log_sd = self.prior(zero).chunk(2, dim=1)
            log_pz = Normal(mean, torch.exp(log_sd)).log_prob(out)
            log_pz = log_pz.view(batch_size, -1).sum(dim=1)
            return out, log_det, log_pz


    def reverse(self, z1, z2, reconstruct=False):
        if reconstruct:
            if self.split:
                z = torch.cat([z1, z2], 1)
            else:
                z = z2

        else:
            if self.split:
                mean, log_sd = self.prior(z1).chunk(2, 1)
                z2 = self.prior.mean + z2 * self.prior.loc
                z = torch.cat([z1, z2], dim=1)

            else:
                z = self.prior.loc + z2 * self.prior.scale

        for flow in self.flows[::-1]:
            z = flow.reverse(z)

        b_size, n_channel, height, width = z1.shape

        unsqueezed = z1.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(b_size, n_channel // 4, height * 2, width * 2)

        return unsqueezed
