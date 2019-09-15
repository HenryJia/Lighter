import math

import numpy as np
import scipy
import scipy.linalg

import torch
from torch import Tensor
import torch.nn as nn
from pytorch.nn import init
import torch.nn.functional as F
from torch.distributions import Normal

from ..utils.torch_utils import gaussian_log_pz, gaussian_sample


class ActNorm(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        # We could use register_buffer but we don't really need it in our state_dict
        #self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
        self.initialized = False


    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = flatten.mean(1)[:, None, None]
            std = flatten.std(1)[:, None, None]

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))


    def forward(self, input):
        _, _, height, width = input.shape

        #if self.initialized.item() == 0:
        if not self.initialized:
            self.initialize(input)
            #self.initialized.fill_(1)
            self.initialized = True

        log_abs = logabs(self.scale)

        logdet = height * width * torch.sum(log_abs)

        return self.scale * (input + self.loc), logdet


    def reverse(self, output):
        return output / self.scale - self.loc


class InvertibleConv2d(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # Original code uses normal, let's use kaiming_uniform_ instead
        #weight = torch.randn(in_channels, in_channels)

        weight = torch.Tensor(in_channels, in_channels)
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        q, _ = torch.qr(weight)
        weight = q[..., None, None]
        self.weight = nn.Parameter(weight)


    def forward(self, input):
        _, _, height, width = input.shape

        out = F.conv2d(input, self.weight)
        logdet = height * width * torch.slogdet(self.weight.squeeze().double())[1]

        return out, logdet


    def reverse(self, output):
        return F.conv2d(output, self.weight.squeeze().inverse()[..., None, None])



class InvertibleConv2dLU(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # Original code uses normal, let's use kaiming_uniform_ instead
        #weight = torch.randn(in_channels, in_channels)

        weight = torch.Tensor(in_channels, in_channels)
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        weight = weight.numpy()

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

        # Because these aren't learnable parameters, but we still want them in our state_dict
        # we use register_buffer to add them
        self.register_buffer('w_p', w_p)

        # We don't really need these because they are constant
        #self.register_buffer('u_mask', torch.from_numpy(u_mask))
        #self.register_buffer('l_mask', torch.from_numpy(l_mask))
        #self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))

        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(w_s)
        self.w_u = nn.Parameter(w_u)


    def calc_weight(self):
        weight = self.w_p @ (self.w_l * self.l_mask + self.l_eye) @ ((self.w_u * self.u_mask) + torch.diag(self.w_s))

        return weight


    def forward(self, input):
        _, _, height, width = input.shape

        weight = self.calc_weight()

        out = F.conv2d(input, weight[..., None, None])
        logdet = height * width * sum(torch.logabs(self.w_s))

        return out, logdet


    def reverse(self, output):
        weight = self.calc_weight()

        return F.conv2d(output, weight.inverse()[..., None, None])



class CouplingBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels=512, affine=True):
        super().__init__()

        self.affine = affine

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.ReLU(inplace=True),
            Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1),
        )

        with torch.no_grad():
            self.net[0].bias.zero_()
            self.net[2].bias.zero_()
            self.net[4].weight.zero_()
            self.net[4].bias.zero_()

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(in_a).chunk(2, 1)
            s = torch.exp(log_s) # Switch this around and stick with the method in the paper
            out_a = s * in_a + t
            #s = F.sigmoid(log_s + 2)
            #out_b = (in_b + t) * s

            logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)

        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = 0

        return torch.cat([in_a, out_b], 1), logdet

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
    def __init__(self, in_channels, affine=True, conv_lu=True):
        super().__init__()

        self.actnorm = ActNorm(in_channels)

        if conv_lu:gaussian_log_pz
            self.invconv = InvertibleConv2dLU(in_channels)

        else:
            self.invconv = InvertibleConv2d(in_channels)

        self.coupling = CouplingBlock(in_channels, affine=affine)


    def forward(self, x):
        out, logdet1 = self.actnorm(x)
        out, logdet2 = self.invconv(out)
        out, logdet3 = self.coupling(out)

        logdet = logdet1 + logdet2 + logdet3

        return out, logdet

    def reverse(self, output):
        x = self.coupling.reverse(output)
        x = self.invconv.reverse(x)
        x = self.actnorm.reverse(x)

        return x



class GlowBlock(nn.Module):
    def __init__(self, in_channels, n_flow, split=True, affine=True, conv_lu=True):
        super().__init__()

        squeeze_dim = in_channels * 4

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, affine=affine, conv_lu=conv_lu))

        self.split = split

        if split:
            self.prior = nn.Conv2d(squeeze_dim // 2, squeeze_dim)
            with torch.no_grad():
                self.prior.weights.zero_()
                self.prior.bias.zero_()
        else:
            self.prior = Normal(nn.Paramter(torch.zeros(squeeze_dim, 1, 1)), torch.log(nn.Parameter(torch.zeros(squeeze_dim, 1, 1))))


    def forward(self, x):
        batch_size, n_channel, height, width = x.shape
        squeezed = x.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)

        logdet = 0

        for flow in self.flows:
            out, det = flow(out)
            logdet = logdet + det

        if self.split:
            z1, z2 = out.chunk(2, dim=1)
            mean, log_sd = self.prior(z1).chunk(2, dim=1)
            log_pz = Normal(mean, torch.exp(log_sd)).log_pzrob(z2)
            log_pz = log_pz.view(batch_size, -1).sum(dim=1)
            return z1, z2, logdet, log_pz

        else:
            log_pz = self.prior.log_pzrob(out) # This is identical to convolving with zeros
            log_pz = log_pz.view(batch_size, -1).sum(dim=1)
            return out, logdet, log_pz


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
