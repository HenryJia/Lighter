import math

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ..layers import Permute



class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention from the Attention Is All You Need Paper

    Parameters
    ----------
    temperature: None or Float
        Temperature paramter for the softmax, if left as None, then it will be set to sqrt(dk) like in the paper
    """

    def __init__(self, temperature = None):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature


    # We take a query, key and value vector as input, with mask as optional input
    def forward(self, q, k, v, mask = None):
        attn = torch.bmm(q, k.transpose(1, 2))
        if self.temperature:
            attn = attn / self.temperature
        else:
            attn = attn / math.sqrt(q.shape[2])

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = F.softmax(attn, dim = 2)
        output = torch.bmm(attn, v)

        return output, attn



class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module from Attention Is All You Need

    Paramters
    ---------
    n_head: Integer
        Number of attention heads
    d_model:
        Dimensionality of input vectors
    d_k: Integer
        Dimensionality of key and query vectors
    d_v: Integer
        Dimensionality of value vectors
    """

    def __init__(self, n_head, d_model, d_k, d_v):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(d_model, n_head * d_k)
        self.w_k = nn.Linear(d_model, n_head * d_k)
        self.w_v = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention()



    def forward(self, q, k, v, mask = None):
        num_batches, len_q, _ = q.size()
        num_batches, len_k, _ = k.size()
        num_batches, len_v, _ = v.size()

        q = self.w_q(q).view(num_batches, len_q, self.n_head, self.d_k)
        k = self.w_k(k).view(num_batches, len_k, self.n_head, self.d_k)
        v = self.w_v(v).view(num_batches, len_v, self.n_head, self.d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, self.d_k) # (n_head * n_batches) x len_q x d_k
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, self.d_k) # (n_head * n_batches) x len_k x d_k
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, self.d_v) # (n_head * n_batches) x len_v x d_v

        if mask is not None:
            mask = mask.repeat(self.n_head, 1, 1) # (n*b) x .. x ..

        output, attn = self.attention(q, k, v, mask = mask)

        output = output.view(self.n_head, num_batches, len_q, self.d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(num_batches, len_q, -1) # b x lq x (n*dv)

        return output, attn



class FeedForwardBlock(nn.Module):
    """
    Simple wrapper for a 2 layer feed forward network

    Paramters
    ---------
    d_in: Integer
        Input dimensions, the output dimensions will be the same
    d_hid Integer
        Hidden dimensions
    """

    def __init__(self, d_in, d_hid):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

    def forward(self, x):
        return self.w_2(F.leaky_relu(self.w_1(x)))



class WaveNetBlock(nn.Module):
    """
    Simple Causa Convolution block from WaveNet

    Parameters
    ----------
    in_channels: Integer
        Number of input channels
    out_channels: Integer
        Number of output channels
    skip_channels: Integer
        Number of channels for skip connection output
    dilation: Integer
        Dilation of the dilated causal convolution
    """
    def __init__(self, in_channels, out_channels, skip_channels, dilation = 1):
        super(WaveNetBlock, self).__init__()

        self.filter_conv = nn.Conv1d(in_channels, in_channels, kernel_size = 2, stride = 1, dilation = dilation, padding = 0, bias = True)
        self.gate_conv = nn.Conv1d(in_channels, in_channels, kernel_size = 2, stride = 1, dilation = dilation, padding = 0, bias = True)
        self.res_conv = nn.Conv1d(in_channels, out_channels, kernel_size = 1)


    def forward(self, x, h = None): # x is our input, h is our extra conditional variable
        if h is not None:
            output = x + h
        else:
            output = x

        output = torch.tanh(self.filter_conv(output)) * torch.sigmoid(self.gate_conv(output))

        output = self.res_conv(output) + x[:, :, -output.shape[2]:]
        return output



class CSTRCharacterModel(nn.Module):
    """
    Character level model for CSTR Dataset

    """
    def __init__(self):
        super(CSTRCharacterModel, self).__init__()

        # Convolutional Encoder Section
        self.enc_conv = nn.Sequential(nn.Conv1d(256, 128, kernel_size = 5, stride = 2, padding = 2),
                                      nn.LeakyReLU(),
                                      nn.Conv1d(128, 64, kernel_size = 5, stride = 2, padding = 2),
                                      nn.LeakyReLU(),
                                      nn.Conv1d(64, 64, kernel_size = 5, stride = 2, padding = 2),
                                      nn.LeakyReLU(),
                                      Permute(0, 2, 1),
                                      FeedForwardBlock(d_in = 64, d_hid = 128))

        self.enc_attn = nn.ModuleList([MultiHeadAttention(n_head = 4, d_model = 64, d_k = 16, d_v = 16),
                                       MultiHeadAttention(n_head = 4, d_model = 64, d_k = 16, d_v = 16)])

        self.enc_ff = nn.ModuleList([FeedForwardBlock(d_in = 64, d_hid = 128),
                                     FeedForwardBlock(d_in = 64, d_hid = 128)])

        # For reducing initial dimensionality since we have 1 hot with 256 categories initially
        self.dec_conv = nn.Conv1d(256, 64, kernel_size = 1)

        self.wavenet_blocks = nn.ModuleList()
        self.dec_attn = MultiHeadAttention(n_head = 4, d_model = 64, d_k = 16, d_v = 16)
        for i in range(4):
            self.wavenet_blocks.append(WaveNetBlock(in_channels = 64, out_channels = 64, skip_channels = 64, dilation = 2**i))

        self.dec_out = nn.Sequential(nn.Conv1d(64, 256, kernel_size = 1),
                                     nn.LogSoftmax(dim = 1))


    def forward(self, x, y):
        # Note: We expect x and y in the format of (num_batches, dims, time_steps)

        # Run the convolutional part of the encoder
        enc_output = self.enc_conv(x)

        # Now do the Attention based part
        for attn, ff in zip(self.enc_attn, self.enc_ff):
            # Use the same thing for query, key & value
            # Also use residual connections
            residual = enc_output
            enc_output, _ = attn(enc_output, enc_output, enc_output)
            enc_output = F.layer_norm(enc_output + residual, enc_output.shape[-1:])
            enc_output = F.layer_norm(ff(enc_output) + enc_output, enc_output.shape[-1:])

        # Now prepare to apply decoder
        pad = torch.zeros(y.shape[0], y.shape[1], 2**4 - 1).to(y.device)
        output = torch.cat([pad, y.float()], dim = 2)
        output = self.dec_conv(output) # Reduce dimensionality

        # Doing this all in 1 go requires way too much memory, so we'll do it batchwise, which we can
        h = self.dec_attn(q = output.permute(0, 2, 1).contiguous(), k = enc_output, v = enc_output)[0].permute(0, 2, 1).contiguous()

        output = self.wavenet_blocks[0](output, h)
        for wn in self.wavenet_blocks[1:]:
            output = wn(output)

        output = self.dec_out(output)

        return output
