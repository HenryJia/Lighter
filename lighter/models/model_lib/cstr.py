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

        return output



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

        output = self.attention(q, k, v, mask = mask)

        output = output.view(self.n_head, num_batches, len_q, self.d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(num_batches, len_q, -1) # b x lq x (n*dv)

        return output



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



class PositionEncoding(nn.Module):
    """
    Simple Positional Encoding layer from Attention Is All You Need

    Note dimensionality of input (last dimension) must be even
    """
    def __init__(self, max_len = 10000):
        super(PositionEncoding, self).__init__()
        self.max_len = max_len

    def forward(self, x, start = 0):
        length = x.shape[1]
        dim = x.shape[2]

        with torch.no_grad():
            encoding = torch.zeros((length, dim)).to(device = x.device, dtype = x.dtype)
            pos = torch.arange(start, start + length).view(-1, 1).float() / (self.max_len ** (2 * torch.arange(dim // 2).view(1, -1) / dim)).float()
            encoding[:, ::2] = torch.sin(pos)
            encoding[:, 1::2] = torch.cos(pos)

        return x + encoding


class WaveNetBlock(nn.Module):
    """
    Simple Causal Convolution block from WaveNet

    Parameters
    ----------
    in_channels: Integer
        Number of input channels
    out_channels: Integer
        Number of output channels
    skip_channels: Integer
        Number of channels for skip connection output
        If none, then no skip output will be used
    dilation: Integer
        Dilation of the dilated causal convolution
    """
    def __init__(self, in_channels, out_channels, skip_channels = None, dilation = 1):
        super(WaveNetBlock, self).__init__()

        self.filter_conv = nn.Conv1d(in_channels, in_channels, kernel_size = 2, stride = 1, dilation = dilation, padding = 0, bias = True)
        self.gate_conv = nn.Conv1d(in_channels, in_channels, kernel_size = 2, stride = 1, dilation = dilation, padding = 0, bias = True)
        self.res_conv = nn.Conv1d(in_channels, out_channels, kernel_size = 1)

        self.skip_channels = skip_channels
        if skip_channels:
            self.skip_channels = skip_channels
            self.skip_conv = nn.Conv1d(in_channels, skip_channels, kernel_size = 1)


    def forward(self, x, h = None): # x is our input, h is our extra conditional variable
        if h is not None:
            inp = x + h
        else:
            inp = x

        gated = torch.tanh(self.filter_conv(inp)) * torch.sigmoid(self.gate_conv(inp))

        output = self.res_conv(gated) + x[:, :, -gated.shape[2]:]

        if self.skip_channels:
            skip = self.skip_conv(gated)
            return output, skip
        else:
            return output



class CSTREncoderModel(nn.Module):
    """
    Encoder half of the CSTR model

    """
    def __init__(self, depth = 6):
        super(CSTREncoderModel, self).__init__()
        self.conv = nn.Sequential(nn.Embedding(num_embeddings = 256, embedding_dim = 64),
                                  PositionEncoding(max_len = int(1e7)),
                                  Permute((0, 2, 1)),
                                  nn.Conv1d(64, 128, kernel_size = 5, stride = 2, padding = 2),
                                  nn.LeakyReLU(),
                                  nn.Conv1d(128, 64, kernel_size = 5, stride = 2, padding = 2),
                                  nn.LeakyReLU(),
                                  nn.Conv1d(64, 64, kernel_size = 5, stride = 2, padding = 2),
                                  nn.LeakyReLU(),
                                  Permute(0, 2, 1),
                                  FeedForwardBlock(d_in = 64, d_hid = 128))

        self.rnn = nn.GRU(64, 32, 1, batch_first = True, bidirectional = True)

        #self.attn = nn.ModuleList([MultiHeadAttention(n_head = 4, d_model = 64, d_k = 16, d_v = 16),
                                   #MultiHeadAttention(n_head = 4, d_model = 64, d_k = 16, d_v = 16)])

        #self.ff = nn.ModuleList([FeedForwardBlock(d_in = 64, d_hid = 128),
                                 #FeedForwardBlock(d_in = 64, d_hid = 128)])


    def forward(self, x, h0 = None, return_state = False):
        # Note: We expect x in the format of (num_batches, time_steps)

        # Run the convolutional part of the encoder
        output = self.conv(x)
        if h0 is None:
            h0 = torch.zeros((2 * 1, output.shape[0], 32)).to(output.device)
        output, ht = self.rnn(output, h0)

        # Now do the Attention based part
        #for attn, ff in zip(self.attn, self.ff):
            ## Use the same thing for query, key & value
            ## Also use residual connections
            #residual = output
            #output = attn(output, output, output)
            #output = F.layer_norm(output + residual, output.shape[-1:])
            #output = F.layer_norm(ff(output) + output, output.shape[-1:])

        if return_state:
            return output, ht
        else:
            return output



class CSTRWaveNetDecoderModel(nn.Module):
    """
    Decoder half of the CSTR model

    """
    def __init__(self, depth = 4):
        super(CSTRWaveNetDecoderModel, self).__init__()

        self.depth = depth

        self.speaker_embed = nn.Embedding(num_embeddings = 108, embedding_dim = 64)
        self.embed = nn.Embedding(num_embeddings = 256, embedding_dim = 64)
        self.conv = nn.Sequential(Permute((0, 2, 1)), nn.Conv1d(64, 64, kernel_size = 1))
        self.rnn = nn.GRU(64, 64, 1, batch_first = True)

        self.wavenet_blocks = nn.ModuleList()
        self.attn = MultiHeadAttention(n_head = 4, d_model = 64, d_k = 16, d_v = 16)
        for i in range(depth):
            self.wavenet_blocks.append(WaveNetBlock(in_channels = 64, out_channels = 64, dilation = 2**i))

        self.out = nn.Sequential(nn.Conv1d(64, 256 + 1, kernel_size = 1), nn.LogSoftmax(dim = 1))


    def forward(self, x, speaker, y, h0 = None, return_state = False, return_seq = True):
        # Note: We expect x and y in the format of (num_batches, dims, time_steps)

        # Now prepare to apply decoder
        with torch.no_grad():
            pad = torch.zeros(x.shape[0], 2**self.depth).to(x.device).long()
            if y is not None:
                output = torch.cat([pad, y.long()], dim = 1)
            else:
                output = pad
            if not return_seq:
                output = output[:, -2 ** self.depth:]

        output = self.embed(output)
        output = self.conv(output)

        if h0 is None:
            h0 = torch.zeros((1, output.shape[0], 64)).to(output.device)

        q, ht = self.rnn(output.permute(0, 2, 1).contiguous(), h0)
        h = self.attn(q = q, k = x, v = x).permute(0, 2, 1).contiguous()
        h = h + self.speaker_embed(speaker)[..., None]
        output = self.wavenet_blocks[0](output, h)

        for wn in self.wavenet_blocks[1:]:
            output = wn(output)

        output = self.out(output)

        if return_state:
            return output, ht
        else:
            return output



class CSTRWaveNetModel(nn.Module):
    """
    WaveNet based model for CSTR Dataset

    TODO: Refactor this to use RNN attention

    """
    def __init__(self, depth = 6):
        super(CSTRWaveNetModel, self).__init__()

        self.encoder = CSTREncoderModel()
        self.decoder = CSTRWaveNetDecoderModel()


    def forward(self, x, speaker, y, h0 = None, return_state = False):
        output = self.encoder(x)
        output = self.decoder(output, speaker, y, h0, return_state)

        return output
