import math

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ..utils import Permute
from ..wavenet import WaveNetBlock
from ..attention import MultiHeadAttention



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



#class CSTRWaveNetModel(nn.Module):
    #"""
    #WaveNet based model for CSTR Dataset

    #TODO: Refactor this to use RNN attention

    #"""
    #def __init__(self, depth = 6):
        #super(CSTRWaveNetModel, self).__init__()

        #self.encoder = CSTREncoderModel()
        #self.decoder = CSTRWaveNetDecoderModel()


    #def forward(self, x, speaker, y, h0 = None, return_state = False):
        #output = self.encoder(x)
        #output = self.decoder(output, speaker, y, h0, return_state)

        #return output


class CSTRBridgeModel(nn.Module):
    """
    WaveNet model for generating audio from Mel Spectrograms

    """
    def __init__(self, depth = 7, width = 128):
        super(CSTRBridgeModel, self).__init__()

        self.depth = depth

        self.conv = nn.ModuleList([nn.BatchNorm1d(256),
                                   nn.ConvTranspose1d(256, width, stride = 2, kernel_size = 4, padding = 1),
                                   nn.LeakyReLU(),
                                   nn.Conv1d(in_channels = width, out_channels = width, kernel_size = 5, padding = 2),
                                   nn.LeakyReLU()])

        for i in range(self.depth - 1):
            self.conv.append(nn.ConvTranspose1d(width, width, stride = 2, kernel_size = 4, padding = 1))
            self.conv.append(nn.LeakyReLU())
            self.conv.append(nn.Conv1d(in_channels = width, out_channels = width, kernel_size = 5, padding = 2))
            self.conv.append(nn.LeakyReLU())



    def forward(self, x):
        output = x
        for c in self.conv:
            output = c(output)
        return output



class CSTRWaveNetModel(nn.Module):
    """
    WaveNet model for generating audio from Mel Spectrograms

    """
    def __init__(self, depth = 10, stacks = 2, width = 128):
        super(CSTRWaveNetModel, self).__init__()

        self.depth = depth
        self.stacks = stacks

        self.embed = nn.Embedding(num_embeddings = 256, embedding_dim = width)

        self.wavenet_blocks = nn.ModuleList()
        for j in range(stacks):
            for i in range(depth):
                self.wavenet_blocks.append(WaveNetBlock(in_channels = width, out_channels = width, skip_channels = width, dilation = 2**i))

        self.out = nn.Sequential(nn.Conv1d(width, 256, kernel_size = 1),
                                 nn.LeakyReLU(),
                                 nn.Conv1d(256, 256, kernel_size = 1),
                                 nn.LogSoftmax(dim = 1))


    def forward(self, h, y):
        # Note: We expect x and y in the format of (num_batches, dims, time_steps)
        output = self.embed(y).permute((0, 2, 1)).contiguous()

        skip = output * 0

        for wn in self.wavenet_blocks:
            output, s = wn(output, h)
            skip += s
        output += skip

        return self.out(output)


    def get_receptive_field(self):
        return self.stacks * 2 ** self.depth - (self.stacks - 1)



class CSTRMelModel(nn.Module):
    """
    Model for generating audio from Mel Spectrograms

    """
    def __init__(self, bridge_depth = 7, wavenet_depth = 10, stacks = 2, width = 128):
        super(CSTRMelModel, self).__init__()

        self.bridge = CSTRBridgeModel(depth = bridge_depth, width = width)
        self.wavenet = CSTRWaveNetModel(depth = wavenet_depth, stacks = stacks, width = width)


    def forward(self, x, y):
        output = self.bridge(x)
        if y.shape[1] < output.shape[2]:
            output = output[..., -y.shape[1]:]
        return self.wavenet(output, y)
