import math

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


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
