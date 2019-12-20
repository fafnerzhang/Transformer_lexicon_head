# attention.py

import torch
from torch import nn
import math
import torch.nn.functional as F
from train_utils import clones

"""def attention(query, key, value, mask=None, dropout=None):
    "Implementation of Scaled dot product attention"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Multi-head attention"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)"""


def attention(q, k, v, d_k,Lexicon_head, mask=None, dropout=None):
    #expand lexicon and original score

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    Lexicon_head = (Lexicon_head.unsqueeze(2).unsqueeze(3)).transpose(1, 2).transpose(2, 3).expand(q.size(0),1,q.size(2),q.size(2))


    #mask the padding position
    if mask is not None:
        mask = mask.unsqueeze(1)

        scores = scores.masked_fill(mask == 1, -1e9)
        Lexicon_head = Lexicon_head.masked_fill(mask==1,-1e9)

    #softmax score and lexicon
    scores = F.softmax(scores, dim=-1)
    Lexicon_head = F.softmax(Lexicon_head,dim=-1)

    #apply dropout
    if dropout is not None:
        scores = dropout(scores)
        Lexicon_head = dropout(Lexicon_head)


    Lexicon_head = Lexicon_head.repeat(1, q.size(1), 1, 1)
    #print("\nscores.shape", scores.shape, "\nscores", scores, "\nv.shape", v.shape, "v\n", v)
    #print("\nlexicon_head.shape\n", Lexicon_head.shape, "\nLexicon:\n", Lexicon_head)

    scores = (scores+Lexicon_head)/2

    #print("output shape",scores.shape,"output\n",scores)
    # using  multiply value
    output = torch.matmul(scores, v)




    #Lexicon_output = v.squeeze(1) * Lexicon_head.unsqueeze(2)


    return output,scores


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, d_model,sent_len, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        self.Lexicon_linear = nn.Linear(sent_len,sent_len)
        self.softmax = F.softmax
    def forward(self, q, k, v, mask=None,Lexicon=None):
        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sentlen * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        #linear and softmax to Lexicon_head
        Lexicon_head = self.Lexicon_linear(Lexicon/1000)

        # calculate attention using function we will define next
        output,scores = attention(q, k, v, self.d_k,Lexicon_head, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        #print("concat",concat)
        """output = concat * Lexicon_head.unsqueeze(2)"""
        #print("Lexicon_head*concat",output)
        output = self.out(concat)

        return output