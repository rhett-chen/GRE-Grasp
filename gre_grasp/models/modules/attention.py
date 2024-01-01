import torch
import torch.nn as nn
import math


class MultiHeadAttn(nn.Module):
    def __init__(self, dim, nhead, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.nhead = nhead
        self.head_dim = int(dim // nhead)
        assert self.nhead * self.head_dim == self.dim
        self.linears = nn.ModuleList([nn.Linear(dim, dim) for _ in range(4)])
        if dropout == 0:
            self.dropout = None
        else:
            self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(queries, keys, values, mask=None, dropout=None):
        """
            queries: B x H x S x headdim
            keys: B x H x L x headdim
            values: B x H x L x headdim
            mask: B x 1 x S x L
        """
        head_dim = queries.size(-1)
        scores = queries @ keys.transpose(-1, -2) / math.sqrt(head_dim)  # B x H x S x L
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = torch.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        return scores @ values  # B x H x S x head_dim

    def forward(self, query, key, value, mask=None):
        """  (bs, max_len, word_feat_dim)
            query: B x S x D
            key: B x L x D
            value: B x L x D
            mask: B x S x L
        """
        batch_size = query.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)  # B x 1 x S x L, 1 for heads
        queries, keys, values = [
            layer(x).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
            for layer, x in zip(self.linears[:3], (query, key, value))
        ]  # (bs, nhead, max_len, head_dim) for word feat
        result = self.attention(queries, keys, values, mask, self.dropout)  # (bs, nhead, max_len, head_dim)
        result = result.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)  # (bs, max_len, dim)

        return self.linears[-1](result)


class AttentionModule(nn.Module):
    def __init__(self, dim, n_head, msa_dropout):
        super().__init__()
        self.dim = dim
        self.msa = MultiHeadAttn(dim, n_head, dropout=msa_dropout)
        self.norm1 = nn.LayerNorm(dim)

    def forward(self, q, k, v, mask):
        msa = self.msa(q, k, v, mask)
        x = self.norm1(v + msa)

        return x
