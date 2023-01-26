import math
from functools import partial

import torch
from torch import nn

from .. import utils
from . import Params


class MultiheadAttention(nn.Module):
    def __init__(self, in_dim: int, key_dim: int = None, n_heads: int = 8,
                 bias: bool = True):
        super().__init__()
        if key_dim is None:
            key_dim = in_dim
        if key_dim % n_heads:
            raise ValueError("key dimension must be divisible by n_heads")

        self.k = key_dim // n_heads
        self.h = n_heads
        self._inv_kdim = 1 / math.sqrt(self.k)

        self.linear = nn.ModuleList([
            nn.Linear(in_dim, key_dim, bias=False) for _ in range(3)])
        self.proj = nn.Linear(key_dim, in_dim, bias=bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        def split_heads(t: torch.Tensor):
            # Shape: (N, L, h * K) -> (N, L, h, K)
            t = t.view(n, l, self.h, self.k)
            # Shape: (N, L, h, K) -> (N * h, L, K)
            return t.transpose(1, 2).reshape(n * self.h, l, self.k)

        def merge_heads(t: torch.Tensor):
            # Shape: (N * h, L, K) -> (N, h, L, K)
            t = t.view(n, self.h, l, self.k)
            # Shape: (N, h, L, K) -> (N, L, h * K)
            return t.transpose(1, 2).reshape(n, l, self.h * self.k)

        # Make sure x is a 3D tensor
        n, l, _ = x.shape

        # Shape: (N, L, D) -> (N, L, h * K)
        qkv = (linear(x) for linear in self.linear)
        # Shape: (N, L, h * K) -> (N * h, L, K)
        q, k, v = (split_heads(t) for t in qkv)

        # Shape: (N * h, L, K) @ (N * h, K, L) -> (N * h, L, L)
        a = torch.bmm(q, k.transpose(1, 2)) * self._inv_kdim
        # Shape: (N * h, L, L) -> (N * h, L, L)
        w = utils.masked_softmax(a, mask, dim=-1)
        # Shape: (N * h, L, L) @ (N * h, L, K) -> (N * h, L, K)
        attention = torch.bmm(w, v)
        # Shape: (N * h, L, K) -> (N, L, h * K)
        attention = merge_heads(attention)

        # Shape: (N, L, K * h) -> (N, L, D)
        return self.proj(attention)


class TransformerBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, key_dim: int = None,
                 n_heads: int = 8, bias: bool = True, dropout: float = 0.1):
        super().__init__()

        if hidden_dim <= in_dim:
            raise ValueError("hidden dimension must be greater than input")

        self.attention = MultiheadAttention(in_dim, key_dim, n_heads, bias)
        self.a_dropout = nn.Dropout(dropout)
        self.a_norm = nn.LayerNorm(in_dim)

        self.linear = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
            nn.Dropout(dropout),
        )
        self.l_norm = nn.LayerNorm(in_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        h1 = self.attention(x, mask)
        h1 = self.a_dropout(h1)
        h1 = self.a_norm(x + h1)
        h2 = self.linear(h1)
        out = self.l_norm(h1 + h2)
        return out


class Transformer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_blocks: int,
                 key_dim: int = None, n_heads: int = 8,
                 bias: bool = True, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.blocks = nn.ModuleList([
            TransformerBlock(
                in_dim, hidden_dim, key_dim, n_heads, bias, dropout)
            for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        if mask is not None:
            mask = mask.repeat(self.n_heads, 1, 1)
        for layer in self.blocks:
            x = layer(x, mask)
        return x

    @staticmethod
    def positional_encoding(X: torch.Tensor, mask: torch.Tensor):
        _, l, d = X.shape

        # Shape (L, 1)
        pos = torch.arange(l, device=X.device).unsqueeze(1)

        # Shape (L, D // 2)
        sin = torch.sin(pos / 10000 ** (
            torch.arange(0, d, 2, device=X.device).unsqueeze(0) / d))
        cos = torch.cos(pos / 10000 ** (
            torch.arange(1, d, 2, device=X.device).unsqueeze(0) / d))

        pos_enc = torch.empty(l, d, device=X.device)
        pos_enc[:, 0::2] = sin
        pos_enc[:, 1::2] = cos

        return (X + pos_enc.unsqueeze(0)) * mask


class TransformerClassifier(nn.Module):
    class Params(Params):
        embedding_dim: int = 32
        key_dim: int = 32
        hidden_dim: int = 128
        n_heads: int = 4
        n_layers: int = 3
        dropout: float = 0.2
        readout: str = "max"

    def __init__(
            self, params: "TransformerClassifier.Params", num_tokens: int):
        super().__init__()

        self.params = params

        self.embedding = nn.Embedding(
            num_tokens, params.embedding_dim, padding_idx=0)
        self.transformer = Transformer(
            params.embedding_dim, params.hidden_dim, params.n_layers,
            key_dim=params.key_dim, n_heads=params.n_heads,
            dropout=params.dropout)

        if params.readout == "sum":
            readout = utils.masked_sum
        elif params.readout == "mean":
            readout = utils.masked_mean
        elif params.readout == "max":
            readout = utils.masked_max
        else:
            raise ValueError(f"Unknown readout: {params.readout}")

        self.readout = partial(readout, dim=1)
        self.proj = nn.Linear(params.embedding_dim, 2)

    def forward(self, X: torch.Tensor):
        # Shape: (N, L) -> (N, L, 1)
        mask = utils.get_mask(X).unsqueeze(2)

        # Shape: (N, L) -> (N, L, D)
        h = self.embedding(X)
        # Shape: (N, L, D) -> (N, L, D)
        h = self.transformer(h, mask)
        # Shape: (N, L, D) -> (N, D)
        h = self.readout(h, mask)
        # Shape: (N, D) -> (N, 2)
        return self.proj(h)
