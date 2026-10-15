"""Decoder-style (GPT) Transformer block.

Pre-LayerNorm arrangement, as in GPT-2:

    x = x + Dropout(MHA(LN1(x)))        # tokens exchange information
    x = x + Dropout(FFN(LN2(x)))        # each token is processed independently

* Residual paths keep an identity route for the signal and the gradient, so
  deep stacks train stably; each sublayer only learns an update to x.
* LayerNorm normalises each token's C features to zero mean / unit variance
  (with learned scale and shift). Putting it *before* each sublayer (pre-LN)
  leaves the residual path a clean identity, which gives better-behaved
  gradients than the original post-LN design; GPT-2 and later models use it.
  The model applies one final LayerNorm after the last block.
* The FFN is two linear layers with a GELU in between, expanding to ffn_dim
  (usually 4*C) and projecting back to C.
"""
from __future__ import annotations

import torch.nn as nn

from .attention import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, ffn_dim)   # (B, T, C) -> (B, T, ffn_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(ffn_dim, d_model)   # (B, T, ffn_dim) -> (B, T, C)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.0, causal: bool = True):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout=dropout, causal=causal)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_dim, dropout)

    def forward(self, x, mask=None):  # x: (B, T, C) -> (B, T, C)
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.ffn(self.ln2(x))
        return x
