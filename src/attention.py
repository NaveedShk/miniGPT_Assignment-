"""Scaled dot-product attention, causal masking, and multi-head attention.

Notation used in the shape comments
    B  batch size
    T  sequence length (number of tokens)
    C  model width (d_model)
    H  number of heads
    D  per-head width  (head_dim = C // H, i.e. d_k = d_v = D)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(T: int, device=None) -> torch.Tensor:
    """Boolean (T, T) mask, True where attention is ALLOWED.

    Row i = query position, column j = key position. Position i may attend to
    j <= i (lower triangle incl. diagonal). The strictly upper triangle (j > i,
    the future) is False.

        [[1, 0, 0],
         [1, 1, 0],
         [1, 1, 1]]
    """
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))


def scaled_dot_product_attention(q, k, v, mask=None, dropout_p: float = 0.0, training: bool = False):
    """softmax(Q K^T / sqrt(d_k)) V, written out explicitly.

    q: (..., T_q, d_k)   k: (..., T_k, d_k)   v: (..., T_k, d_v)
    mask: boolean, broadcastable to (..., T_q, T_k); True = may attend.
    Returns (output (..., T_q, d_v), weights (..., T_q, T_k)).
    """
    d_k = q.size(-1)
    # 1) similarity of every query with every key
    scores = q @ k.transpose(-2, -1)  # (..., T_q, T_k)
    # 2) scale so the dot products don't grow with d_k and saturate the softmax
    scores = scores / math.sqrt(d_k)
    # 3) forbidden positions get -inf so that exp(-inf) = 0 after the softmax
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    # 4) row-wise softmax: each query's weights over the keys sum to 1
    weights = F.softmax(scores, dim=-1)
    if dropout_p > 0.0 and training:
        weights = F.dropout(weights, p=dropout_p)
    # 5) weighted average of the value vectors
    out = weights @ v  # (..., T_q, d_v)
    return out, weights


class SelfAttention(nn.Module):
    """Single-head self-attention with learned Q/K/V projections (milestone 1)."""

    def __init__(self, d_model: int, d_k: int | None = None, causal: bool = True):
        super().__init__()
        d_k = d_k or d_model
        self.q_proj = nn.Linear(d_model, d_k, bias=False)
        self.k_proj = nn.Linear(d_model, d_k, bias=False)
        self.v_proj = nn.Linear(d_model, d_k, bias=False)
        self.causal = causal

    def forward(self, x):  # x: (B, T, C)
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)  # (B, T, d_k) each
        mask = causal_mask(x.size(1), x.device) if self.causal else None
        return scaled_dot_product_attention(q, k, v, mask)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with explicit split / merge of heads.

    Pipeline (shapes):
        x                       (B, T, C)
        Q, K, V = x W_q, W_k, W_v   (B, T, C) each
        split_heads             (B, T, C) -> (B, T, H, D) -> (B, H, T, D)
        attention per head      (B, H, T, D)   weights (B, H, T, T)
        merge_heads             (B, H, T, D) -> (B, T, H, D) -> (B, T, C)  [concat]
        output projection W_o   (B, T, C)
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, causal: bool = True, bias: bool = True):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)

        # Debug / evidence hooks (off by default).
        self.trace = None  # set to a list to record (step, shape) pairs
        self.store_attention = False
        self.last_attention = None  # (B, H, T, T) when store_attention is True

    def split_heads(self, x):
        """(B, T, C) -> (B, H, T, D): carve the C channels into H heads of width D."""
        B, T, C = x.shape
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def merge_heads(self, x):
        """(B, H, T, D) -> (B, T, C): concatenate the heads back along the channel axis."""
        B, H, T, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * D)

    def _log(self, name, t):
        if self.trace is not None:
            self.trace.append((name, tuple(t.shape)))

    def forward(self, x, mask=None):
        B, T, C = x.shape
        self._log("input x", x)

        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        self._log("Q = x W_q", q)

        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        self._log("Q split into heads", q)

        if mask is None and self.causal:
            mask = causal_mask(T, x.device)  # (T, T), broadcast over B and H

        # Every head attends independently: the leading (B, H) dims are batch dims.
        heads, weights = scaled_dot_product_attention(q, k, v, mask, self.dropout, self.training)
        self._log("attention weights per head", weights)
        self._log("per-head outputs", heads)
        if self.store_attention:
            self.last_attention = weights.detach()

        out = self.merge_heads(heads)
        self._log("heads concatenated", out)

        out = self.resid_dropout(self.out_proj(out))
        self._log("output projection W_o", out)
        return out
