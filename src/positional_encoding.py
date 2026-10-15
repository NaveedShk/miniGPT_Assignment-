"""Positional information.

Self-attention is permutation-equivariant: without position information,
shuffling the input tokens just shuffles the outputs, so "dog bites man" and
"man bites dog" would look the same. We add a position-dependent vector to every
token embedding.

Sinusoidal encoding (Vaswani et al., 2017):
    PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
    PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
Each pair of dimensions is a sinusoid with its own wavelength, from 2*pi up to
10000*2*pi, so low dimensions change quickly with position and high dimensions
slowly. PE[pos + k] is a fixed linear function (a rotation) of PE[pos], which
makes relative offsets easy to express. It has no parameters and is defined for
any position.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_table(max_len: int, d_model: int) -> torch.Tensor:
    """(max_len, d_model) table of sinusoidal position encodings."""
    if d_model % 2 != 0:
        raise ValueError("d_model must be even for sinusoidal encoding")
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)          # (L, 1)
    i2 = torch.arange(0, d_model, 2, dtype=torch.float32)                  # 2i = 0, 2, 4, ...
    inv_freq = torch.exp(-math.log(10000.0) * i2 / d_model)                # 1 / 10000^(2i/d)
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * inv_freq)
    pe[:, 1::2] = torch.cos(pos * inv_freq)
    return pe


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        # buffer: saved with the model and moved with .to(device), but not trained
        self.register_buffer("pe", sinusoidal_table(max_len, d_model), persistent=False)

    def forward(self, T: int) -> torch.Tensor:
        if T > self.pe.size(0):
            raise ValueError(f"sequence length {T} exceeds max_len {self.pe.size(0)}")
        return self.pe[:T]  # (T, C), broadcast over the batch


class LearnedPositionalEncoding(nn.Module):
    """Extra/ablation: one trainable vector per absolute position (GPT-2 style)."""

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, T: int) -> torch.Tensor:
        return self.emb(torch.arange(T, device=self.emb.weight.device))


def build_positional_encoding(kind: str, max_len: int, d_model: int) -> nn.Module | None:
    if kind == "sinusoidal":
        return SinusoidalPositionalEncoding(max_len, d_model)
    if kind == "learned":
        return LearnedPositionalEncoding(max_len, d_model)
    if kind == "none":
        return None
    raise ValueError(f"unknown positional encoding: {kind}")
