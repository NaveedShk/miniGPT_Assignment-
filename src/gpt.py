"""MiniGPT: a GPT-style decoder-only causal language model.

    token ids          (B, T)
    token embedding    (B, T, C)    * sqrt(C)
    + positions        (T, C)       sinusoidal (default) or learned
    dropout
    N x TransformerBlock (causal)   (B, T, C)
    final LayerNorm    (B, T, C)
    LM head            (B, T, V)    logits over the vocabulary
    loss = cross_entropy(logits[b, t], targets[b, t])  where targets = inputs shifted by one
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional_encoding import build_positional_encoding
from .transformer_block import TransformerBlock


@dataclass
class GPTConfig:
    vocab_size: int = 65
    context_length: int = 128
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    ffn_dim: int | None = None      # defaults to 4 * d_model
    dropout: float = 0.1
    pos_encoding: str = "sinusoidal"  # "sinusoidal" | "learned" | "none"
    tie_weights: bool = False
    causal: bool = True             # False only for the BERT-style mask experiment

    def __post_init__(self):
        if self.ffn_dim is None:
            self.ffn_dim = 4 * self.d_model


class MiniGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        C = config.d_model
        self.token_emb = nn.Embedding(config.vocab_size, C)
        self.pos_enc = build_positional_encoding(config.pos_encoding, config.context_length, C)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(C, config.n_heads, config.ffn_dim, config.dropout, config.causal)
            for _ in range(config.n_layers)
        )
        self.ln_f = nn.LayerNorm(C)
        self.lm_head = nn.Linear(C, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if config.tie_weights:
            # input embedding and output projection share one (V, C) matrix
            self.lm_head.weight = self.token_emb.weight
        # GPT-2: shrink the init of projections that write into the residual stream
        # so the variance of x does not grow with depth.
        for name, p in self.named_parameters():
            if name.endswith("attn.out_proj.weight") or name.endswith("ffn.fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding) and m is self.token_emb:
            # std = 1/sqrt(C) and the forward pass multiplies by sqrt(C), so the
            # embeddings enter with unit scale, comparable to the sinusoids.
            nn.init.normal_(m.weight, mean=0.0, std=self.config.d_model ** -0.5)

    def forward(self, idx, targets=None, mask=None):
        """idx: (B, T) token ids. targets: (B, T) next-token ids or None.

        Returns (logits (B, T, V), loss or None).
        """
        B, T = idx.shape
        if T > self.config.context_length:
            raise ValueError(f"sequence length {T} > context_length {self.config.context_length}")
        x = self.token_emb(idx) * math.sqrt(self.config.d_model)  # (B, T, C)
        if self.pos_enc is not None:
            x = x + self.pos_enc(T)                                # (T, C) broadcast over B
        x = self.drop(x)
        for block in self.blocks:
            x = block(x, mask=mask)                                # (B, T, C)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                   # (B, T, V)

        loss = None
        if targets is not None:
            # flatten so every (batch, position) pair is one classification example;
            # ignore_index lets the MLM experiment skip unmasked positions.
            loss = F.cross_entropy(
                logits.view(B * T, -1), targets.view(B * T), ignore_index=-100
            )
        return logits, loss

    def set_store_attention(self, flag: bool) -> None:
        for block in self.blocks:
            block.attn.store_attention = flag

    def attention_maps(self) -> list[torch.Tensor]:
        """Per-layer attention weights (B, H, T, T) from the last forward pass."""
        return [block.attn.last_attention for block in self.blocks]

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = None,
        top_p: float | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling. idx: (B, T0) prompt ids -> (B, T0 + max_new_tokens).

        temperature == 0 means greedy decoding (argmax).
        """
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            # the model only knows context_length positions: keep the most recent ones
            idx_cond = idx[:, -self.config.context_length :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # only the last position predicts the next token
            if temperature == 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                if top_p is not None and 0 < top_p < 1:
                    probs = _top_p_filter(probs, top_p)
                next_id = torch.multinomial(probs, 1, generator=generator)
            idx = torch.cat([idx, next_id], dim=1)
        self.train(was_training)
        return idx


def _top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus sampling: keep the smallest set of tokens whose probability mass >= top_p."""
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)
    remove = cumulative - sorted_probs > top_p  # tokens entirely beyond the nucleus
    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
    filtered = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
    return filtered / filtered.sum(dim=-1, keepdim=True)
