"""Figures 3 and 5: multi-head tensor-shape trace and model summary / parameter count.

    python -m experiments.shape_trace
    python -m experiments.shape_trace --checkpoint checkpoints/tiny_gpt.pt
"""
import argparse

import torch

from src.attention import MultiHeadAttention
from src.gpt import GPTConfig, MiniGPT
from src.utils import ROOT, count_parameters, format_params, load_checkpoint, load_config


def trace_multihead(B=2, T=8, C=128, H=4):
    print(f"=== Multi-head attention shape trace  (B={B}, T={T}, d_model={C}, n_heads={H}, head_dim={C // H}) ===")
    mha = MultiHeadAttention(C, H)
    mha.trace = []
    mha(torch.randn(B, T, C))
    for name, shape in mha.trace:
        print(f"  {name:<30s} {str(shape):>20s}")


def trace_model(model: MiniGPT, B=2, T=8):
    cfg = model.config
    print(f"\n=== Full model forward trace (B={B}, T={T}) ===")
    x = torch.randint(0, cfg.vocab_size, (B, T))
    print(f"  {'token ids':<30s} {str(tuple(x.shape)):>20s}")
    h = model.token_emb(x) * cfg.d_model ** 0.5
    print(f"  {'token embedding':<30s} {str(tuple(h.shape)):>20s}")
    if model.pos_enc is not None:
        pe = model.pos_enc(T)
        print(f"  {'positional encoding':<30s} {str(tuple(pe.shape)):>20s}  (broadcast over batch)")
        h = h + pe
    for i, block in enumerate(model.blocks):
        h = block(h)
        print(f"  {f'after block {i}':<30s} {str(tuple(h.shape)):>20s}")
    h = model.ln_f(h)
    logits = model.lm_head(h)
    print(f"  {'final LayerNorm':<30s} {str(tuple(h.shape)):>20s}")
    print(f"  {'LM head logits':<30s} {str(tuple(logits.shape)):>20s}")


def summary(model: MiniGPT):
    print("\n=== Model summary ===")
    print(model)
    print("\nParameters by component:")
    groups = {}
    for name, p in model.named_parameters():
        key = name.split(".")[0] if not name.startswith("blocks") else "blocks." + name.split(".")[2]
        groups[key] = groups.get(key, 0) + p.numel()
    for k, v in groups.items():
        print(f"  {k:<20s} {v:>10,}")
    n = count_parameters(model)
    print(f"  {'TOTAL':<20s} {n:>10,}  ({format_params(n)})")
    print(f"\nconfig: {model.config}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()
    if args.checkpoint:
        model, _, _ = load_checkpoint(args.checkpoint)
    else:
        m = load_config(ROOT / "configs" / "tiny_gpt.yaml")["model"]
        model = MiniGPT(GPTConfig(vocab_size=65, **m))
    model.eval()
    trace_multihead(C=model.config.d_model, H=model.config.n_heads)
    with torch.no_grad():
        trace_model(model)
    summary(model)


if __name__ == "__main__":
    main()
