"""Command-line inference.

    python -m src.generate --prompt "ROMEO:" --max-new-tokens 300
    python -m src.generate --prompt "To be" --temperature 0      # greedy
    python -m src.generate --checkpoint checkpoints/tiny_gpt.pt --top-k 20 --seed 1
"""
from __future__ import annotations

import argparse

import torch

from .utils import ROOT, count_parameters, format_params, get_device, load_checkpoint


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 200, temperature: float = 0.8,
                  top_k: int | None = None, top_p: float | None = None, seed: int | None = None):
    """Returns (full_text, dropped_chars). Unknown prompt characters are dropped."""
    device = next(model.parameters()).device
    dropped = tokenizer.unknown_chars(prompt)
    ids = tokenizer.encode(prompt, errors="ignore")
    if not ids:  # empty prompt: start from a newline, a natural "start of text" in this corpus
        ids = tokenizer.encode("\n", errors="ignore") or [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p, generator=gen)
    return tokenizer.decode(out[0].tolist()), dropped


def main():
    ap = argparse.ArgumentParser(description="Generate text with a trained MiniGPT checkpoint")
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "tiny_gpt.pt"))
    ap.add_argument("--prompt", default="ROMEO:")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8, help="0 = greedy decoding")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = get_device(args.device)
    model, tok, ckpt = load_checkpoint(args.checkpoint, device)
    text, dropped = generate_text(model, tok, args.prompt, args.max_new_tokens, args.temperature,
                                  args.top_k, args.top_p, args.seed)
    cfg = model.config

    print(f"Prompt: {args.prompt}")
    if dropped:
        print(f"(dropped characters not in vocabulary: {dropped})")
    print(f"Generated: {text}")
    print()
    print(f"model parameters: {format_params(count_parameters(model))} ({count_parameters(model):,})")
    print(f"layers/heads/d_model: {cfg.n_layers}/{cfg.n_heads}/{cfg.d_model} | vocab: {cfg.vocab_size}")
    print(f"context length: {cfg.context_length}")
    print(f"temperature: {args.temperature} | top_k: {args.top_k} | top_p: {args.top_p} | seed: {args.seed}")
    print(f"checkpoint: {args.checkpoint} (step {ckpt['metrics'].get('step')}, val loss {ckpt['metrics'].get('val_loss'):.4f})")
    print(f"device: {device}")


if __name__ == "__main__":
    main()
