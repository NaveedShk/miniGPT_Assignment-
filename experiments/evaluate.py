"""Evaluate a trained checkpoint.

* validation cross-entropy, perplexity and bits-per-character on the WHOLE val split
* five fixed prompts, sampled with a fixed seed (temperature 0.8)
* decoding comparison: greedy vs temperature sampling, with simple diversity /
  repetition metrics (distinct-n, repeated 4-grams, fraction of real words)

    python -m experiments.evaluate --checkpoint checkpoints/tiny_gpt.pt
"""
import argparse
import json
import math
import re

import torch

from src.dataset import build_data, load_text
from src.generate import generate_text
from src.utils import ROOT, count_parameters, load_checkpoint

FIXED_PROMPTS = [
    "ROMEO:",
    "JULIET:\nO Romeo,",
    "KING HENRY:\nWhat news",
    "First Citizen:\n",
    "To be, or not to be",
]


@torch.no_grad()
def full_val_loss(model, val_stream, batch_size=64):
    model.eval()
    total, n = 0.0, 0
    for x, y in val_stream.iter_eval_batches(batch_size, max_batches=0):
        _, loss = model(x, y)
        total += loss.item() * x.numel()
        n += x.numel()
    return total / n


def ngrams(s, n):
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def text_stats(text, vocab_words):
    words = re.findall(r"[A-Za-z']+", text)
    w_ngrams = [tuple(words[i : i + 2]) for i in range(len(words) - 1)]
    c4 = ngrams(text, 4)
    return {
        "distinct_word_1": round(len(set(words)) / max(1, len(words)), 3),
        "distinct_word_2": round(len(set(w_ngrams)) / max(1, len(w_ngrams)), 3),
        "repeated_char_4gram_frac": round(1 - len(set(c4)) / max(1, len(c4)), 3),
        "valid_word_frac": round(sum(w.lower() in vocab_words for w in words) / max(1, len(words)), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "tiny_gpt.pt"))
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model, tok, ckpt = load_checkpoint(args.checkpoint)
    _, _, val_stream = build_data(model.config.context_length)
    vocab_words = {w.lower() for w in re.findall(r"[A-Za-z']+", load_text())}

    loss = full_val_loss(model, val_stream)
    metrics = {
        "checkpoint": args.checkpoint,
        "parameters": count_parameters(model),
        "val_loss": round(loss, 4),
        "val_perplexity": round(math.exp(loss), 3),
        "val_bits_per_char": round(loss / math.log(2), 4),
        "uniform_baseline_loss": round(math.log(tok.vocab_size), 4),
    }
    print(json.dumps(metrics, indent=2))

    md = ["# Evaluation\n", "```json", json.dumps(metrics, indent=2), "```\n",
          f"## Fixed-prompt samples (temperature 0.8, seed {args.seed}, {args.tokens} new chars)\n"]
    samples = []
    for i, p in enumerate(FIXED_PROMPTS):
        text, _ = generate_text(model, tok, p, args.tokens, 0.8, seed=args.seed + i)
        samples.append({"prompt": p, "text": text})
        print(f"\n===== prompt {i + 1}: {p!r} =====\n{text}")
        md += [f"### Prompt {i + 1}: `{p!r}`\n", "```", text, "```\n"]

    md.append("## Decoding strategy comparison (prompt `ROMEO:`)\n")
    md.append("| strategy | distinct words | distinct bigrams | repeated char 4-grams | real-word fraction |")
    md.append("|---|---|---|---|---|")
    decoding = {}
    for name, kw in [("greedy (T=0)", dict(temperature=0.0)),
                     ("T=0.5", dict(temperature=0.5)),
                     ("T=0.8", dict(temperature=0.8)),
                     ("T=1.0", dict(temperature=1.0)),
                     ("T=1.5", dict(temperature=1.5)),
                     ("T=0.8, top-k=10", dict(temperature=0.8, top_k=10))]:
        text, _ = generate_text(model, tok, "ROMEO:", 500, seed=args.seed, **kw)
        st = text_stats(text[len("ROMEO:"):], vocab_words)
        decoding[name] = {"stats": st, "text": text}
        md.append(f"| {name} | {st['distinct_word_1']} | {st['distinct_word_2']} | "
                  f"{st['repeated_char_4gram_frac']} | {st['valid_word_frac']} |")
        print(f"\n[{name}] {st}\n{text[:200]}")
    md.append("")
    for name, d in decoding.items():
        md += [f"**{name}**\n", "```", d["text"][:400], "```\n"]

    out_dir = ROOT / "outputs" / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation.md").write_text("\n".join(md), encoding="utf-8")
    (out_dir / "evaluation.json").write_text(
        json.dumps({"metrics": metrics, "samples": samples, "decoding": decoding}, indent=2), encoding="utf-8")
    print(f"\nwrote {out_dir / 'evaluation.md'}")


if __name__ == "__main__":
    main()
