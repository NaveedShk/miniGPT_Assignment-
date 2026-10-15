"""Lesson 06 vs 07: bidirectional (BERT MLM) vs causal (GPT CLM) attention.

Part 1 - mask matrices (figure): GPT uses a lower-triangular mask; BERT lets
every token attend to every token and instead hides some *input* tokens with
[MASK].

Part 2 - short controlled experiment with the same small Transformer trained
three ways for the same number of steps:

  A) causal + next-token targets (GPT)       - the honest CLM setup
  B) bidirectional + next-token targets      - "BERT attention on a GPT objective"
     The target y[t] = x[t+1] is *in the input*, so the model can copy it from the
     future: training loss collapses, but only because it cheats. At the last
     position the next token is not in the input, so loss there is the honest
     number, and generation (where the future does not exist yet) fails.
  C) bidirectional + masked-LM targets (BERT) - 15% of input tokens are replaced
     by a [MASK] id and only those positions are scored. Uses both left and right
     context, which is great for understanding/filling in, but it is not a
     left-to-right generator.

    python -m experiments.mask_comparison --steps 600
"""
import argparse
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.attention import causal_mask
from src.dataset import build_data
from src.gpt import GPTConfig, MiniGPT
from src.utils import ROOT, get_device, set_seed

FIG_DIR = ROOT / "outputs" / "figures"


def plot_masks(T=10):
    gpt = causal_mask(T).float()
    bert = torch.ones(T, T)
    masked_positions = [2, 6]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(gpt, cmap="Blues", vmin=0, vmax=1)
    axes[0].set_title("GPT (causal LM): attend to j <= i")
    axes[1].imshow(bert, cmap="Blues", vmin=0, vmax=1)
    for p in masked_positions:
        axes[1].axvline(p, color="orange", lw=6, alpha=0.5)
    axes[1].set_title("BERT (masked LM): attend to all j\norange = [MASK]ed input tokens")
    for ax in axes:
        ax.set_xlabel("key position j")
        ax.set_ylabel("query position i")
        ax.set_xticks(range(T))
        ax.set_yticks(range(T))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_mlm_vs_clm_masks.png", dpi=150)


def mlm_batch(x, mask_id, p, g):
    """BERT-style corruption: choose 15% of positions; predict the original token there."""
    chosen = torch.rand(x.shape, generator=g) < p
    targets = torch.where(chosen, x, torch.full_like(x, -100))  # -100 = ignored by the loss
    inputs = x.clone()
    inputs[chosen] = mask_id  # (BERT also uses 80/10/10 replace/random/keep; we keep it simple)
    return inputs, targets


@torch.no_grad()
def eval_variant(model, stream, variant, mask_id, device, g):
    model.eval()
    full, last, n = 0.0, 0.0, 0
    for x, y in stream.iter_eval_batches(32, 10, "cpu"):
        if variant == "C":
            inp, tgt = mlm_batch(x, mask_id, 0.15, g)
            _, loss = model(inp.to(device), tgt.to(device))
            full += loss.item()
        else:
            logits, loss = model(x.to(device), y.to(device))
            full += loss.item()
            last += torch.nn.functional.cross_entropy(logits[:, -1].cpu(), y[:, -1]).item()
        n += 1
    model.train()
    return full / n, (last / n if variant != "C" else None)


def generate_sample(model, tok, device, n=150):
    idx = torch.tensor([tok.encode("ROMEO:")], device=device)
    g = torch.Generator(device=device).manual_seed(0)
    return tok.decode(model.generate(idx, n, temperature=0.8, generator=g)[0].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--context", type=int, default=64)
    args = ap.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_masks()

    device = get_device()
    tok, train_s, val_s = build_data(args.context)
    mask_id = tok.vocab_size  # one extra id reserved for [MASK]
    results, curves = {}, {}
    for variant, causal, desc in [
        ("A", True, "causal attention + next-token (GPT)"),
        ("B", False, "bidirectional attention + next-token (leaky)"),
        ("C", False, "bidirectional attention + masked-LM (BERT)"),
    ]:
        set_seed(0)
        cfg = GPTConfig(vocab_size=tok.vocab_size + (variant == "C"), context_length=args.context, d_model=128,
                        n_heads=4, n_layers=2, dropout=0.0, causal=causal)
        model = MiniGPT(cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
        g = torch.Generator().manual_seed(0)
        curve = []
        for step in range(args.steps):
            x, y = train_s.get_batch(32, generator=g)
            if variant == "C":
                x, y = mlm_batch(x, mask_id, 0.15, g)
            _, loss = model(x.to(device), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            curve.append(loss.item())
        val_full, val_last = eval_variant(model, val_s, variant, mask_id, device, torch.Generator().manual_seed(1))
        res = {
            "description": desc,
            "final_train_loss": round(sum(curve[-50:]) / len(curve[-50:]), 4),
            "val_loss_scored_positions": round(val_full, 4),
            "val_loss_last_position_only": None if val_last is None else round(val_last, 4),
        }
        if variant != "C":
            res["sample"] = generate_sample(model, tok, device)
        results[variant] = res
        curves[variant] = curve
        print(f"\n[{variant}] {desc}")
        print(json.dumps({k: v for k, v in res.items() if k != "sample"}, indent=2))
        if "sample" in res:
            print("sample:", res["sample"][:150].replace("\n", " / "))

    out = ROOT / "outputs" / "logs" / "mask_comparison.json"
    out.write_text(json.dumps(results, indent=2))

    fig, ax = plt.subplots(figsize=(8, 4))
    smooth = lambda c: [sum(c[max(0, i - 20): i + 1]) / len(c[max(0, i - 20): i + 1]) for i in range(len(c))]
    for v, c in curves.items():
        ax.plot(smooth(c), label=f"{v}: {results[v]['description']}")
    ax.axhline(math.log(tok.vocab_size), color="grey", ls="--", lw=1, label="uniform guess")
    ax.set_xlabel("step")
    ax.set_ylabel("train loss")
    ax.set_title("Same Transformer, different attention constraint / objective")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_mlm_vs_clm_losses.png", dpi=150)
    print(f"\nsaved {out}, fig_mlm_vs_clm_masks.png, fig_mlm_vs_clm_losses.png")


if __name__ == "__main__":
    main()
