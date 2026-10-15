"""Figure 7: attention maps from the trained model.

    python -m experiments.attention_visualization --prompt "ROMEO:\nWhat light"

Saves one grid per layer (all heads) and prints, for every head, the average
distance it looks back and how much weight goes to the previous character -
a quick, quantitative description of what each head does.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.utils import ROOT, load_checkpoint

FIG_DIR = ROOT / "outputs" / "figures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "tiny_gpt.pt"))
    ap.add_argument("--prompt", default="ROMEO:\nBut soft, what light through yonder window breaks?")
    args = ap.parse_args()

    model, tok, _ = load_checkpoint(args.checkpoint)
    text = args.prompt.replace("\\n", "\n")  # allow a literal "\n" on the command line
    ids = tok.encode(text, errors="ignore")[: model.config.context_length]
    chars = [repr(tok.itos[i])[1:-1] for i in ids]
    model.set_store_attention(True)
    with torch.no_grad():
        model(torch.tensor([ids]))
    maps = model.attention_maps()  # list over layers of (1, H, T, T)

    T = len(ids)
    pos = torch.arange(T)
    dist = (pos[:, None] - pos[None, :]).clamp(min=0).float()  # how far back query i looks at key j
    print(f"prompt ({T} chars): {text!r}\n")
    print("layer head | mean look-back distance | weight on previous char | weight on self")
    for L, att in enumerate(maps):
        H = att.size(1)
        fig, axes = plt.subplots(1, H, figsize=(4.2 * H, 4.6))
        for h in range(H):
            w = att[0, h]
            ax = axes[h] if H > 1 else axes
            ax.imshow(w, cmap="viridis", vmin=0, vmax=w.max().item())
            ax.set_title(f"layer {L} head {h}")
            if T <= 64:
                ax.set_xticks(range(T), chars, fontsize=6, rotation=90)
                ax.set_yticks(range(T), chars, fontsize=6)
            ax.set_xlabel("key (attended to)")
            if h == 0:
                ax.set_ylabel("query")
            look_back = (w * dist).sum(-1).mean().item()
            prev = torch.diagonal(w, -1).mean().item()
            self_w = torch.diagonal(w).mean().item()
            print(f"  {L:3d}   {h:2d} | {look_back:23.2f} | {prev:23.3f} | {self_w:14.3f}")
        fig.suptitle(f"Attention weights, layer {L} (rows sum to 1; upper triangle is masked)")
        fig.tight_layout()
        path = FIG_DIR / f"fig7_attention_layer{L}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
    print(f"\nsaved {len(maps)} figures to {FIG_DIR}/fig7_attention_layer*.png")


if __name__ == "__main__":
    main()
