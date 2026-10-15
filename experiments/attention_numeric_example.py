"""Milestone 1 evidence: the 3-token attention example computed step by step by the code.

Use this to check the hand calculation in the report (Figure 1) and to produce
the causal-mask figure (Figure 2).

    python -m experiments.attention_numeric_example
"""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.attention import causal_mask, scaled_dot_product_attention
from src.utils import ROOT

FIG_DIR = ROOT / "outputs" / "figures"

# Same matrices as tests/test_attention.py. Replace with the values from your
# own hand calculation if you chose different ones.
Q = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
K = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
V = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
TOKENS = ["tok1", "tok2", "tok3"]


def show(name, t):
    print(f"\n{name}:")
    for row in t.tolist():
        print("  [" + ", ".join(f"{v:8.4f}" if math.isfinite(v) else "    -inf" for v in row) + "]")


def main():
    torch.set_printoptions(precision=4)
    d_k = Q.size(-1)
    show("Q", Q)
    show("K", K)
    show("V", V)

    scores = Q @ K.T
    show("1) Q K^T", scores)
    scaled = scores / math.sqrt(d_k)
    show(f"2) Q K^T / sqrt(d_k)   (sqrt({d_k}) = {math.sqrt(d_k):.4f})", scaled)
    weights = torch.softmax(scaled, dim=-1)
    show("3) softmax (row-wise)  -> each row sums to 1", weights)
    out = weights @ V
    show("4) output = weights @ V", out)

    print("\n================ with causal mask ================")
    mask = causal_mask(3)
    show("causal mask (1 = allowed, 0 = future/blocked)", mask.float())
    masked = scaled.masked_fill(~mask, float("-inf"))
    show("5) masked scores", masked)
    w_c = torch.softmax(masked, dim=-1)
    show("6) masked softmax", w_c)
    out_c = w_c @ V
    show("7) masked output", out_c)

    # cross-check against the library implementation used by the model
    lib_out, lib_w = scaled_dot_product_attention(Q, K, V, mask)
    assert torch.allclose(lib_out, out_c) and torch.allclose(lib_w, w_c)
    print("\nsrc.attention.scaled_dot_product_attention matches the step-by-step result.")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    panels = [
        (mask.float(), "causal mask (1 = may attend)", "Greys"),
        (weights, "attention weights, no mask", "viridis"),
        (w_c, "attention weights, causal", "viridis"),
    ]
    for ax, (m, title, cmap) in zip(axes, panels):
        ax.imshow(m, cmap=cmap, vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{m[i, j]:.3f}", ha="center", va="center",
                        color="red" if cmap == "Greys" else "white", fontsize=11)
        ax.set_xticks(range(3), TOKENS)
        ax.set_yticks(range(3), TOKENS)
        ax.set_xlabel("key (attended to)")
        ax.set_ylabel("query (attending)")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_causal_mask_3tok.png", dpi=150)

    # larger mask heatmap
    T = 12
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(causal_mask(T).float(), cmap="Blues")
    ax.set_title(f"Causal mask, T={T}: row t may attend to columns <= t")
    ax.set_xlabel("key position")
    ax.set_ylabel("query position")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_causal_mask_12.png", dpi=150)
    print(f"saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
