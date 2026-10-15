"""Figure 4: sinusoidal positional encoding.

Panels
  (a) heatmap of PE[pos, dim]: low dims oscillate fast, high dims slowly
  (b) a few dimensions as curves over position
  (c) cosine similarity between positions: depends (mostly) on the distance |i - j|
  (d) effect on representations: the same token at different positions gets
      different input vectors

    python -m experiments.positional_encoding_viz
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.positional_encoding import sinusoidal_table
from src.utils import ROOT

FIG = ROOT / "outputs" / "figures" / "fig4_positional_encoding.png"


def main(max_len=128, d_model=128):
    pe = sinusoidal_table(max_len, d_model)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    im = ax.imshow(pe.T, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    ax.set_xlabel("position")
    ax.set_ylabel("embedding dimension")
    ax.set_title("(a) sinusoidal PE[pos, dim]")
    fig.colorbar(im, ax=ax)

    ax = axes[0, 1]
    for d in [0, 1, 8, 32, 64, 126]:
        ax.plot(pe[:, d], label=f"dim {d}")
    ax.set_xlabel("position")
    ax.set_title("(b) individual dimensions: wavelength grows with dim")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    unit = F.normalize(pe, dim=-1)
    im = ax.imshow(unit @ unit.T, cmap="viridis")
    ax.set_xlabel("position j")
    ax.set_ylabel("position i")
    ax.set_title("(c) cosine similarity PE[i]·PE[j]")
    fig.colorbar(im, ax=ax)

    ax = axes[1, 1]
    torch.manual_seed(0)
    token = torch.randn(d_model)  # one fixed token embedding (unit scale, like the model)
    reps = token + pe              # the same token placed at every position
    sim_to_pos0 = F.cosine_similarity(reps, reps[0:1], dim=-1)
    ax.plot(sim_to_pos0)
    ax.set_xlabel("position")
    ax.set_ylabel("cos sim to the same token at position 0")
    ax.set_title("(d) same token, different positions -> different inputs")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print(f"saved {FIG}")
    # numeric evidence: similarity decays with distance, regardless of absolute position
    sim = unit @ unit.T
    for k in [1, 4, 16, 64]:
        print(f"mean cos sim at distance {k:3d}: {torch.diagonal(sim, k).mean():.4f} "
              f"(std over positions {torch.diagonal(sim, k).std():.4f})")


if __name__ == "__main__":
    main()
