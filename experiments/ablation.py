"""Controlled ablations: change ONE factor, keep everything else fixed.

    python -m experiments.ablation                       # depth, heads, context
    python -m experiments.ablation --studies depth --steps 1500
    python -m experiments.ablation --studies position    # optional extra study

Every run uses the baseline config (configs/tiny_gpt.yaml) with the same seed,
batch size, LR schedule and number of steps; only the studied parameter changes.
Results: outputs/ablation/ablation_results.{csv,md,json} + plots + one sample per run.
"""
import argparse
import copy
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.train import train
from src.utils import ROOT, load_config

OUT = ROOT / "outputs" / "ablation"

STUDIES = {
    "depth": ("model.n_layers", [2, 4, 6]),
    "heads": ("model.n_heads", [2, 4, 8]),
    "context": ("model.context_length", [64, 128, 256]),
    "position": ("model.pos_encoding", ["sinusoidal", "learned", "none"]),
}


def set_key(cfg, dotted, value):
    sec, key = dotted.split(".")
    cfg[sec][key] = value


def run_study(name, base, steps):
    key, values = STUDIES[name]
    rows = []
    for v in values:
        cfg = copy.deepcopy(base)
        set_key(cfg, key, v)
        cfg["train"].update(
            max_steps=steps,
            eval_interval=max(steps // 10, 1),
            sample_interval=steps,
            warmup_steps=min(100, steps // 10),
            run_name=f"ablation_{name}_{v}",
            out_dir="checkpoints/ablation",
        )
        print(f"\n######## {name} = {v} ########")
        s = train(cfg, quiet=True)
        attn_mb = None
        if name == "context":
            # analytic size of the stored attention probabilities for one training step
            m = cfg["model"]
            attn_mb = cfg["train"]["batch_size"] * m["n_heads"] * v * v * 4 * m["n_layers"] / 2**20
        sample = (ROOT / "outputs" / "samples" / f"ablation_{name}_{v}.txt").read_text(encoding="utf-8")
        row = {
            "study": name,
            "value": v,
            "parameters": s["parameters"],
            "final_train_loss": s["final_train_loss"],
            "final_val_loss": s["final_val_loss"],
            "best_val_loss": s["best_val_loss"],
            "val_ppl": s["best_val_ppl"],
            "train_time_sec": s["train_time_sec"],
            "step_time_sec": s["step_time_sec"],
            "tokens_per_sec": s["tokens_per_sec"],
            "peak_gpu_memory_mb": s["peak_memory_mb"],
            "attention_matrix_mb": None if attn_mb is None else round(attn_mb, 1),
            "sample": sample.split("\n", 1)[1][:300],
        }
        print({k: row[k] for k in row if k != "sample"})
        rows.append(row)
    return rows


def plot_study(name, rows):
    key, _ = STUDIES[name]
    labels = [str(r["value"]) for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for r in rows:
        log = list(csv.DictReader(open(ROOT / "outputs" / "logs" / f"ablation_{name}_{r['value']}.csv")))
        axes[0].plot([int(x["step"]) for x in log], [float(x["val_loss"]) for x in log], "o-", ms=3, label=f"{key.split('.')[1]}={r['value']}")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("validation loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].bar(labels, [r["best_val_loss"] for r in rows], color="#4C78A8")
    lo = min(r["best_val_loss"] for r in rows)
    axes[1].set_ylim(lo * 0.95, max(r["best_val_loss"] for r in rows) * 1.02)
    axes[1].set_title("best validation loss")
    axes[2].bar(labels, [r["step_time_sec"] for r in rows], color="#F58518")
    axes[2].set_title("training-step time (s, excl. eval)")
    for ax in axes[1:]:
        ax.set_xlabel(key.split(".")[1])
    fig.suptitle(f"Ablation: {name}")
    fig.tight_layout()
    fig.savefig(OUT / f"ablation_{name}.png", dpi=150)
    plt.close(fig)


def write_tables(all_rows):
    cols = [c for c in all_rows[0] if c != "sample"]
    with open(OUT / "ablation_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    (OUT / "ablation_results.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in all_rows:
        lines.append("| " + " | ".join("" if r[c] is None else str(r[c]) for c in cols) + " |")
    lines.append("\n## Samples (prompt `ROMEO:`, temperature 0.8, seed 0)\n")
    for r in all_rows:
        lines.append(f"**{r['study']} = {r['value']}**\n\n```\n{r['sample']}\n```\n")
    (OUT / "ablation_results.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", nargs="+", default=["depth", "heads", "context"], choices=list(STUDIES))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--config", default=str(ROOT / "configs" / "tiny_gpt.yaml"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_config(args.config)

    all_rows = []
    prev = OUT / "ablation_results.json"
    if prev.exists():  # keep results of studies not re-run this time
        all_rows = [r for r in json.loads(prev.read_text()) if r["study"] not in args.studies]
    for name in args.studies:
        rows = run_study(name, base, args.steps)
        plot_study(name, rows)
        all_rows.extend(rows)
    order = list(STUDIES)
    all_rows.sort(key=lambda r: order.index(r["study"]))
    write_tables(all_rows)
    print(f"\nwrote {OUT}/ablation_results.md")


if __name__ == "__main__":
    main()
