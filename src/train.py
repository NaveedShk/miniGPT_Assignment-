"""Train MiniGPT on Tiny Shakespeare.

    python -m src.train                                   # baseline config
    python -m src.train --set model.n_layers=2 --set train.run_name=depth2
    python -m src.train --overfit                         # tiny-batch debugging gate

Writes
    checkpoints/<run_name>.pt              best-validation checkpoint
    outputs/logs/<run_name>.csv            step, lr, train/val loss, tokens/s
    outputs/logs/<run_name>_summary.json   final metrics (params, time, losses, hardware)
    outputs/samples/<run_name>.txt         samples logged during training
    outputs/figures/<run_name>_loss.png    train vs val loss curve
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from pathlib import Path

import torch

from .dataset import build_data
from .gpt import GPTConfig, MiniGPT
from .utils import (
    ROOT,
    apply_overrides,
    count_parameters,
    format_params,
    get_device,
    load_config,
    save_checkpoint,
    set_seed,
)


def lr_at(step: int, cfg: dict) -> float:
    """Linear warm-up, then cosine decay from lr to min_lr."""
    lr, min_lr, warmup, total = cfg["lr"], cfg["min_lr"], cfg["warmup_steps"], cfg["max_steps"]
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (lr - min_lr)


def build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """AdamW; weight decay only on matrices (not biases, LayerNorm gains or PE)."""
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": cfg["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if cfg.get("optimizer", "adamw") != "adamw":
        raise ValueError("only adamw is implemented")
    return torch.optim.AdamW(groups, lr=cfg["lr"], betas=tuple(cfg["betas"]))


@torch.no_grad()
def evaluate(model, stream, batch_size, max_batches, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for x, y in stream.iter_eval_batches(batch_size, max_batches, device):
        _, loss = model(x, y)
        total += loss.item() * x.size(0)
        n += x.size(0)
    model.train()
    return total / n


def sample_text(model, tok, prompt, n_tokens, temperature, device, seed=0) -> str:
    g = torch.Generator(device=device).manual_seed(seed)
    idx = torch.tensor([tok.encode(prompt, errors="ignore")], dtype=torch.long, device=device)
    out = model.generate(idx, n_tokens, temperature=temperature, generator=g)
    return tok.decode(out[0].tolist())


def plot_losses(log_rows: list[dict], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r["step"] for r in log_rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, [r["train_loss"] for r in log_rows], label="train (smoothed)")
    ax.plot(steps, [r["val_loss"] for r in log_rows], label="validation")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy (nats / char)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train(cfg: dict, quiet: bool = False) -> dict:
    mcfg, dcfg, tcfg = cfg["model"], cfg["data"], cfg["train"]
    set_seed(tcfg["seed"])
    device = get_device(tcfg.get("device", "auto"))

    tok, train_stream, val_stream = build_data(
        mcfg["context_length"], dcfg["val_fraction"], ROOT / dcfg["path"]
    )
    gcfg = GPTConfig(vocab_size=tok.vocab_size, **mcfg)
    model = MiniGPT(gcfg).to(device)
    n_params = count_parameters(model)
    optimizer = build_optimizer(model, tcfg)

    run = tcfg["run_name"]
    ckpt_path = ROOT / tcfg["out_dir"] / f"{run}.pt"
    log_path = ROOT / "outputs" / "logs" / f"{run}.csv"
    sample_path = ROOT / "outputs" / "samples" / f"{run}.txt"
    for p in (log_path, sample_path):
        p.parent.mkdir(parents=True, exist_ok=True)

    hardware = torch.cuda.get_device_name(0) if device.type == "cuda" else f"{device.type} ({platform.processor() or platform.machine()})"
    if not quiet:
        print(f"device: {hardware}")
        print(f"vocab_size: {tok.vocab_size} | train tokens: {len(train_stream.ids):,} | val tokens: {len(val_stream.ids):,}")
        print(f"model parameters: {n_params:,} ({format_params(n_params)})")
        print(f"config: {gcfg}")

    g = torch.Generator().manual_seed(tcfg["seed"])
    log_rows, samples = [], []
    best_val, best_step = float("inf"), -1
    ema_loss = None
    tokens_per_step = tcfg["batch_size"] * mcfg["context_length"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    step_time = 0.0       # time spent in optimisation steps only (excludes eval/sampling)
    interval_step_time = 0.0

    model.train()
    for step in range(tcfg["max_steps"] + 1):
        lr = lr_at(step, tcfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        is_last = step == tcfg["max_steps"]
        if step % tcfg["eval_interval"] == 0 or is_last:
            val_loss = evaluate(model, val_stream, tcfg["batch_size"], tcfg["eval_batches"], device)
            now = time.time()
            steps_done = (step - log_rows[-1]["step"]) if log_rows else 0
            row = {
                "step": step,
                "lr": lr,
                "train_loss": ema_loss if ema_loss is not None else val_loss,
                "val_loss": val_loss,
                "val_ppl": math.exp(val_loss),
                "tokens_per_sec": tokens_per_step * steps_done / interval_step_time if steps_done else 0.0,
                "elapsed_sec": now - t0,
            }
            interval_step_time = 0.0
            log_rows.append(row)
            if not quiet:
                print(
                    f"step {step:5d} | lr {lr:.2e} | train {row['train_loss']:.4f} | "
                    f"val {val_loss:.4f} (ppl {row['val_ppl']:.2f}) | {row['tokens_per_sec']:,.0f} tok/s | {row['elapsed_sec']:.0f}s"
                )
            if val_loss < best_val:
                best_val, best_step = val_loss, step
                save_checkpoint(ckpt_path, model, tok, cfg, {"step": step, "val_loss": val_loss})
        if (step > 0 and step % tcfg["sample_interval"] == 0) or is_last:
            text = sample_text(model, tok, tcfg["sample_prompt"], tcfg["sample_tokens"], tcfg["temperature"], device)
            samples.append(f"===== step {step} =====\n{text}\n")
            if not quiet:
                print(f"--- sample @ step {step} ---\n{text}\n---")
        if is_last:
            break

        ts = time.perf_counter()
        x, y = train_stream.get_batch(tcfg["batch_size"], generator=g, device=device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if tcfg.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        optimizer.step()
        ema_loss = loss.item() if ema_loss is None else 0.9 * ema_loss + 0.1 * loss.item()  # .item() syncs the GPU
        dt = time.perf_counter() - ts
        step_time += dt
        interval_step_time += dt

    train_time = time.time() - t0
    # final train loss measured the same way as validation (no dropout, fixed windows)
    final_train = evaluate(model, train_stream, tcfg["batch_size"], tcfg["eval_batches"], device)

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    sample_path.write_text("\n".join(samples), encoding="utf-8")
    plot_losses(log_rows, ROOT / "outputs" / "figures" / f"{run}_loss.png", f"{run}: {format_params(n_params)} params")

    summary = {
        "run_name": run,
        "parameters": n_params,
        "model_config": gcfg.__dict__,
        "train_config": tcfg,
        "hardware": hardware,
        "train_time_sec": round(train_time, 1),           # wall clock incl. evaluation
        "step_time_sec": round(step_time, 1),             # optimisation steps only
        "tokens_per_sec": round(tokens_per_step * tcfg["max_steps"] / max(step_time, 1e-9), 1),
        "peak_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if device.type == "cuda" else None,
        "final_train_loss": round(final_train, 4),
        "final_val_loss": round(log_rows[-1]["val_loss"], 4),
        "best_val_loss": round(best_val, 4),
        "best_val_step": best_step,
        "best_val_ppl": round(math.exp(best_val), 3),
        "checkpoint": ckpt_path.relative_to(ROOT).as_posix(),
    }
    (ROOT / "outputs" / "logs" / f"{run}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not quiet:
        print(json.dumps(summary, indent=2))
    return summary


def overfit_check(steps: int = 300, quiet: bool = False) -> tuple[float, float]:
    """Debugging gate: a small model must memorise ONE repeated batch.

    If gradients flow and the targets are shifted correctly, the loss on a
    fixed batch drops towards zero. If it plateaus near log(vocab), something
    in the pipeline is broken.
    """
    set_seed(0)
    tok, train_stream, _ = build_data(context_length=32)
    model = MiniGPT(GPTConfig(vocab_size=tok.vocab_size, context_length=32, d_model=64, n_heads=4, n_layers=2, dropout=0.0))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x, y = train_stream.get_batch(4, generator=torch.Generator().manual_seed(0))
    first = None
    for step in range(steps):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
        if not quiet and step % 50 == 0:
            print(f"overfit step {step:4d} | loss {loss.item():.4f}")
    final = loss.item()
    if not quiet:
        print(f"initial loss {first:.4f} (log V = {math.log(tok.vocab_size):.4f}) -> final {final:.4f}")
    return first, final


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "tiny_gpt.yaml"))
    ap.add_argument("--set", action="append", metavar="KEY=VALUE", help="override, e.g. model.n_layers=6")
    ap.add_argument("--overfit", action="store_true", help="run the tiny-batch overfit check and exit")
    args = ap.parse_args()
    if args.overfit:
        overfit_check()
        return
    cfg = apply_overrides(load_config(args.config), args.set)
    train(cfg)


if __name__ == "__main__":
    main()
