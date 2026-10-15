"""Lesson 01: sequential depth of an RNN vs. self-attention, measured.

An RNN layer must compute h_t = f(h_{t-1}, x_t) for t = 1..T one step after
another: T sequential operations (sequential depth O(T)), O(T * d^2) work and
O(T * d) memory. A self-attention layer computes all T outputs with a few
batched matrix multiplies: O(1) sequential steps, O(T^2 * d) work and O(T^2)
memory for the score matrix. On parallel hardware the attention layer is
faster for the sequence lengths we use, at the price of quadratic memory.

This script times one forward pass of each layer type for growing T and
records the size of the attention score matrix.

    python -m experiments.why_transformers
"""
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.attention import MultiHeadAttention
from src.utils import ROOT, get_device


class LoopRNN(nn.Module):
    """Elman RNN written as an explicit Python loop to expose the T sequential steps."""

    def __init__(self, d):
        super().__init__()
        self.cell = nn.RNNCell(d, d)

    def forward(self, x):  # (B, T, d)
        h = torch.zeros(x.size(0), x.size(2), device=x.device)
        outs = []
        for t in range(x.size(1)):  # cannot start step t before step t-1 has finished
            h = self.cell(x[:, t], h)
            outs.append(h)
        return torch.stack(outs, 1)


def bench(fn, x, reps):
    with torch.no_grad():
        fn(x)  # warm-up
        if x.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn(x)
        if x.is_cuda:
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000


def main(B=16, d=128, heads=4):
    device = get_device()
    torch.manual_seed(0)
    rnn_loop = LoopRNN(d).to(device).eval()
    rnn_cudnn = nn.RNN(d, d, batch_first=True).to(device).eval()  # fused kernel, still sequential inside
    attn = MultiHeadAttention(d, heads, causal=True).to(device).eval()

    rows = []
    for T in [16, 32, 64, 128, 256, 512]:
        x = torch.randn(B, T, d, device=device)
        reps = 20
        row = {
            "T": T,
            "rnn_loop_ms": bench(rnn_loop, x, reps),
            "rnn_fused_ms": bench(lambda z: rnn_cudnn(z)[0], x, reps),
            "attention_ms": bench(attn, x, reps),
            "rnn_sequential_steps": T,
            "attention_sequential_steps": 1,
            "attention_scores_MB": B * heads * T * T * 4 / 2**20,
            "rnn_hidden_states_MB": B * T * d * 4 / 2**20,
        }
        rows.append(row)
        print(" | ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))

    out = ROOT / "outputs" / "logs" / "why_transformers.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": str(device), "B": B, "d": d, "rows": rows}, indent=2))

    Ts = [r["T"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(Ts, [r["rnn_loop_ms"] for r in rows], "o-", label="RNN (step loop)")
    a1.plot(Ts, [r["rnn_fused_ms"] for r in rows], "o-", label="RNN (fused nn.RNN)")
    a1.plot(Ts, [r["attention_ms"] for r in rows], "o-", label="multi-head self-attention")
    a1.set_xlabel("sequence length T")
    a1.set_ylabel("forward time (ms)")
    a1.set_title(f"Wall-clock time on {device.type} (B={B}, d={d})")
    a1.set_xscale("log", base=2)
    a1.set_yscale("log")
    a1.legend()
    a1.grid(alpha=0.3)
    a2.plot(Ts, [r["attention_scores_MB"] for r in rows], "o-", label="attention scores O(T^2)")
    a2.plot(Ts, [r["rnn_hidden_states_MB"] for r in rows], "o-", label="RNN hidden states O(T)")
    a2.set_xlabel("sequence length T")
    a2.set_ylabel("activation memory (MB, one layer)")
    a2.set_title("Memory trade-off")
    a2.set_xscale("log", base=2)
    a2.set_yscale("log")
    a2.legend()
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / "figures" / "why_transformers.png", dpi=150)
    print(f"saved {out} and outputs/figures/why_transformers.png")


if __name__ == "__main__":
    main()
