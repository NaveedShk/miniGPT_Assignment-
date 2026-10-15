"""Tiny overfit gate and checkpoint reload. Runs offline (no dataset download)."""
import math
import subprocess
import sys

import torch

from src.gpt import GPTConfig, MiniGPT
from src.tokenizer import CharTokenizer
from src.utils import ROOT, load_checkpoint, save_checkpoint

TEXT = "First Citizen:\nBefore we proceed any further, hear me speak.\n" * 4


def _setup():
    torch.manual_seed(0)
    tok = CharTokenizer.from_text(TEXT)
    ids = torch.tensor(tok.encode(TEXT))
    T = 32
    starts = [0, 7, 19, 40]
    x = torch.stack([ids[s : s + T] for s in starts])
    y = torch.stack([ids[s + 1 : s + T + 1] for s in starts])
    cfg = GPTConfig(vocab_size=tok.vocab_size, context_length=T, d_model=64, n_heads=4, n_layers=2, dropout=0.0)
    return tok, MiniGPT(cfg), x, y


def test_loss_starts_near_uniform():
    tok, model, x, y = _setup()
    _, loss = model(x, y)
    assert abs(loss.item() - math.log(tok.vocab_size)) < 0.5


def test_tiny_batch_overfits():
    tok, model, x, y = _setup()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for _ in range(200):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first or loss.item()
    assert loss.item() < 0.1 * first, (first, loss.item())
    assert loss.item() < 0.2


def test_checkpoint_reload_in_fresh_process(tmp_path):
    tok, model, x, y = _setup()
    path = tmp_path / "tiny.pt"
    save_checkpoint(path, model, tok, {"note": "test"}, {"step": 0, "val_loss": 0.0})

    model2, tok2, _ = load_checkpoint(path)
    model.eval()
    assert tok2.chars == tok.chars
    assert torch.allclose(model(x)[0], model2(x)[0])

    # a brand-new Python process can load the weights and generate
    cmd = [sys.executable, "-m", "src.generate", "--checkpoint", str(path),
           "--prompt", "First", "--max-new-tokens", "20", "--seed", "0", "--device", "cpu"]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    assert "Generated: First" in res.stdout
