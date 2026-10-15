"""Tiny Shakespeare download, train/val split, and next-token batches.

Next-token dataset construction
-------------------------------
Given the encoded corpus ``data = [t0, t1, t2, ...]`` and context length T,
a training example starting at offset i is::

    x = data[i     : i + T]        # inputs   (positions 0..T-1)
    y = data[i + 1 : i + T + 1]    # targets  (the same window shifted by one)

so ``y[t]`` is the token that follows ``x[t]``. Combined with the causal mask,
position t predicts ``y[t]`` while only being allowed to see ``x[0..t]``,
which gives T next-token training signals per window.
"""
from __future__ import annotations

from pathlib import Path

import torch

from .tokenizer import CharTokenizer
from .utils import ROOT, download_file

# Tiny Shakespeare (Karpathy, char-rnn). This is the same 1,115,394-character text
# served by the Hugging Face `tiny_shakespeare` / `karpathy/tiny_shakespeare` dataset.
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DEFAULT_DATA_PATH = ROOT / "data" / "tinyshakespeare.txt"


def load_text(path: str | Path = DEFAULT_DATA_PATH, download: bool = True) -> str:
    path = Path(path)
    if not path.exists():
        if not download:
            raise FileNotFoundError(path)
        print(f"Downloading Tiny Shakespeare to {path} ...")
        download_file(DATA_URL, path)
    return path.read_text(encoding="utf-8")


def train_val_split(ids: torch.Tensor, val_fraction: float = 0.1):
    """Contiguous split: the last `val_fraction` of the corpus is validation.

    A contiguous (rather than random) split avoids leaking overlapping windows
    between train and val.
    """
    n = int(len(ids) * (1 - val_fraction))
    return ids[:n], ids[n:]


class TokenStream:
    """Holds one split of the encoded corpus and samples (x, y) batches from it."""

    def __init__(self, ids: torch.Tensor, context_length: int):
        if len(ids) <= context_length:
            raise ValueError("split is shorter than the context length")
        self.ids = ids
        self.T = context_length

    def __len__(self) -> int:
        return len(self.ids) - self.T  # number of distinct windows

    def get_window(self, i: int):
        x = self.ids[i : i + self.T]
        y = self.ids[i + 1 : i + self.T + 1]  # shifted by one: next-token targets
        return x, y

    def get_batch(self, batch_size: int, generator: torch.Generator | None = None, device="cpu"):
        """Random contiguous windows -> x, y of shape (B, T)."""
        starts = torch.randint(0, len(self), (batch_size,), generator=generator)
        x = torch.stack([self.ids[i : i + self.T] for i in starts.tolist()])
        y = torch.stack([self.ids[i + 1 : i + self.T + 1] for i in starts.tolist()])
        return x.to(device), y.to(device)

    def iter_eval_batches(self, batch_size: int, max_batches: int, device="cpu"):
        """Deterministic, non-overlapping windows for a stable validation estimate."""
        n_windows = (len(self.ids) - 1) // self.T
        starts = [k * self.T for k in range(n_windows)]
        # spread the evaluated windows evenly over the split
        if max_batches and len(starts) > batch_size * max_batches:
            step = len(starts) / (batch_size * max_batches)
            starts = [starts[int(j * step)] for j in range(batch_size * max_batches)]
        for b in range(0, len(starts), batch_size):
            chunk = starts[b : b + batch_size]
            x = torch.stack([self.ids[i : i + self.T] for i in chunk])
            y = torch.stack([self.ids[i + 1 : i + self.T + 1] for i in chunk])
            yield x.to(device), y.to(device)


def build_data(context_length: int, val_fraction: float = 0.1, data_path=DEFAULT_DATA_PATH):
    text = load_text(data_path)
    tok = CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    train_ids, val_ids = train_val_split(ids, val_fraction)
    return tok, TokenStream(train_ids, context_length), TokenStream(val_ids, context_length)
