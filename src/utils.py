"""Shared helpers: config loading, seeding, device selection, checkpoints."""
from __future__ import annotations

import os
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_override(value: str):
    """Turn a CLI string like '4', '3e-4', 'true', 'learned' into a typed value."""
    return yaml.safe_load(value)


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply `section.key=value` overrides, e.g. `model.n_layers=6`."""
    for item in overrides or []:
        key, _, value = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = parse_override(value)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_params(n: int) -> str:
    return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.1f}K"


def save_checkpoint(path: str | Path, model, tokenizer, train_cfg: dict, metrics: dict) -> None:
    """Everything needed to rebuild the model in a fresh process lives in one file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(model.config),
            "tokenizer": tokenizer.to_dict(),
            "train_config": train_cfg,
            "metrics": metrics,
        },
        path,
    )


def download_file(url: str, dest: str | Path) -> Path:
    """Download `url` to `dest` (used for the dataset and for remote checkpoints).

    Google Drive share links of the form .../file/d/<id>/view are rewritten to a
    direct-download URL.
    """
    import re
    import urllib.request

    m = re.search(r"drive\.google\.com/file/d/([^/]+)", url)
    if m:
        url = f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)
    return dest


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    """Rebuild (model, tokenizer, checkpoint_dict) from a checkpoint file.

    If the file does not exist and the env var CHECKPOINT_URL is set, the
    checkpoint is downloaded first (GitHub release asset or Google Drive link).
    """
    from .gpt import GPTConfig, MiniGPT
    from .tokenizer import CharTokenizer

    path = Path(path)
    if not path.exists():
        url = os.environ.get("CHECKPOINT_URL")
        if not url:
            raise FileNotFoundError(
                f"{path} not found. Train a model first (python -m src.train) "
                "or set CHECKPOINT_URL to download one."
            )
        print(f"Downloading checkpoint from {url} ...")
        download_file(url, path)

    # weights_only=True: refuse arbitrary pickled objects (safe for downloaded checkpoints)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = MiniGPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    tokenizer = CharTokenizer.from_dict(ckpt["tokenizer"])
    return model, tokenizer, ckpt
