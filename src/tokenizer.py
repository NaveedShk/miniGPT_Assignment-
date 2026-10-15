"""Deterministic character-level tokenizer.

Design decisions
----------------
* Vocabulary = the sorted set of unique characters in the training corpus.
  Sorting makes the char -> id mapping deterministic: the same corpus always
  yields the same vocabulary, independent of dict/set iteration order.
* Special-token policy: no BOS/EOS/PAD tokens are needed. We train on
  contiguous fixed-length windows of text (so no padding) and generate until a
  length limit (so no EOS). The model starts generation from any user prompt.
* Unknown characters: a prompt may contain characters that never appear in the
  corpus (e.g. digits or emoji are absent from Tiny Shakespeare). With
  ``errors="strict"`` (default) encoding raises; with ``errors="ignore"`` the
  unknown characters are dropped. The CLI/API use ``ignore`` and report which
  characters were dropped.
"""
from __future__ import annotations

import json
from pathlib import Path


class CharTokenizer:
    def __init__(self, chars: list[str]):
        if len(chars) != len(set(chars)):
            raise ValueError("vocabulary contains duplicate characters")
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def unknown_chars(self, text: str) -> list[str]:
        return sorted({ch for ch in text if ch not in self.stoi})

    def encode(self, text: str, errors: str = "strict") -> list[int]:
        if errors == "strict":
            unknown = self.unknown_chars(text)
            if unknown:
                raise ValueError(f"characters not in vocabulary: {unknown!r}")
            return [self.stoi[ch] for ch in text]
        if errors == "ignore":
            return [self.stoi[ch] for ch in text if ch in self.stoi]
        raise ValueError(f"unknown errors mode: {errors}")

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    # --- persistence (the vocab is stored inside every checkpoint too) ---
    def to_dict(self) -> dict:
        return {"type": "char", "chars": self.chars}

    @classmethod
    def from_dict(cls, d: dict) -> "CharTokenizer":
        return cls(d["chars"])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
