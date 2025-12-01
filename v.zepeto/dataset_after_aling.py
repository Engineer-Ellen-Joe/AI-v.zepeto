"""
Dataset loader for After_aling 01..07 with chunking.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import torch
from torch.utils.data import Dataset


def read_files(data_dir: Path, names: Sequence[str]) -> List[str]:
    texts = []
    for name in names:
        path = data_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        texts.append(path.read_text(encoding="utf-8"))
    return texts


class ChunkedLMDataset(Dataset):
    """Token stream chunked into contiguous blocks."""

    def __init__(self, tokens: torch.Tensor, block_size: int):
        if tokens.dim() != 1:
            raise ValueError("tokens must be 1-D.")
        if block_size <= 1:
            raise ValueError("block_size must be >1.")
        self.tokens = tokens
        self.block_size = block_size
        self.n = (len(tokens) - 1) // block_size
        if self.n <= 0:
            raise ValueError("Not enough tokens for one block.")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size
        x = self.tokens[start:end]
        y = self.tokens[start + 1 : end + 1]
        return x, y


def tokenize_texts(tokenizer, texts: Sequence[str]) -> torch.Tensor:
    token_ids = []
    for txt in texts:
        enc = tokenizer(
            txt,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=False,
            max_length=None,
        )
        token_ids.append(enc["input_ids"].squeeze(0))
    return torch.cat(token_ids, dim=0)

