"""
Tokenizer utilities for DeepSeek-V3.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
from transformers import AutoTokenizer


def get_deepseek_v3_tokenizer():
    """Load DeepSeek-V3 tokenizer with safe padding defaults."""
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def encode_texts(
    tokenizer, texts: List[str], max_length: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode a batch of texts into token IDs and attention masks."""
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return encoded["input_ids"], encoded["attention_mask"]


def decode_tokens(tokenizer, input_ids) -> List[str]:
    """Decode token IDs back to text."""
    return tokenizer.batch_decode(input_ids, skip_special_tokens=True)

