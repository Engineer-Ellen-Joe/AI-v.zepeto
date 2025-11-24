from __future__ import annotations
from typing import List

from transformers import AutoTokenizer


_MODEL_NAME = "deepseek-ai/DeepSeek-V3"


def get_tokenizer():
    """
    DeepSeek V3 토크나이저 인스턴스를 반환.
    """
    tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
    return tokenizer


def encode(texts, tokenizer=None, add_special_tokens: bool = True, max_length: int = 512):
    """
    문자열 또는 문자열 리스트를 input_ids 텐서로 변환.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()
    return tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=add_special_tokens,
    )


def decode(token_ids, tokenizer=None, skip_special_tokens: bool = True) -> List[str]:
    if tokenizer is None:
        tokenizer = get_tokenizer()
    if token_ids.ndim == 1:
        token_ids = token_ids.unsqueeze(0)
    return [tokenizer.decode(row, skip_special_tokens=skip_special_tokens) for row in token_ids]
