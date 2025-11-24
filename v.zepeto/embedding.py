"""
High-level CEPTA embedding: tokens -> path activations -> hidden vectors.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import Tensor, nn

from cepta_ssm import CeptaSSMLowRank
from perceptron_cepta import CeptaConfig, CeptaEmbedding


def _dtype_from_str(name: str) -> torch.dtype:
    name = name.lower()
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


@dataclass
class CeptaEmbeddingConfig:
    vocab_size: int
    P: int
    alpha: int
    P_r: int
    d_model: int
    max_seq_len: int
    dtype_store: str = "fp32"
    gate: str = "hard"
    ste_band_tau: float = 1.0
    ste_sigmoid_gamma: float = 5.0
    update_mode_W: str = "all"
    update_mode_f: str = "all"


class SinusoidalPositionalEncoding(nn.Module):
    """Absolute sinusoidal position encoding."""

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError("Positional encoding expects (B, T, D).")
        T = x.size(1)
        if T > self.pe.size(0):
            raise ValueError(f"Sequence length {T} exceeds max_len {self.pe.size(0)}.")
        return x + self.pe[:T].unsqueeze(0).to(dtype=x.dtype, device=x.device)


class CeptaTokenEmbedding(nn.Module):
    """Token -> CEPTA index pathway -> SSM -> hidden vector."""

    def __init__(self, cfg: CeptaEmbeddingConfig):
        super().__init__()
        dtype_store = _dtype_from_str(cfg.dtype_store)
        cepta_cfg = CeptaConfig(
            P=cfg.P,
            d_or_vocab=cfg.vocab_size,
            alpha=cfg.alpha,
            use_index=True,
            gate=cfg.gate,  # type: ignore[arg-type]
            ste_band_tau=cfg.ste_band_tau,
            ste_sigmoid_gamma=cfg.ste_sigmoid_gamma,
            update_mode_W=cfg.update_mode_W,  # type: ignore[arg-type]
            update_mode_f=cfg.update_mode_f,  # type: ignore[arg-type]
            dtype_store=dtype_store,
        )
        self.cepta = CeptaEmbedding(cepta_cfg)
        self.ssm = CeptaSSMLowRank(P=cfg.P, P_r=cfg.P_r)
        self.to_hidden = nn.Linear(cfg.P, cfg.d_model, bias=False)
        self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model, max_len=cfg.max_seq_len)

    def forward(self, input_ids: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be (B, T).")
        U, F, Y = self.cepta(input_ids=input_ids)
        t = F * U
        t_tilde, _ = self.ssm(t, F=F)
        x = self.to_hidden(t_tilde)
        x = self.pos_enc(x)
        pack = {"U": U, "F": F, "t": t, "t_tilde": t_tilde, "Y": Y}
        return x, pack

