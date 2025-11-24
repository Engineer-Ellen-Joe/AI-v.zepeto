from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from perceptron_cepta import CeptaConfig, CeptaEmbedding
from cepta_ssm import CeptaSSMLiteLowRank


@dataclass
class CeptaModelConfig:
    vocab_size: int
    P: int                  # 경로(퍼셉트론) 수
    alpha: int              # 팬아웃(출력 기저 수) – 여기선 주로 임베딩 구조에만 사용
    num_layers: int = 4
    P_r: int = 64           # SSM 상태 차원
    use_fp16: bool = False
    use_bf16: bool = True


class PathMLP(nn.Module):
    def __init__(self, P: int, expansion: float = 2.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(P * expansion)
        self.fc1 = nn.Linear(P, hidden)
        self.fc2 = nn.Linear(hidden, P)
        self.dropout = nn.Dropout(dropout)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,T,P)
        x = self.fc1(t)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CeptaBlock(nn.Module):
    """
    한 블록:
      1) 시간축 SSM (경로공간)
      2) 경로 MLP
    """
    def __init__(self, P: int, P_r: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(P)
        self.ssm = CeptaSSMLiteLowRank(P=P, P_r=P_r, residual=True)
        self.norm2 = nn.LayerNorm(P)
        self.mlp = PathMLP(P, expansion=2.0, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, t: torch.Tensor, F_gate: torch.Tensor,
                cache: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # t, F_gate: (B,T,P)
        # 1) SSM (Pre-LN)
        h = self.norm1(t)
        h_ssm, new_cache = self.ssm(h, F_gate, cache=cache)
        t = t + self.dropout(h_ssm)

        # 2) Path-MLP (Pre-LN)
        h2 = self.norm2(t)
        h_mlp = self.mlp(h2)
        t = t + self.dropout(h_mlp)
        return t, new_cache


class CeptaPathTransformerLM(nn.Module):
    """
    CEPTA-Path 기반 LM:
      - 토큰 → CEPTA 임베딩 (Index)
      - t0 = F⊙U (경로 활성)
      - L개의 CeptaBlock으로 t를 갱신
      - 최종 t → vocab projection
    """
    def __init__(self, cfg: CeptaModelConfig):
        super().__init__()
        self.cfg = cfg

        cepta_cfg = CeptaConfig(
            P=cfg.P,
            d_or_vocab=cfg.vocab_size,
            alpha=cfg.alpha,
            use_index=True,
            gate='hard',
            ste_tau=0.0,
            ste_gamma=1.0,
            dtype_store='bf16' if cfg.use_bf16 else ('fp16' if cfg.use_fp16 else 'fp32')
        )
        self.cepta_emb = CeptaEmbedding(cepta_cfg)

        self.blocks = nn.ModuleList(
            [CeptaBlock(P=cfg.P, P_r=cfg.P_r, dropout=0.0) for _ in range(cfg.num_layers)]
        )

        # LM Head: 경로공간 P → vocab
        self.ln_f = nn.LayerNorm(cfg.P)
        self.lm_head = nn.Linear(cfg.P, cfg.vocab_size, bias=False)

    def forward(self,
                input_ids: torch.Tensor,
                cache_states: Optional[list] = None
                ) -> Tuple[torch.Tensor, list]:
        """
        Args:
            input_ids: (B,T)
            cache_states: [ (B,P_r), ... ] or None

        Returns:
            logits: (B,T,vocab_size)
            new_cache_states: list of (B,P_r)
        """
        B, T = input_ids.shape
        device = input_ids.device

        # 1) CEPTA 임베딩
        U, Fhard, Y = self.cepta_emb(input_ids=input_ids)  # U,F: (B,T,P)
        t = (Fhard * U).to(dtype=torch.float32)            # 경로 발화 신호

        # 2) 블록 순회
        if cache_states is None:
            cache_states = [None] * len(self.blocks)

        new_cache_states = []
        for blk, cache in zip(self.blocks, cache_states):
            t, new_cache = blk(t, F_gate=Fhard, cache=cache)
            new_cache_states.append(new_cache)

        # 3) 출력 투영
        h = self.ln_f(t)
        logits = self.lm_head(h)  # (B,T,vocab)

        return logits, new_cache_states
