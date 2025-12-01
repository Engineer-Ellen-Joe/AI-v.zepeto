"""
CEPTA-based Transformer blocks leveraging CEPTA perceptron + SSM context mixing.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cepta_ssm import CeptaSSMLowRank
from embedding import CeptaEmbeddingConfig, CeptaTokenEmbedding
from perceptron_cepta import CeptaConfig, CeptaEmbedding, CeptaRouting


class RMSNorm(nn.Module):
    """DeepSeek-style RMSNorm."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * self.weight / norm


class FeedForward(nn.Module):
    """GELU MLP."""

    def __init__(self, dim: int, multiple: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * multiple)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CeptaContextBlock(nn.Module):
    """CEPTA dense pathway + low-rank SSM context mixer."""

    def __init__(self, d_model: int, P: int, alpha: int, P_r: int, dropout: float = 0.0):
        super().__init__()
        self.to_P = nn.Linear(d_model, P, bias=False)
        cepta_cfg = CeptaConfig(
            P=P,
            d_or_vocab=d_model,
            alpha=alpha,
            use_index=False,
        )
        self.cepta_dense = CeptaEmbedding(cepta_cfg)
        self.router = CeptaRouting(reduce="sum")
        self.ssm = CeptaSSMLowRank(P=P, P_r=P_r)
        self.from_P = nn.Linear(P, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_pack: bool = False):
        if x.dim() != 3:
            raise ValueError("Input to CeptaContextBlock must be (B, T, D).")
        pre_path = self.to_P(x)
        U, F, Y = self.cepta_dense(X_dense=x)
        t = self.router(Y) + pre_path
        t_tilde, _ = self.ssm(t, F=F)
        h = self.from_P(t_tilde)
        h = self.dropout(h)
        if return_pack:
            pack = {"U": U, "F": F, "Y": Y, "t": t, "t_tilde": t_tilde, "X_dense": x}
            return h, pack
        return h


class CeptaTransformerBlock(nn.Module):
    """Pre-LN block with CEPTA context and MLP."""

    def __init__(
        self,
        d_model: int,
        P: int,
        alpha: int,
        P_r: int,
        dropout: float = 0.0,
        mlp_multiple: float = 16.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.context = CeptaContextBlock(d_model, P, alpha, P_r, dropout=dropout)
        self.norm2 = RMSNorm(d_model)
        self.mlp = FeedForward(d_model, multiple=mlp_multiple, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_pack: bool = False):
        h1 = self.norm1(x)
        ctx_out = self.context(h1, return_pack=return_pack)
        if return_pack:
            y, ctx_pack = ctx_out
        else:
            y = ctx_out
        y = self.dropout(y)
        x = x + y

        h2 = self.norm2(x)
        z = self.mlp(h2)
        z = self.dropout(z)
        x = x + z
        if return_pack:
            return x, {"context": ctx_pack}
        return x


class CeptaTransformerLM(nn.Module):
    """Stacked CEPTA-based transformer language model."""

    def __init__(self, emb_cfg: CeptaEmbeddingConfig, n_layers: int):
        super().__init__()
        self.embedding = CeptaTokenEmbedding(emb_cfg)
        self.blocks = nn.ModuleList(
            [
                CeptaTransformerBlock(
                    d_model=emb_cfg.d_model,
                    P=emb_cfg.P,
                    alpha=emb_cfg.alpha,
                    P_r=emb_cfg.P_r,
                    dropout=0.0,
                    mlp_multiple=16.0,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = RMSNorm(emb_cfg.d_model)
        self.lm_head = nn.Linear(emb_cfg.d_model, emb_cfg.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_packs: bool = False,
    ):
        x, emb_pack = self.embedding(input_ids, return_pack=return_packs)
        packs = {"embedding": emb_pack, "blocks": []} if return_packs else None
        for block in self.blocks:
            if return_packs:
                x, blk_pack = block(x, return_pack=True)
                packs["blocks"].append(blk_pack)
            else:
                x = block(x, return_pack=False)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        if return_packs:
            return logits, packs
        return logits
