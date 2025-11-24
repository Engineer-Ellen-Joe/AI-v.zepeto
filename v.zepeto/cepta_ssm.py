"""
Low-rank Cross-Path SSM layer for CEPTA.

Implements the temporal/state mixing described in the specification:
    r_t = t_t @ V_r
    a_t = sigmoid(r_t @ W_lambda + b_lambda)
    s_t = a_t * s_{t-1} + (F_t ⊙ t_t) @ V_b / sqrt(P_r)
    t_tilde = s_t @ V_o
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class CeptaSSMLowRank(nn.Module):
    """Low-rank cross-path state space mixer."""

    def __init__(
        self,
        P: int,
        P_r: int,
        a_min: float = 0.01,
        a_max: float = 0.995,
        tau_init: float = 64.0,
        residual: bool = True,
        dtype_state: torch.dtype = torch.float32,
        state_rms_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        if P <= 0 or P_r <= 0:
            raise ValueError("P and P_r must be positive.")
        self.P = P
        self.P_r = P_r
        self.a_min = a_min
        self.a_max = a_max
        self.residual = residual
        self.state_rms_norm = state_rms_norm
        self.eps = eps

        factory_kwargs = {"dtype": dtype_state}
        self.V_r = nn.Parameter(torch.empty(P, P_r, **factory_kwargs))
        self.V_b = nn.Parameter(torch.empty(P, P_r, **factory_kwargs))
        self.V_o = nn.Parameter(torch.empty(P_r, P, **factory_kwargs))
        self.W_lambda = nn.Parameter(torch.empty(P_r, P_r, **factory_kwargs))
        self.b_lambda = nn.Parameter(torch.empty(P_r, **factory_kwargs))
        self.reset_parameters(tau_init=tau_init)

    def reset_parameters(self, tau_init: float = 64.0) -> None:
        nn.init.xavier_uniform_(self.V_r)
        nn.init.xavier_uniform_(self.V_b)
        nn.init.xavier_uniform_(self.V_o)
        nn.init.zeros_(self.W_lambda)
        a0 = math.exp(-1.0 / max(tau_init, 1.0))
        logit_a0 = math.log(a0 / (1.0 - a0 + 1e-9))
        nn.init.constant_(self.b_lambda, logit_a0)

    def forward(
        self,
        t: Tensor,
        F: Optional[Tensor] = None,
        cache: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Mix path signals over time.

        Args:
            t: (B, P) or (B, T, P) path potentials (typically F⊙U).
            F: Optional gate mask same shape as t. If None, treated as ones.
            cache: Optional previous state (B, P_r) for streaming.

        Returns:
            t_tilde: Mixed path signals, same rank as t.
            new_cache: Final state (B, P_r).
        """
        if t.dim() not in (2, 3):
            raise ValueError("t must be rank-2 or rank-3 (B,P) or (B,T,P).")
        original_rank2 = t.dim() == 2
        if original_rank2:
            t = t.unsqueeze(1)
        B, T, P = t.shape
        if P != self.P:
            raise ValueError(f"Expected P={self.P}, got {P}.")

        t_fp = t.float()
        if F is None:
            F_fp = torch.ones_like(t_fp)
        else:
            if F.shape != t.shape:
                raise ValueError("F must match shape of t.")
            F_fp = F.detach().float()

        if cache is not None and cache.shape != (B, self.P_r):
            raise ValueError("cache must be (B, P_r).")

        V_r = self.V_r.float()
        V_b = self.V_b.float()
        V_o = self.V_o.float()
        W_l = self.W_lambda.float()
        b_l = self.b_lambda.float()

        r = torch.matmul(t_fp, V_r)  # (B, T, P_r)
        a = torch.sigmoid(torch.matmul(r, W_l) + b_l)  # (B, T, P_r)
        a = torch.clamp(a, min=self.a_min, max=self.a_max)

        states = []
        s_prev = cache.float() if cache is not None else torch.zeros(
            B, self.P_r, device=t.device, dtype=torch.float32
        )
        scale = 1.0 / math.sqrt(self.P_r)

        for i in range(T):
            t_i = t_fp[:, i, :]
            F_i = F_fp[:, i, :]
            a_i = a[:, i, :]
            s_prev = a_i * s_prev + torch.matmul(F_i * t_i, V_b) * scale
            if self.state_rms_norm:
                denom = torch.sqrt(
                    s_prev.pow(2).mean(dim=-1, keepdim=True) + self.eps
                )
                s_prev = s_prev / denom
            states.append(s_prev)

        s = torch.stack(states, dim=1)  # (B, T, P_r)
        t_tilde = torch.matmul(s, V_o)  # (B, T, P)
        if self.residual:
            t_tilde = t_tilde + t_fp

        if original_rank2:
            t_tilde_out = t_tilde[:, 0, :]
        else:
            t_tilde_out = t_tilde
        new_cache = s[:, -1, :]
        return t_tilde_out, new_cache

