from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn


class CeptaSSMLiteLowRank(nn.Module):
    def __init__(self,
                 P: int,
                 P_r: int = 64,
                 a_min: float = 0.01,
                 a_max: float = 0.995,
                 tau_init: float = 64.0,
                 residual: bool = False):
        """
        Args:
            P: 경로(퍼셉트론) 수
            P_r: 저랭크 상태 차원 (≪ P)
            a_min, a_max: forget gate 클램프 범위
            tau_init: 초기 기억 길이(대략적인 평균 time constant)
            residual: True면 t̃_t + t_t로 잔차 추가
        """
        super().__init__()
        assert P > 0 and P_r > 0
        self.P = P
        self.P_r = P_r
        self.a_min = float(a_min)
        self.a_max = float(a_max)
        self.residual = bool(residual)

        # 저랭크 투영 및 회복 행렬
        self.V_r = nn.Parameter(torch.empty(P, P_r))   # t -> r_t (게이트용)
        self.V_b = nn.Parameter(torch.empty(P, P_r))   # t -> 주입 b_t
        self.V_o = nn.Parameter(torch.empty(P_r, P))   # s_t -> t̃_t
        # 게이트: r_t -> a_t
        self.W_lambda = nn.Linear(P_r, P_r, bias=True)

        self.reset_parameters(tau_init)

    def reset_parameters(self, tau_init: float = 64.0):
        nn.init.xavier_uniform_(self.V_r)
        nn.init.xavier_uniform_(self.V_b)
        nn.init.xavier_uniform_(self.V_o)
        nn.init.xavier_uniform_(self.W_lambda.weight)
        # 목표 기억 길이 tau_init에 대응하는 평균 망각율 ā = 1 - 1/tau
        tau = max(float(tau_init), 1.0)
        a_bar = 1.0 - 1.0 / tau
        a_bar = float(max(1e-3, min(1.0 - 1e-3, a_bar)))
        b = torch.log(torch.tensor(a_bar / (1.0 - a_bar)))
        with torch.no_grad():
            self.W_lambda.bias.copy_(b.expand(self.P_r))

    def forward(self,
                t: torch.Tensor,
                F: Optional[torch.Tensor] = None,
                cache: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            t: (B,P) or (B,T,P) 경로 발화 신호 (보통 t = F⊙U)
            F: (B,P) or (B,T,P) 게이트 (0/1). None이면 1로 처리.
               반드시 detach된 게이트 사용 권장.
            cache: (B,P_r) 이전 상태 (스트리밍용). None이면 0으로 초기화.

        Returns:
            t_tilde: (B,P) or (B,T,P)  (문맥이 반영된 경로 신호)
            new_cache: (B,P_r) 마지막 시점 상태
        """
        orig_dtype = t.dtype
        t_f = t.float()
        if F is None:
            F_f = torch.ones_like(t_f)
        else:
            F_f = F.detach().to(dtype=torch.float32)

        squeeze = False
        if t_f.dim() == 2:
            B, P = t_f.shape
            T = 1
            t_f = t_f.unsqueeze(1)   # (B,1,P)
            F_f = F_f.unsqueeze(1)
            squeeze = True
        elif t_f.dim() == 3:
            B, T, P = t_f.shape
        else:
            raise ValueError("t must be (B,P) or (B,T,P)")

        if P != self.P:
            raise ValueError(f"P mismatch: input P={P}, module P={self.P}")

        V_r = self.V_r.float()
        V_b = self.V_b.float()
        V_o = self.V_o.float()

        # 1) 저랭크 투영 및 forget gate
        r = torch.matmul(t_f, V_r)                 # (B,T,P_r)
        a = torch.sigmoid(self.W_lambda(r))        # (B,T,P_r)
        a = torch.clamp(a, self.a_min, self.a_max) # 안정화

        # 2) 발화한 경로만 상태에 주입
        inj = torch.matmul(F_f * t_f, V_b)         # (B,T,P_r)
        inj = inj * (1.0 / max(1.0, float(self.P_r)) ** 0.5)

        # 3) 시간축 상태 스캔
        if cache is None:
            s_prev = torch.zeros((B, self.P_r), device=t.device, dtype=torch.float32)
        else:
            s_prev = cache.float()

        outs = []
        for i in range(T):
            s_prev = a[:, i, :] * s_prev + inj[:, i, :]
            outs.append(s_prev.unsqueeze(1))
        s = torch.cat(outs, dim=1)   # (B,T,P_r)

        # 4) 상태 RMS 정규화
        rms = (s.pow(2).mean(dim=-1, keepdim=True).add_(1e-6)).rsqrt_()
        s = s * rms

        # 5) 경로공간 복귀
        t_tilde = torch.matmul(s, V_o)  # (B,T,P)
        if self.residual:
            t_tilde = t_tilde + t_f

        new_cache = s_prev  # (B,P_r)

        if squeeze:
            t_tilde = t_tilde.squeeze(1)
        return t_tilde.to(orig_dtype), new_cache
