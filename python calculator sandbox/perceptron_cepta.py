# -*- coding: utf-8 -*-
"""
perceptron_cepta.py

오로지 CEPTA 퍼셉트론 구성요소만 포함:
- Custom autograd(Function) 두 경로: Dense, Index(gather)
- CEPTA 임베딩(CeptaEmbedding): W,f 를 nn.Parameter 로 보관. 게이트 F 는 detach 상수
- 경로기반 라우팅(CeptaRouting): 열 소프트맥스 또는 column top-k 마스크

다음 항목은 포함하지 않음: 토크나이저, 위치부여, 트랜스포머 블록/모델, 학습 루프, MLP, RMSNorm 등
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "CeptaConfig",
    "CeptaEmbedding",
    "CeptaRouting",
    "CeptaSSMLiteLowRank",
]

# ------------------------------------------------------------
# 내부 유틸 (시간/토큰 평탄화)
# ------------------------------------------------------------

def _flat_time(x: torch.Tensor) -> Tuple[torch.Tensor, int]:
    if x.dim() == 3:
        B, T, d = x.shape
        return x.reshape(B * T, d), T
    elif x.dim() == 2:
        return x, 1
    else:
        raise ValueError("X must be (B,d) or (B,T,d)")


def _flat_ids(ids: torch.Tensor) -> Tuple[torch.Tensor, int]:
    if ids.dim() == 2:
        B, T = ids.shape
        return ids.reshape(B * T), T
    elif ids.dim() == 1:
        return ids, 1
    else:
        raise ValueError("input_ids must be (B,) or (B,T)")


# ------------------------------------------------------------
# CEPTA 임베딩용 Custom Autograd — 엄격한 기울기 규정
# ------------------------------------------------------------
class _CeptaDenseFn(torch.autograd.Function):
    """
    Dense 경로: U = X @ W^T,  F = 1[U>=SP] (detach),  t = F * U,  Y = t ⊗ f

    Backward(게이트 상수 가정):
      G_t = einsum_{α}(gY * f)
      dW = (G_t ⊙ F_eff)^T @ X
      df = t^T @ gY
      dX = (G_t ⊙ F_eff) @ W
      dSP = 0  (hard) | STE 선택 시 유도식 반영

    ste_mode ∈ {'none','band','sigmoid'}
    update_mode_W,f ∈ {'all','active','inactive'}  (활성 기준은 F 평균>0)
    """

    @staticmethod
    def forward(ctx,
                X: torch.Tensor,
                W: torch.Tensor,
                f: torch.Tensor,
                SP: torch.Tensor,
                update_mode_W: Literal['all','active','inactive'] = 'all',
                update_mode_f: Literal['all','active','inactive'] = 'all',
                ste_mode:      Literal['none','band','sigmoid'] = 'none',
                ste_tau: float = 0.0,
                ste_gamma: float = 1.0,
                ):
        assert X.dim() == 2 and W.shape[0] == f.shape[0] == SP.shape[0]
        B, d = X.shape
        P = W.shape[0]
        U = X.matmul(W.t())  # (B,P)
        Fhard = (U >= SP.view(1, P)).to(dtype=torch.float32).detach()
        t = Fhard * U
        Y = t.unsqueeze(-1) * f.unsqueeze(0)  # (B,P,α)

        active_vec = (Fhard.mean(dim=0) > 0).to(torch.float32)  # (P,)
        M_W = active_vec if update_mode_W == 'active' else (1.0 - active_vec if update_mode_W == 'inactive' else torch.ones_like(active_vec))
        M_f = active_vec if update_mode_f == 'active' else (1.0 - active_vec if update_mode_f == 'inactive' else torch.ones_like(active_vec))

        ctx.save_for_backward(X, W, f, SP, U, Fhard, M_W, M_f)
        ctx.ste_mode = ste_mode
        ctx.ste_tau = float(ste_tau)
        ctx.ste_gamma = float(ste_gamma)
        return U, Fhard.to(dtype=Y.dtype), Y

    @staticmethod
    def backward(ctx, gU: torch.Tensor, _gF: torch.Tensor, gY: torch.Tensor):
        X, W, f, SP, U, Fhard, M_W, M_f = ctx.saved_tensors
        G_t = torch.einsum('bpa,pa->bp', gY.float(), f.float())  # (B,P)

        if ctx.ste_mode != 'none':
            z = (U - SP.view(1, -1)).float()
            if ctx.ste_mode == 'band':
                F_ste = (z.abs() <= ctx.ste_tau).to(z.dtype)
            else:
                sig = torch.sigmoid(ctx.ste_gamma * z)
                F_ste = ctx.ste_gamma * sig * (1.0 - sig)
            F_eff = Fhard.float() + F_ste
        else:
            F_eff = Fhard.float()

        G_U = G_t * F_eff  # (B,P)

        dW = G_U.t().matmul(X.float())            # (P,d)
        dW = (dW.t() * M_W.float()).t()           # 행 마스크를 기울기 수식에 포함

        t = Fhard.float() * U.float()
        df = torch.einsum('bp,bpa->pa', t, gY.float())
        df = (df.t() * M_f.float()).t()

        dX = G_U.matmul(W.float())

        if ctx.ste_mode == 'none':
            dSP = torch.zeros_like(SP)
        else:
            dL_dt = G_t
            dF_dU = (F_eff - Fhard.float())
            dSP = - (dL_dt * dF_dU * U.float()).sum(dim=0)

        return dX, dW, df, dSP, None, None, None, None, None


class _CeptaIndexFn(torch.autograd.Function):
    """Index 경로: U[b*,p] = W[p, tok[b*]]  (W: (P,V))"""

    @staticmethod
    def forward(ctx,
                input_ids: torch.Tensor,
                W: torch.Tensor,
                f: torch.Tensor,
                SP: torch.Tensor,
                update_mode_W: Literal['all','active','inactive'] = 'all',
                update_mode_f: Literal['all','active','inactive'] = 'all',
                ste_mode:      Literal['none','band','sigmoid'] = 'none',
                ste_tau: float = 0.0,
                ste_gamma: float = 1.0,
                ):
        assert input_ids.dim() == 1 and W.dim() == 2
        Bstar = input_ids.shape[0]
        P, V = W.shape
        U = W.index_select(dim=1, index=input_ids).t()  # (B*,P)
        Fhard = (U >= SP.view(1, P)).to(dtype=torch.float32).detach()
        t = Fhard * U
        Y = t.unsqueeze(-1) * f.unsqueeze(0)

        active_vec = (Fhard.mean(dim=0) > 0).to(torch.float32)
        M_W = active_vec if update_mode_W == 'active' else (1.0 - active_vec if update_mode_W == 'inactive' else torch.ones_like(active_vec))
        M_f = active_vec if update_mode_f == 'active' else (1.0 - active_vec if update_mode_f == 'inactive' else torch.ones_like(active_vec))

        ctx.save_for_backward(input_ids, W, f, SP, U, Fhard, M_W, M_f)
        ctx.P = P
        ctx.V = V
        ctx.ste_mode = ste_mode
        ctx.ste_tau = float(ste_tau)
        ctx.ste_gamma = float(ste_gamma)
        return U, Fhard.to(dtype=Y.dtype), Y

    @staticmethod
    def backward(ctx, gU: torch.Tensor, _gF: torch.Tensor, gY: torch.Tensor):
        tok, W, f, SP, U, Fhard, M_W, M_f = ctx.saved_tensors
        P, V = ctx.P, ctx.V
        G_t = torch.einsum('bpa,pa->bp', gY.float(), f.float())

        if ctx.ste_mode != 'none':
            z = (U - SP.view(1, -1)).float()
            if ctx.ste_mode == 'band':
                F_ste = (z.abs() <= ctx.ste_tau).to(z.dtype)
            else:
                sig = torch.sigmoid(ctx.ste_gamma * z)
                F_ste = ctx.ste_gamma * sig * (1.0 - sig)
            F_eff = Fhard.float() + F_ste
        else:
            F_eff = Fhard.float()
        G_U = G_t * F_eff

        dW = torch.zeros((P, V), dtype=torch.float32, device=W.device)
        dW.scatter_add_(dim=1, index=tok.view(1, -1).expand(P, -1), src=G_U.t())
        dW = (dW.t() * M_W.float()).t()

        t = Fhard.float() * U.float()
        df = torch.einsum('bp,bpa->pa', t, gY.float())
        df = (df.t() * M_f.float()).t()

        dTok = None

        if ctx.ste_mode == 'none':
            dSP = torch.zeros_like(SP)
        else:
            dL_dt = G_t
            dF_dU = (F_eff - Fhard.float())
            dSP = - (dL_dt * dF_dU * U.float()).sum(dim=0)

        return dTok, dW, df, dSP, None, None, None, None, None


# ------------------------------------------------------------
# CEPTA 임베딩 모듈
# ------------------------------------------------------------
@dataclass
class CeptaConfig:
    P: int
    d_or_vocab: int
    alpha: int
    use_index: bool = False
    gate: Literal['hard','ste_band','ste_sigmoid'] = 'hard'
    ste_tau: float = 0.0
    ste_gamma: float = 1.0
    dtype_store: Literal['bf16','fp16','fp32'] = 'bf16'


class CeptaEmbedding(nn.Module):
    """CEPTA 임베딩: (U,F,t,Y) 생성. 엄격한 그라디언트 구현 포함.
    - use_index=True: input_ids 필요. False: X_dense 필요.
    - 파라미터 저장 dtype은 bf16/fp16/fp32 선택. 연산/축적은 float32.
    """
    def __init__(self, cfg: CeptaConfig):
        super().__init__()
        self.cfg = cfg
        P, D, A = cfg.P, cfg.d_or_vocab, cfg.alpha
        if cfg.dtype_store == 'bf16':
            p_dtype = torch.bfloat16
        elif cfg.dtype_store == 'fp16':
            p_dtype = torch.float16
        else:
            p_dtype = torch.float32
        self.W = nn.Parameter(torch.empty(P, D, dtype=p_dtype))
        self.f = nn.Parameter(torch.empty(P, A, dtype=p_dtype))
        self.SP = nn.Parameter(torch.zeros(P, dtype=torch.float32))  # 임계값은 fp32
        self.reset_parameters()
        self.update_mode_W: Literal['all','active','inactive'] = 'all'
        self.update_mode_f: Literal['all','active','inactive'] = 'all'

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.float())
        nn.init.xavier_uniform_(self.f.float())
        nn.init.zeros_(self.SP)

    def forward(self,
                *,
                X_dense: Optional[torch.Tensor] = None,
                input_ids: Optional[torch.Tensor] = None,
                ):
        cfg = self.cfg
        gate = cfg.gate
        ste_tag = 'none' if gate == 'hard' else ('band' if gate == 'ste_band' else 'sigmoid')
        if cfg.use_index:
            if X_dense is not None or input_ids is None:
                raise ValueError("use_index=True 이면 input_ids만 제공해야 합니다.")
            tok_flat, T = _flat_ids(input_ids)
            U, Fhard, Y = _CeptaIndexFn.apply(
                tok_flat,
                self.W.float(),
                self.f.float(),
                self.SP,
                self.update_mode_W,
                self.update_mode_f,
                ste_tag,
                cfg.ste_tau,
                cfg.ste_gamma,
            )
            if T > 1:
                P = self.cfg.P
                A = self.cfg.alpha
                U = U.view(-1, T, P)
                Fhard = Fhard.view(-1, T, P)
                Y = Y.view(-1, T, P, A)
            return U, Fhard, Y
        else:
            if input_ids is not None or X_dense is None:
                raise ValueError("use_index=False 이면 X_dense만 제공해야 합니다.")
            Xf, T = _flat_time(X_dense)
            U, Fhard, Y = _CeptaDenseFn.apply(
                Xf.float(),
                self.W.float(),
                self.f.float(),
                self.SP,
                self.update_mode_W,
                self.update_mode_f,
                ste_tag,
                cfg.ste_tau,
                cfg.ste_gamma,
            )
            if T > 1:
                P = self.cfg.P
                A = self.cfg.alpha
                U = U.view(-1, T, P)
                Fhard = Fhard.view(-1, T, P)
                Y = Y.view(-1, T, P, A)
            return U, Fhard, Y


# ------------------------------------------------------------
# 경로기반 라우팅 A_l (열 softmax 또는 column top-k)
# ------------------------------------------------------------
class CeptaRouting(nn.Module):
    def __init__(self, P_in: int, P_out: int, *, mode: Literal['softmax','topk'] = 'softmax', topk: int = 8):
        super().__init__()
        self.P_in = P_in
        self.P_out = P_out
        self.mode = mode
        self.topk = int(topk)
        self.A = nn.Parameter(torch.empty(P_in, P_out))
        nn.init.xavier_uniform_(self.A)

    def col_softmax(self, A: torch.Tensor) -> torch.Tensor:
        return (A.float() - A.float().max(dim=0, keepdim=True).values).softmax(dim=0)

    def col_topk_mask(self, A: torch.Tensor) -> torch.Tensor:
        k = min(self.topk, A.shape[0])
        vals, idx = torch.topk(A.float(), k=k, dim=0)
        M = torch.zeros_like(A, dtype=torch.float32)
        M.scatter_(dim=0, index=idx, src=torch.ones_like(vals))
        return M

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        A = self.A
        if self.mode == 'softmax':
            Atilde = self.col_softmax(A)
        else:
            M = self.col_topk_mask(A)
            col_sum = (A.float() * M).sum(dim=0, keepdim=True).clamp_min(1e-9)
            Atilde = (A.float() * M) / col_sum
        if t.dim() == 2:
            return t.float().matmul(Atilde)
        elif t.dim() == 3:
            B, T, Pin = t.shape
            return t.reshape(B * T, Pin).float().matmul(Atilde).view(B, T, self.P_out)
        else:
            raise ValueError("t must be (B*,P) or (B,T,P)")


# ------------------------------------------------------------
# 시간축 SSM-lite (Low-Rank Cross-Path)
#   - 경로 공간(P) 저랭크 투영(P_r)에서 상태 누적
#   - 발화 게이트 F로 주입 항을 하드 마스킹(F.detach())
#   - 입력 형상: (B,P) 또는 (B,T,P)  출력 동일 + cache(B,P_r)
# ------------------------------------------------------------
class CeptaSSMLiteLowRank(nn.Module):
    def __init__(self, P: int, P_r: int = 64,
                 a_min: float = 0.01, a_max: float = 0.995,
                 tau_init: float = 64.0,
                 residual: bool = False):
        super().__init__()
        assert P_r > 0 and P > 0
        self.P = P
        self.P_r = P_r
        self.a_min = float(a_min)
        self.a_max = float(a_max)
        self.residual = bool(residual)

        # 저랭크 투영 및 회복 행렬
        self.V_r = nn.Parameter(torch.empty(P, P_r))   # t -> r_t (게이트용)
        self.V_b = nn.Parameter(torch.empty(P, P_r))   # t -> 주입 b_t
        self.V_o = nn.Parameter(torch.empty(P_r, P))   # s_t -> ~t
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
        b = torch.log(torch.tensor(a_bar / (1.0 - a_bar)))
        with torch.no_grad():
            self.W_lambda.bias.copy_(b.expand(self.P_r))

    def forward(self, t: torch.Tensor, F: Optional[torch.Tensor] = None,
                cache: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """SSM-lite 저랭크 누적.
        Args:
            t: (B,P) or (B,T,P)
            F: (B,P) or (B,T,P). None이면 1로 처리. 반드시 detach된 게이트 사용 권장.
            cache: (B,P_r) 이전 상태. None이면 0으로 초기화.
        Returns:
            t_tilde: (B,P) or (B,T,P)
            new_cache: (B,P_r)
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
            t_f = t_f.unsqueeze(1)
            F_f = F_f.unsqueeze(1)
            squeeze = True
        elif t_f.dim() == 3:
            B, T, P = t_f.shape
        else:
            raise ValueError("t must be (B,P) or (B,T,P)")

        V_r = self.V_r.float()
        V_b = self.V_b.float()
        V_o = self.V_o.float()

        # r_t = t V_r,  a_t = clamp(sigmoid(W_lambda(r_t)))
        r = torch.matmul(t_f, V_r)  # (B,T,P_r)
        a = torch.sigmoid(self.W_lambda(r))
        a = torch.clamp(a, self.a_min, self.a_max)

        # 주입: 발화한 경로만 누적(F로 하드 마스킹)
        inj = torch.matmul(F_f * t_f, V_b)  # (B,T,P_r)

        # 상태 스캔
        if cache is None:
            s_prev = torch.zeros((B, self.P_r), device=t.device, dtype=torch.float32)
        else:
            s_prev = cache.float()

        outs = []
        for i in range(T):
            s_prev = a[:, i, :] * s_prev + inj[:, i, :]
            outs.append(s_prev.unsqueeze(1))
        s = torch.cat(outs, dim=1)  # (B,T,P_r)

        # 경로 공간으로 복귀
        t_tilde = torch.matmul(s, V_o)  # (B,T,P)
        if self.residual:
            t_tilde = t_tilde + t_f
        new_cache = s_prev  # (B,P_r)

        if squeeze:
            t_tilde = t_tilde.squeeze(1)
        return t_tilde.to(orig_dtype), new_cache
