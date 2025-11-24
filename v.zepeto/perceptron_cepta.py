"""
CEPTA perceptron core components.

This module implements the CEPTA path-wise perceptron with custom autograd
functions for dense and index routes, optional straight-through estimators,
row-wise update masks, and local Oja-style updates plus SP homeostasis helpers.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import torch
from torch import Tensor, nn

GateMode = Literal["hard", "ste_band", "ste_sigmoid"]
UpdateMode = Literal["all", "active", "inactive"]


@dataclass
class CeptaConfig:
    """Configuration for CEPTA perceptron.

    Args:
        P: Number of CEPTA paths/neurons.
        d_or_vocab: Feature dimension for dense mode or vocab size for index mode.
        alpha: Axon fan-out per neuron.
        use_index: If True, use index (embedding) pathway, else dense.
        gate: Gate mode, one of {"hard", "ste_band", "ste_sigmoid"}.
        ste_band_tau: Band width for band STE.
        ste_sigmoid_gamma: Gain for sigmoid STE derivative.
        update_mode_W: Row mask mode for W gradients.
        update_mode_f: Row mask mode for f gradients.
        dtype_store: Storage dtype for parameters (compute runs in fp32).
        device: Optional device placement for parameters.
    """

    P: int
    d_or_vocab: int
    alpha: int
    use_index: bool
    gate: GateMode = "hard"
    ste_band_tau: float = 1.0
    ste_sigmoid_gamma: float = 5.0
    update_mode_W: UpdateMode = "all"
    update_mode_f: UpdateMode = "all"
    dtype_store: torch.dtype = torch.float32
    device: Optional[torch.device] = None


def _compute_masks(
    Fhard: Tensor, update_mode_W: UpdateMode, update_mode_f: UpdateMode
) -> Tuple[Tensor, Tensor]:
    """Compute row-wise masks (M_W, M_f) based on neuron activity."""
    active_vec = (Fhard.mean(dim=0) > 0).float()  # (P,)
    device = Fhard.device
    ones = torch.ones_like(active_vec, device=device)

    def _choose(mode: UpdateMode) -> Tensor:
        if mode == "all":
            return ones
        if mode == "active":
            return active_vec
        if mode == "inactive":
            return ones - active_vec
        raise ValueError(f"Invalid update mode: {mode}")

    return _choose(update_mode_W), _choose(update_mode_f)


def _ste_component(
    U: Tensor,
    SP: Tensor,
    mode: Literal["none", "band", "sigmoid"],
    tau: float,
    gamma: float,
) -> Tensor:
    """Return surrogate derivative term for the gate."""
    if mode == "none":
        return torch.zeros_like(U)
    diff = U - SP
    if mode == "band":
        return (diff.abs() <= tau).float()
    sigma = torch.sigmoid(gamma * diff)
    return gamma * sigma * (1.0 - sigma)


class _CeptaDenseFn(torch.autograd.Function):
    """Custom autograd for dense CEPTA path."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        X: Tensor,
        W: Tensor,
        f: Tensor,
        SP: Tensor,
        ste_mode: Literal["none", "band", "sigmoid"],
        ste_band_tau: float,
        ste_sigmoid_gamma: float,
        update_mode_W: UpdateMode,
        update_mode_f: UpdateMode,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        X_fp = X.float()
        W_fp = W.float()
        f_fp = f.float()
        SP_fp = SP.float()

        U = X_fp @ W_fp.t()  # (B*, P)
        Fhard = (U >= SP_fp).float().detach()
        t = Fhard * U
        Y = t.unsqueeze(-1) * f_fp.unsqueeze(0)

        ctx.save_for_backward(X_fp, W_fp, f_fp, SP_fp, U, Fhard)
        ctx.ste_mode = ste_mode
        ctx.ste_band_tau = ste_band_tau
        ctx.ste_sigmoid_gamma = ste_sigmoid_gamma
        ctx.update_mode_W = update_mode_W
        ctx.update_mode_f = update_mode_f
        return U, Fhard, Y

    @staticmethod
    def backward(ctx, gU: Tensor, gF: Tensor, gY: Tensor):  # type: ignore[override]
        X, W, f, SP, U, Fhard = ctx.saved_tensors
        ste_mode: Literal["none", "band", "sigmoid"] = ctx.ste_mode
        ste_band_tau: float = ctx.ste_band_tau
        ste_sigmoid_gamma: float = ctx.ste_sigmoid_gamma
        update_mode_W: UpdateMode = ctx.update_mode_W
        update_mode_f: UpdateMode = ctx.update_mode_f

        if gY is None:
            gY = torch.zeros(
                U.shape + (f.shape[1],), device=U.device, dtype=U.dtype
            )
        # Core gradient accumulation
        G_t = torch.einsum("bpa,pa->bp", gY, f)  # (B*, P)
        F_ste = _ste_component(U, SP, ste_mode, ste_band_tau, ste_sigmoid_gamma)
        F_eff = Fhard + F_ste
        base = G_t * F_eff  # (B*, P)
        G_U = base + (gU if gU is not None else 0.0)

        dW = G_U.t() @ X  # (P, d)
        dX = G_U @ W  # (B*, d)

        t = Fhard * U
        df = torch.einsum("bp,bpa->pa", t, gY)  # (P, alpha)
        gSP = -(G_t * F_ste).sum(dim=0)  # (P,)
        if gF is not None:
            gSP = gSP - (gF * F_ste).sum(dim=0)

        M_W, M_f = _compute_masks(Fhard, update_mode_W, update_mode_f)
        dW = (dW.t() * M_W).t().to(W.dtype)
        df = (df.t() * M_f).t().to(f.dtype)

        dX = dX.to(X.dtype)
        gSP = gSP.to(SP.dtype)

        # gF is ignored (gate hard), return None to match input slots
        return dX, dW, df, gSP, None, None, None, None, None


class _CeptaIndexFn(torch.autograd.Function):
    """Custom autograd for index (embedding) CEPTA path."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        input_ids: Tensor,
        W: Tensor,
        f: Tensor,
        SP: Tensor,
        ste_mode: Literal["none", "band", "sigmoid"],
        ste_band_tau: float,
        ste_sigmoid_gamma: float,
        update_mode_W: UpdateMode,
        update_mode_f: UpdateMode,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids for index CEPTA must be integer typed.")

        W_fp = W.float()
        f_fp = f.float()
        SP_fp = SP.float()

        tokens = input_ids.view(-1)  # (B*,)
        P, V = W_fp.shape
        tok_exp = tokens.unsqueeze(0).expand(P, -1)  # (P, B*)
        gathered = torch.gather(W_fp, 1, tok_exp)  # (P, B*)
        U = gathered.t()  # (B*, P)

        Fhard = (U >= SP_fp).float().detach()
        t = Fhard * U
        Y = t.unsqueeze(-1) * f_fp.unsqueeze(0)

        ctx.save_for_backward(tokens, W_fp, f_fp, SP_fp, U, Fhard)
        ctx.ste_mode = ste_mode
        ctx.ste_band_tau = ste_band_tau
        ctx.ste_sigmoid_gamma = ste_sigmoid_gamma
        ctx.update_mode_W = update_mode_W
        ctx.update_mode_f = update_mode_f
        return U, Fhard, Y

    @staticmethod
    def backward(ctx, gU: Tensor, gF: Tensor, gY: Tensor):  # type: ignore[override]
        tokens, W, f, SP, U, Fhard = ctx.saved_tensors
        ste_mode: Literal["none", "band", "sigmoid"] = ctx.ste_mode
        ste_band_tau: float = ctx.ste_band_tau
        ste_sigmoid_gamma: float = ctx.ste_sigmoid_gamma
        update_mode_W: UpdateMode = ctx.update_mode_W
        update_mode_f: UpdateMode = ctx.update_mode_f

        if gY is None:
            gY = torch.zeros(
                U.shape + (f.shape[1],), device=U.device, dtype=U.dtype
            )
        G_t = torch.einsum("bpa,pa->bp", gY, f)  # (B*, P)
        F_ste = _ste_component(U, SP, ste_mode, ste_band_tau, ste_sigmoid_gamma)
        F_eff = Fhard + F_ste
        base = G_t * F_eff  # (B*, P)
        G_U = base + (gU if gU is not None else 0.0)

        # dW via scatter add over tokens
        P, V = W.shape
        tok_exp = tokens.unsqueeze(0).expand(P, -1)  # (P, B*)
        dW = torch.zeros_like(W)
        dW.scatter_add_(1, tok_exp, G_U.t())

        t = Fhard * U
        df = torch.einsum("bp,bpa->pa", t, gY)
        gSP = -(G_t * F_ste).sum(dim=0)
        if gF is not None:
            gSP = gSP - (gF * F_ste).sum(dim=0)

        M_W, M_f = _compute_masks(Fhard, update_mode_W, update_mode_f)
        dW = (dW.t() * M_W).t().to(W.dtype)
        df = (df.t() * M_f).t().to(f.dtype)
        gSP = gSP.to(SP.dtype)

        return None, dW, df, gSP, None, None, None, None, None


class CeptaEmbedding(nn.Module):
    """CEPTA perceptron embedding layer (dense or index).

    Forward:
        Dense: X_dense (B, T, d) -> (U, F, Y)
        Index: input_ids (B, T)  -> (U, F, Y)
        Shapes:
            U: (B, T, P), F: (B, T, P), Y: (B, T, P, alpha)
    """

    def __init__(self, cfg: CeptaConfig):
        super().__init__()
        if cfg.P <= 0 or cfg.d_or_vocab <= 0 or cfg.alpha <= 0:
            raise ValueError("P, d_or_vocab, and alpha must be positive.")
        self.cfg = cfg
        factory_kwargs = {"device": cfg.device, "dtype": cfg.dtype_store}
        self.W = nn.Parameter(torch.empty(cfg.P, cfg.d_or_vocab, **factory_kwargs))
        self.f = nn.Parameter(torch.empty(cfg.P, cfg.alpha, **factory_kwargs))
        self.SP = nn.Parameter(torch.zeros(cfg.P, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.f)
        nn.init.zeros_(self.SP)

    @property
    def _ste_mode(self) -> Literal["none", "band", "sigmoid"]:
        if self.cfg.gate == "hard":
            return "none"
        if self.cfg.gate == "ste_band":
            return "band"
        if self.cfg.gate == "ste_sigmoid":
            return "sigmoid"
        raise ValueError(f"Unsupported gate mode {self.cfg.gate}")

    def forward(
        self,
        X_dense: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute CEPTA embedding.

        Args:
            X_dense: Dense inputs (B, T, d), used when use_index=False.
            input_ids: Token IDs (B, T), used when use_index=True.

        Returns:
            U: potentials (B, T, P)
            F: hard gate (B, T, P)
            Y: fan-out outputs (B, T, P, alpha)
        """
        if self.cfg.use_index:
            if input_ids is None:
                raise ValueError("input_ids must be provided for index CEPTA.")
            if X_dense is not None:
                raise ValueError("Only input_ids should be provided for index CEPTA.")
            if input_ids.dim() != 2:
                raise ValueError("input_ids must be (B, T).")
            B, T = input_ids.shape
            flat_ids = input_ids.reshape(-1)
            U, F, Y = _CeptaIndexFn.apply(
                flat_ids,
                self.W,
                self.f,
                self.SP,
                self._ste_mode,
                self.cfg.ste_band_tau,
                self.cfg.ste_sigmoid_gamma,
                self.cfg.update_mode_W,
                self.cfg.update_mode_f,
            )
            U = U.view(B, T, -1)
            F = F.view(B, T, -1)
            Y = Y.view(B, T, self.cfg.P, self.cfg.alpha)
            return U, F, Y

        if X_dense is None:
            raise ValueError("X_dense must be provided for dense CEPTA.")
        if X_dense.dim() != 3:
            raise ValueError("X_dense must be (B, T, d).")
        B, T, d = X_dense.shape
        if d != self.cfg.d_or_vocab:
            raise ValueError(
                f"Expected feature dim {self.cfg.d_or_vocab}, got {d}."
            )
        flat = X_dense.reshape(-1, d)
        U, F, Y = _CeptaDenseFn.apply(
            flat,
            self.W,
            self.f,
            self.SP,
            self._ste_mode,
            self.cfg.ste_band_tau,
            self.cfg.ste_sigmoid_gamma,
            self.cfg.update_mode_W,
            self.cfg.update_mode_f,
        )
        U = U.view(B, T, -1)
        F = F.view(B, T, -1)
        Y = Y.view(B, T, self.cfg.P, self.cfg.alpha)
        return U, F, Y


class CeptaRouting(nn.Module):
    """Utility router over CEPTA fan-out outputs."""

    def __init__(self, reduce: Literal["sum", "mean"] = "sum"):
        super().__init__()
        self.reduce = reduce

    def forward(self, Y: Tensor) -> Tensor:
        """Reduce fan-out dimension.

        Args:
            Y: (B, T, P, alpha)
        Returns:
            Aggregated path signal (B, T, P)
        """
        if Y.dim() != 4:
            raise ValueError("Y must be (B, T, P, alpha).")
        if self.reduce == "sum":
            return Y.sum(dim=-1)
        if self.reduce == "mean":
            return Y.mean(dim=-1)
        raise ValueError(f"Unknown reduce mode {self.reduce}")


def update_cepta_local(
    W: nn.Parameter,
    f: nn.Parameter,
    SP: nn.Parameter,
    Z: Tensor,
    F: Tensor,
    Y: Tensor,
    dW_bp: Optional[Tensor],
    df_bp: Optional[Tensor],
    hyperparams: Optional[Dict[str, float]] = None,
    mode: Literal["dense", "index"] = "dense",
    X_dense: Optional[Tensor] = None,
    input_ids: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Local learning rule combining BP grads, Oja updates, and SP homeostasis.

    Args:
        W, f, SP: Parameters to update in-place.
        Z: Potentials snapshot (B*, P), detached.
        F: Gate snapshot (B*, P), detached.
        Y: Output snapshot (B*, P, alpha), detached.
        dW_bp, df_bp: Optional backprop gradients to mix in.
        hyperparams: Dict of scalars controlling learning rates, clipping, refs.
        mode: "dense" or "index".
        X_dense: Input snapshot for dense mode (B*, d) if Oja_w is desired.
        input_ids: Tokens for index mode Oja updates (B*,).

    Returns:
        Tuple of applied delta tensors (dW_applied, df_applied, dSP_applied).
    """
    hp = {
        "eta_bp_w": 1.0,
        "eta_bp_f": 1.0,
        "eta_w": 0.0,
        "eta_f": 0.0,
        "eta_sp": 0.0,
        "beta_r": 0.01,
        "beta_m": 0.01,
        "r_star": 0.1,
        "m_star": 0.1,
        "lambda_r": 1.0,
        "lambda_m": 1.0,
        "sp_min": -1e3,
        "sp_max": 1e3,
        "w_min": -1e3,
        "w_max": 1e3,
        "f_min": -1e3,
        "f_max": 1e3,
        "z_ref": 1.0,
        "y_ref": 1.0,
        "eps": 1e-6,
        "update_mode_W": "all",
        "update_mode_f": "all",
    }
    if hyperparams:
        hp.update(hyperparams)

    with torch.no_grad():
        Z_fp = Z.float()
        F_fp = F.float()
        Y_fp = Y.float()
        W_fp = W.data.float()
        f_fp = f.data.float()
        SP_fp = SP.data.float()

        M_W, M_f = _compute_masks(
            F_fp, hp["update_mode_W"], hp["update_mode_f"]
        )

        B_star, P = Z_fp.shape
        Z_ref2 = hp["z_ref"] ** 2 + hp["eps"]
        Y_ref2 = hp["y_ref"] ** 2 + hp["eps"]

        # Backprop deltas
        dW_total = torch.zeros_like(W_fp)
        df_total = torch.zeros_like(f_fp)
        dSP_total = torch.zeros_like(SP_fp)

        if dW_bp is not None:
            dW_total += -hp["eta_bp_w"] * dW_bp.float()
        if df_bp is not None:
            df_total += -hp["eta_bp_f"] * df_bp.float()

        # Oja updates for W
        if hp["eta_w"] != 0.0 and mode == "dense" and X_dense is not None:
            if X_dense.dim() != 2:
                raise ValueError("X_dense for local update must be (B*, d).")
            X_fp = X_dense.float()
            term_main = (F_fp * Z_fp).unsqueeze(-1) * X_fp.unsqueeze(1)
            decay = ((F_fp * (Z_fp ** 2 / Z_ref2)).unsqueeze(-1)) * W_fp.unsqueeze(
                0
            )
            oja_W = hp["eta_w"] * (term_main - decay).sum(dim=0)
            dW_total += oja_W
        elif hp["eta_w"] != 0.0 and mode == "index":
            if input_ids is None:
                raise ValueError("input_ids are required for index-mode Oja update.")
            tokens = input_ids.view(-1)
            if tokens.numel() != B_star:
                raise ValueError("input_ids must match the flattened batch for Z/F.")
            tok_exp = tokens.unsqueeze(0).expand(P, -1)  # (P, B*)
            coeff_main = (F_fp * Z_fp).t()  # (P, B*)
            coeff_decay = (F_fp * (Z_fp ** 2 / Z_ref2)).t()  # (P, B*)
            gathered_W = torch.gather(W_fp, 1, tok_exp)
            oja_add = torch.zeros_like(W_fp)
            oja_add.scatter_add_(
                1, tok_exp, hp["eta_w"] * (coeff_main - coeff_decay * gathered_W)
            )
            dW_total += oja_add

        # Oja updates for f
        if hp["eta_f"] != 0.0:
            oja_f = hp["eta_f"] * (
                F_fp.unsqueeze(-1)
                * (
                    Y_fp * Z_fp.unsqueeze(-1)
                    - (Y_fp ** 2 / Y_ref2) * f_fp.unsqueeze(0)
                )
            ).sum(dim=0)
            df_total += oja_f

        # Apply masks and clipping
        dW_applied = (dW_total.t() * M_W).t()
        df_applied = (df_total.t() * M_f).t()
        W.data.copy_(
            torch.clamp(
                W_fp + dW_applied, min=hp["w_min"], max=hp["w_max"]
            ).to(W.dtype)
        )
        f.data.copy_(
            torch.clamp(
                f_fp + df_applied, min=hp["f_min"], max=hp["f_max"]
            ).to(f.dtype)
        )

        # SP homeostasis
        if hp["eta_sp"] != 0.0:
            r_t = F_fp
            m_t = F_fp * torch.clamp(
                (Z_fp - SP_fp) / (hp["z_ref"] + hp["eps"]), min=0.0
            )
            state = hp.get("state", {})
            r_bar = state.get("r_bar") or torch.zeros_like(SP_fp)
            m_bar = state.get("m_bar") or torch.zeros_like(SP_fp)
            beta_r = hp["beta_r"]
            beta_m = hp["beta_m"]
            r_bar = (1 - beta_r) * r_bar + beta_r * r_t.mean(dim=0)
            m_bar = (1 - beta_m) * m_bar + beta_m * m_t.mean(dim=0)
            if "state" in hp:
                hp["state"]["r_bar"] = r_bar
                hp["state"]["m_bar"] = m_bar
            delta_sp = hp["eta_sp"] * (
                hp["lambda_r"] * (r_bar - hp["r_star"])
                + hp["lambda_m"] * (m_bar - hp["m_star"])
            )
            dSP_total = delta_sp
            SP.data.copy_(
                torch.clamp(
                    SP_fp + delta_sp, min=hp["sp_min"], max=hp["sp_max"]
                ).to(SP.dtype)
            )

        return dW_applied, df_applied, dSP_total
