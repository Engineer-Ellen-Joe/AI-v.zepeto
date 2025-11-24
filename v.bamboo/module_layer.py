from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from perceptron_cepta import CeptaPerceptron
from cepta_ssm import (
    LowRankCrossPathSSM,
    linear_backward,
    linear_forward,
    rmsnorm_backward,
    rmsnorm_forward,
)


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def gelu_backward(x: np.ndarray, grad: np.ndarray) -> np.ndarray:
    k0 = np.sqrt(2.0 / np.pi)
    k1 = 0.044715
    x3 = np.power(x, 3)
    tanh_out = np.tanh(k0 * (x + k1 * x3))
    sech2 = 1.0 - np.square(tanh_out)
    grad_gelu = 0.5 * (1.0 + tanh_out) + 0.5 * x * sech2 * k0 * (1 + 3 * k1 * x * x)
    return grad * grad_gelu


@dataclass
class BlockConfig:
    d_model: int
    P: int
    alpha: int
    P_r: int
    lr: float = 1e-3
    eps_rms: float = 1e-6


class RMSNorm:
    def __init__(self, d: int, eps: float = 1e-6, lr: float = 1e-3):
        self.d = d
        self.eps = eps
        self.weight = np.ones((d,), dtype=np.float32)
        self.lr = lr
        self.cache: Dict[str, np.ndarray] = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        Y, rms = rmsnorm_forward(X, self.weight, eps=self.eps)
        self.cache = {"X": X, "rms": rms}
        return Y

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        X = self.cache["X"]
        rms = self.cache["rms"]
        grad_X, grad_w = rmsnorm_backward(X, self.weight, grad_out, rms, eps=self.eps)
        self.weight -= self.lr * grad_w
        return grad_X


class Linear:
    def __init__(self, in_dim: int, out_dim: int, lr: float = 1e-3, bias: bool = True, scale: float = 0.02):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lr = lr
        self.weight = np.random.randn(out_dim, in_dim).astype(np.float32) * scale
        self.bias = np.zeros((out_dim,), dtype=np.float32) if bias else None
        self.cache: Dict[str, np.ndarray] = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        Y = linear_forward(X, self.weight, self.bias)
        self.cache = {"X": X}
        return Y

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        X = self.cache["X"]
        grad_X, grad_W, grad_b = linear_backward(X, self.weight, grad_out)
        self.weight -= self.lr * grad_W
        if self.bias is not None:
            self.bias -= self.lr * grad_b
        return grad_X


class MLP:
    def __init__(self, d_model: int, hidden_mult: float = 4.0, lr: float = 1e-3):
        hidden = int(d_model * hidden_mult)
        self.fc1 = Linear(d_model, hidden, lr=lr)
        self.fc2 = Linear(hidden, d_model, lr=lr)
        self.cache: Dict[str, np.ndarray] = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        h1 = self.fc1.forward(X)
        h_act = gelu(h1)
        out = self.fc2.forward(h_act)
        self.cache = {"h1": h1, "h_act": h_act}
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        h1 = self.cache["h1"]
        h_act = self.cache["h_act"]
        grad_h_act = self.fc2.backward(grad_out)
        grad_h1 = gelu_backward(h1, grad_h_act)
        grad_X = self.fc1.backward(grad_h1)
        return grad_X


class CeptaTransformerBlock:
    def __init__(self, cfg: BlockConfig):
        self.cfg = cfg
        self.norm1 = RMSNorm(cfg.d_model, eps=cfg.eps_rms, lr=cfg.lr)
        self.norm2 = RMSNorm(cfg.d_model, eps=cfg.eps_rms, lr=cfg.lr)
        self.to_P = Linear(cfg.d_model, cfg.d_model, lr=cfg.lr, bias=False)
        self.from_P = Linear(cfg.P, cfg.d_model, lr=cfg.lr, bias=True)
        self.perceptron = CeptaPerceptron(P=cfg.P, d=cfg.d_model, alpha=cfg.alpha, vocab=cfg.d_model)
        self.ssm = LowRankCrossPathSSM(P=cfg.P, Pr=cfg.P_r)
        self.mlp = MLP(cfg.d_model, hidden_mult=4.0, lr=cfg.lr)
        self.cache: Dict[str, np.ndarray] = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        B, T, D = X.shape
        X_flat = X.reshape(B * T, D)
        h1 = self.norm1.forward(X_flat)
        U = self.to_P.forward(h1)
        Z, F, Y, t = self.perceptron.forward_dense(U)
        t_seq = t.reshape(B, T, self.cfg.P)
        F_seq = F.reshape(B, T, self.cfg.P)
        ttilde_all = []
        for b in range(B):
            ttilde_b, _ = self.ssm.forward(t_seq[b], F_seq[b])
            ttilde_all.append(ttilde_b)
        ttilde = np.stack(ttilde_all, axis=0)  # (B,T,P)
        deltaX1 = self.from_P.forward(ttilde.reshape(B * T, self.cfg.P)).reshape(B, T, D)
        X1 = X + deltaX1
        h2 = self.norm2.forward(X1.reshape(B * T, D))
        deltaX2 = self.mlp.forward(h2).reshape(B, T, D)
        out = X1 + deltaX2
        self.cache = {
            "X": X,
            "h1": h1,
            "U": U,
            "F": F_seq,
            "t_seq": t_seq,
            "ttilde": ttilde,
            "deltaX1": deltaX1,
            "h2": h2,
        }
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        X = self.cache["X"]
        h2 = self.cache["h2"]
        t_seq = self.cache["t_seq"]
        F_seq = self.cache["F"]
        ttilde = self.cache["ttilde"]
        deltaX1 = self.cache["deltaX1"]
        B, T, D = grad_out.shape

        grad_X1 = grad_out.copy()
        grad_deltaX2 = grad_out.copy()

        grad_h2 = self.mlp.backward(grad_deltaX2.reshape(B * T, D))
        grad_X1 += self.norm2.backward(grad_h2).reshape(B, T, D)

        grad_ttilde_flat = self.from_P.backward(grad_X1.reshape(B * T, D))

        grad_t_seq = np.zeros_like(t_seq)
        for b in range(B):
            _, grad_t_b = self.ssm.backward(grad_ttilde_flat.reshape(B, T, self.cfg.P)[b])
            grad_t_seq[b] = grad_t_b
            self.ssm.update()

        grad_t_flat = grad_t_seq.reshape(B * T, self.cfg.P)
        grad_U = self.perceptron.backward_dense(delta_Y=None, delta_t=grad_t_flat)
        self.perceptron.update()

        grad_h1 = self.to_P.backward(grad_U)
        grad_X_flat = self.norm1.backward(grad_h1)
        return grad_X_flat.reshape(B, T, D)


class CeptaModel:
    def __init__(self, cfg: BlockConfig):
        self.block = CeptaTransformerBlock(cfg)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.block.forward(X)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return self.block.backward(grad_out)
