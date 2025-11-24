from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from module_layer import BlockConfig, CeptaModel
from cepta_ssm import linear_backward, linear_forward


@dataclass
class EmbeddingConfig:
    vocab_size: int
    d_model: int
    P: int
    alpha: int
    P_r: int
    lr: float = 1e-3
    eps_rms: float = 1e-6


class CeptaEmbeddingModel:
    """
    High-level embedding model that couples token embeddings with CEPTA transformer block and a vocabulary head.
    """

    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        block_cfg = BlockConfig(
            d_model=cfg.d_model,
            P=cfg.P,
            alpha=cfg.alpha,
            P_r=cfg.P_r,
            lr=cfg.lr,
            eps_rms=cfg.eps_rms,
        )
        self.model = CeptaModel(block_cfg)
        self.token_embedding = np.random.randn(cfg.vocab_size, cfg.d_model).astype(np.float32) * 0.02
        self.lm_head_weight = np.random.randn(cfg.vocab_size, cfg.d_model).astype(np.float32) * 0.02
        self.cache = {}

    def embed_tokens(self, token_ids: np.ndarray) -> np.ndarray:
        return self.token_embedding[token_ids]

    def forward(self, token_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = self.embed_tokens(token_ids)
        hidden = self.model.forward(X)
        B, T, D = hidden.shape
        logits = linear_forward(hidden.reshape(B * T, D), self.lm_head_weight, None).reshape(
            B, T, self.cfg.vocab_size
        )
        self.cache = {"token_ids": token_ids, "hidden": hidden}
        return logits, hidden

    def loss_and_grad(self, logits: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
        B, T, V = logits.shape
        logits_max = logits.max(axis=2, keepdims=True)
        exp = np.exp(logits - logits_max)
        probs = exp / exp.sum(axis=2, keepdims=True)
        loss = -np.mean(np.log(probs[np.arange(B)[:, None], np.arange(T), targets]))
        grad_logits = probs
        grad_logits[np.arange(B)[:, None], np.arange(T), targets] -= 1.0
        grad_logits /= (B * T)
        return loss, grad_logits

    def backward(self, grad_logits: np.ndarray):
        token_ids = self.cache["token_ids"]
        hidden = self.cache["hidden"]
        B, T, D = hidden.shape
        grad_hidden_flat, grad_head_W, _ = linear_backward(
            hidden.reshape(B * T, D), self.lm_head_weight, grad_logits.reshape(B * T, self.cfg.vocab_size)
        )
        self.lm_head_weight -= self.cfg.lr * grad_head_W

        grad_hidden = grad_hidden_flat.reshape(B, T, D)
        grad_input = self.model.backward(grad_hidden)
        grad_tok = np.zeros_like(self.token_embedding)
        for b in range(B):
            for t in range(T):
                tok = token_ids[b, t]
                grad_tok[tok] += grad_input[b, t]
        self.token_embedding -= self.cfg.lr * grad_tok

    def train_step(self, token_ids: np.ndarray, targets: np.ndarray) -> float:
        logits, _ = self.forward(token_ids)
        loss, grad_logits = self.loss_and_grad(logits, targets)
        self.backward(grad_logits)
        return loss
