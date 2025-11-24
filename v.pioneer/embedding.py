from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn

from perceptron_cepta import CeptaConfig, CeptaEmbedding
from cepta_ssm import CeptaSSMLiteLowRank
from module_layer import CeptaModelConfig, CeptaPathTransformerLM


class CeptaSentenceEncoder(nn.Module):
    """
    - 내부적으로 CeptaPathTransformerLM을 사용
    - 출력: 시퀀스당 하나의 임베딩 벡터 (예: 마지막 토큰의 경로 상태 평균)
    """
    def __init__(self, model_cfg: CeptaModelConfig, pooling: str = "mean_last"):
        super().__init__()
        self.model = CeptaPathTransformerLM(model_cfg)
        self.pooling = pooling

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B,T)

        Returns:
            embeddings: (B,P)  (경로공간 임베딩)
        """
        logits, cache_states = self.model(input_ids)
        # logits 이전 단계의 t를 직접 사용할 수도 있지만,
        # 여기서는 lm_head 이전의 h를 재사용하기 어려우므로,
        # LM 출력 대신 마지막 블록의 t를 얻기 위해 약간의 우회가 필요하다.
        # 간단하게는 모델 구조를 약간 수정해 t를 반환하도록 할 수 있음.
        # 여기서는 예시로, lm_head.weight를 transposed projection으로 사용해
        # vocab space 대신 path space로 역투영하는 방식으로 구현(approx)할 수도 있지만,
        # 구조 명확성을 위해, CeptaPathTransformerLM에 작은 수정이 들어가는 편이 이상적이다.
        #
        # 간단한 구현 예시: logits에서 softmax를 취해 확률로 만든 뒤,
        # lm_head.weight^T 와 곱해 approximate path representation을 얻는 방식.
        # (정확한 역함수는 아니지만 예시 수준에서는 충분)
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)  # (B,T,V)
            W = self.model.lm_head.weight          # (V,P)
            t_approx = torch.matmul(probs, W)      # (B,T,P) 평균적인 경로 표현 근사

        if self.pooling == "mean_last":
            # 마스크 없다고 가정하고 마지막 토큰만 가져와 평균
            emb = t_approx[:, -1, :]               # (B,P)
        elif self.pooling == "mean_all":
            emb = t_approx.mean(dim=1)             # (B,P)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return emb
