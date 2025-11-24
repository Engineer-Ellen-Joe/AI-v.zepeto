from __future__ import annotations

import torch

from tokenizer import get_tokenizer, encode, decode
from module_layer import CeptaModelConfig, CeptaPathTransformerLM
from embedding import CeptaSentenceEncoder


def build_models(vocab_size: int):
    cfg = CeptaModelConfig(
        vocab_size=vocab_size,
        P=256,          # 경로 수 (예시)
        alpha=4,        # 출력 기저 수 (예시)
        num_layers=4,
        P_r=64,
        use_fp16=False,
        use_bf16=True,
    )
    lm = CeptaPathTransformerLM(cfg)
    encoder = CeptaSentenceEncoder(cfg, pooling="mean_last")
    return lm, encoder


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size

    lm, encoder = build_models(vocab_size)
    lm.to(device)
    encoder.to(device)

    text = "Hello, who are you?"
    batch = encode([text], tokenizer=tokenizer, max_length=64)
    input_ids = batch["input_ids"].to(device)

    # LM forward
    with torch.no_grad():
        logits, _ = lm(input_ids)
        # 마지막 토큰에서 top-5 예측
        last_logits = logits[0, -1, :]
        topk = torch.topk(last_logits, k=5)
        print("Top-5 token ids:", topk.indices.tolist())
        print("Top-5 probs:", torch.softmax(topk.values, dim=-1).tolist())
        print("Top-5 tokens:", [tokenizer.decode([i]) for i in topk.indices.tolist()])

    # Sentence embedding
    with torch.no_grad():
        emb = encoder(input_ids)  # (B,P)
        print("Sentence embedding shape:", emb.shape)


if __name__ == "__main__":
    main()
