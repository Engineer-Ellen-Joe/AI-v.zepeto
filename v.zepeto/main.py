"""
Entry point for CEPTA-based Transformer demo.
"""
from __future__ import annotations

import torch

from embedding import CeptaEmbeddingConfig
from module_layer import CeptaTransformerLM
from tokenizer import decode_tokens, encode_texts, get_deepseek_v3_tokenizer


def build_model():
    tokenizer = get_deepseek_v3_tokenizer()
    vocab_size = tokenizer.vocab_size
    P = 512
    alpha = 4
    P_r = 64
    d_model = 1024
    n_layers = 6
    max_seq_len = 512

    emb_cfg = CeptaEmbeddingConfig(
        vocab_size=vocab_size,
        P=P,
        alpha=alpha,
        P_r=P_r,
        d_model=d_model,
        max_seq_len=max_seq_len,
        dtype_store="bf16",
    )
    model = CeptaTransformerLM(emb_cfg, n_layers=n_layers)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer, device, max_seq_len


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int, device: str, max_length: int):
    input_ids, _ = encode_texts(tokenizer, [prompt], max_length=max_length)
    input_ids = input_ids.to(device)
    for _ in range(max_new_tokens):
        logits = model(input_ids)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        if input_ids.size(1) >= max_length:
            break
    decoded = decode_tokens(tokenizer, input_ids.cpu())[0]
    return decoded


if __name__ == "__main__":
    model, tokenizer, device, max_len = build_model()
    while True:
        try:
            prompt = input(">>> ").strip()
        except EOFError:
            break
        if not prompt:
            continue
        output = generate(model, tokenizer, prompt, max_new_tokens=50, device=device, max_length=max_len)
        print(output)

