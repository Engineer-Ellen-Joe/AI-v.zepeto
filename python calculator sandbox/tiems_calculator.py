from __future__ import annotations
import torch
import re
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler

from cepta_network_lowrank import (
    build_tokenizer,
    DeepSeekCeptaTransformer,
    ModelCfg
)


# -*- coding: utf-8 -*-
"""
cepta_multiply_module.py

Cepta Network LowRank 기반으로 정수 곱셈(num9 * num10)을 학습·추론하는 모듈.
- perceptron_cepta.py, cepta_network_lowrank.py가 같은 폴더에 있어야 합니다.
- AutoTokenizer 사용.
- 학습: 합성 데이터 "a*b=c"를 토크나이즈하여 next-token 예측으로 학습.
- 추론: 프롬프트 "a*b="에 대해 그리디 생성 → 결과 정수 파싱.

사용 예:
    python cepta_multiply_module.py --tokenizer gpt2 --steps 2000 --max_n 99 \
        --prompt 12 34
"""
# ------------------------------
# 하이퍼파라미터
# ------------------------------
@dataclass
class TrainArgs:
    tokenizer: str = 'gpt2'
    max_n: int = 99           # 0..max_n 범위에서 곱셈 학습
    dataset_size: int = 20000 # 학습 샘플 수
    batch_size: int = 64
    steps: int = 2000
    lr: float = 3e-4
    wd: float = 0.0
    clip_norm: float = 1.0
    use_bf16: bool = True
    use_fp16: bool = False

    # 모델 크기(경량)
    d_model: int = 256
    n_layers: int = 4
    P: int = 128
    alpha: int = 4
    route_mode: str = 'topk'
    topk: int = 8
    dropout: float = 0.0
    dtype_store: str = 'bf16'
    ssm: str = 'lowrank'
    P_r: int = 64
    ssm_residual: bool = True

# ------------------------------
# 데이터 생성
# ------------------------------
def make_samples(max_n: int, count: int, *, seed: int = 0) -> List[str]:
    rnd = random.Random(seed)
    s = []
    for _ in range(count):
        a = rnd.randint(0, max_n)
        b = rnd.randint(0, max_n)
        c = a * b
        s.append(f"{a}*{b}={c}\n")
    return s


def collate_batch(texts: List[str], tok, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    # teacher-forcing: 입력은 문장의 마지막 토큰을 제외, 타겟은 첫 토큰 제외
    enc = tok(texts, return_tensors='pt', padding=True, truncation=True)
    ids = enc['input_ids']  # (B,T)
    attn = enc['attention_mask']
    # 보장: pad_token 존재. pad에 대한 타겟은 -100으로 마스킹.
    inp = ids[:, :-1]
    tgt = ids[:, 1:]
    pad_id = tok.pad_token_id
    mask = (inp != pad_id) & (tgt != pad_id)
    tgt = tgt.masked_fill(~mask, -100)
    return inp.to(device), tgt.to(device)

# ------------------------------
# 학습 루프
# ------------------------------

def build_model(tok, args: TrainArgs, device: str) -> DeepSeekCeptaTransformer:
    cfg = ModelCfg(
        vocab_size=len(tok), d_model=args.d_model, n_layers=args.n_layers,
        P=args.P, alpha=args.alpha, route_mode=args.route_mode, topk=args.topk,
        dropout=args.dropout, gate='hard', dtype_store=args.dtype_store,
        ssm=args.ssm, P_r=args.P_r, ssm_residual=args.ssm_residual,
    )
    model = DeepSeekCeptaTransformer(cfg).to(device)
    return model


def train(model: DeepSeekCeptaTransformer, tok, args: TrainArgs, device: str) -> None:
    texts = make_samples(args.max_n, args.dataset_size, seed=0)
    loader = DataLoader(texts, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        collate_fn=lambda batch: collate_batch(batch, tok, device))

    amp_dtype = torch.bfloat16 if args.use_bf16 else (torch.float16 if args.use_fp16 else None)
    scaler = GradScaler('cuda', enabled=args.use_fp16 and not args.use_bf16)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    def step(inp, tgt):
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=(amp_dtype is not None)):
            logits = model(inp)  # (B,T,V)
            # logits와 tgt 길이가 일치해야 함
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B*T, V), tgt.reshape(B*T), ignore_index=-100)
        return loss

    model.train()
    it = iter(loader)
    for s in range(1, args.steps + 1):
        try:
            inp, tgt = next(it)
        except StopIteration:
            it = iter(loader)
            inp, tgt = next(it)

        optim.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            loss = step(inp, tgt)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            scaler.step(optim)
            scaler.update()
        else:
            loss = step(inp, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optim.step()

        if s % max(1, args.steps // 10) == 0:
            print(f"step {s}/{args.steps}  loss={float(loss):.4f}")

# ------------------------------
# 생성 및 곱셈 추론
# ------------------------------

def generate(model: DeepSeekCeptaTransformer, tok, prompt: str, max_new_tokens: int = 16, device: str = 'cpu') -> str:
    model.eval()
    with torch.no_grad():
        enc = tok([prompt], return_tensors='pt')
        ids = enc['input_ids'].to(device)
        for _ in range(max_new_tokens):
            logits = model(ids)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
            if next_id.item() == tok.eos_token_id:
                break
        text = tok.batch_decode(ids, skip_special_tokens=True)[0]
        return text


def cepta_multiply(model: DeepSeekCeptaTransformer, tok, a: int, b: int, device: str = 'cpu') -> Tuple[int, str]:
    prompt = f"{a}*{b}="
    out = generate(model, tok, prompt, max_new_tokens=16, device=device)
    # 결과 파싱: 마지막에 등장하는 "=NNN" 패턴
    m = re.search(r"=\s*([+-]?\d+)", out)
    if not m:
        raise RuntimeError(f"생성 실패: '{out}'")
    pred = int(m.group(1))
    return pred, out

# ------------------------------
# CLI
# ------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', type=str, default='gpt2')
    parser.add_argument('--max_n', type=int, default=99)
    parser.add_argument('--dataset_size', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--prompt', type=int, nargs=2, help='num9 num10')
    args_ns = parser.parse_args()

    targs = TrainArgs(
        tokenizer=args_ns.tokenizer,
        max_n=args_ns.max_n,
        dataset_size=args_ns.dataset_size,
        batch_size=args_ns.batch_size,
        steps=args_ns.steps,
        lr=args_ns.lr,
        use_fp16=bool(args_ns.fp16),
        use_bf16=bool(args_ns.bf16),
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = build_tokenizer(targs.tokenizer)
    model = build_model(tok, targs, device)

    # 학습
    train(model, tok, targs, device)

    # 추론
    if args_ns.prompt is not None:
        a, b = args_ns.prompt
        pred, raw = cepta_multiply(model, tok, a, b, device)
        print(raw)
        print(f"pred: {pred}  (gt: {a*b})")
    else:
        for a, b in [(3, 7), (12, 34), (25, 25)]:
            pred, raw = cepta_multiply(model, tok, a, b, device)
            print(raw)
            print(f"pred: {pred}  (gt: {a*b})")
