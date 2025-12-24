from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from embedding import CeptaEmbeddingConfig
from module_layer import CeptaTransformerLM
from tokenizer import get_deepseek_v3_tokenizer
from perceptron_cepta import update_cepta_local, CeptaEmbedding
from dataset_after_aling import ChunkedLMDataset, read_files, tokenize_texts


def build_custom_order() -> List[str]:
    order: List[str] = []
    lo = 11
    hi = 20
    while hi >= lo:
        order.append(f"{hi}.txt")
        if hi == lo:
            break
        order.append(f"{lo}.txt")
        hi -= 1
        lo += 1
    return order


def read_corpus(data_dir: Path, use_custom_order: bool = False) -> List[str]:
    """Read text files, optionally in custom interleaved order."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    texts: List[str] = []
    if use_custom_order:
        order = build_custom_order()
        for name in order:
            path = data_dir / name
            if not path.exists():
                raise FileNotFoundError(f"Expected file missing for custom order: {path}")
            texts.append(path.read_text(encoding="utf-8"))
    else:
        for path in sorted(data_dir.glob("*.txt")):
            texts.append(path.read_text(encoding="utf-8"))
    if not texts:
        raise ValueError(f"No .txt files found in {data_dir}")
    return texts


def tokenize_corpus(tokenizer, texts: List[str]) -> torch.Tensor:
    """Tokenize and concatenate a list of texts into a single 1-D tensor of token IDs."""
    token_ids: List[torch.Tensor] = []
    for txt in texts:
        enc = tokenizer(
            txt,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=False,
            max_length=None,
        )
        token_ids.append(enc["input_ids"].squeeze(0))
    return torch.cat(token_ids, dim=0)


def tokenize_text(tokenizer, text: str) -> torch.Tensor:
    """Tokenize a single text into 1-D tensor of token IDs."""
    enc = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=False,
        max_length=None,
    )
    return enc["input_ids"].squeeze(0)


class LMDataset(Dataset):
    """Contiguous block language modeling dataset."""

    def __init__(self, tokens: torch.Tensor, block_size: int):
        if tokens.dim() != 1:
            raise ValueError("tokens must be a 1-D tensor.")
        if block_size <= 1:
            raise ValueError("block_size must be > 1.")
        self.tokens = tokens
        self.block_size = block_size
        self.n_blocks = (len(tokens) - 1) // block_size
        if self.n_blocks <= 0:
            raise ValueError("Not enough tokens to form a single block.")

    def __len__(self) -> int:
        return self.n_blocks

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.block_size
        end = start + self.block_size
        x = self.tokens[start:end]
        y = self.tokens[start + 1 : end + 1]
        return x, y


def split_tokens(tokens: torch.Tensor, train_ratio: float = 0.9) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split token stream into train and test portions."""
    n_train = int(len(tokens) * train_ratio)
    if n_train <= 0 or n_train >= len(tokens):
        raise ValueError("Invalid train split; adjust train_ratio.")
    return tokens[:n_train], tokens[n_train:]


def build_model(vocab_size: int, max_seq_len: int, dtype_store: str = "bf16") -> CeptaTransformerLM:
    emb_cfg = CeptaEmbeddingConfig(
        vocab_size=vocab_size,
        P=2048,
        alpha=4,
        P_r=256,
        d_model=1024,
        max_seq_len=max_seq_len,
        dtype_store=dtype_store,
    )
    model = CeptaTransformerLM(emb_cfg, n_layers=6)
    return model


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    local_hp: dict,
    log_every: int = 0,
) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    step = 0
    for input_ids, labels in dataloader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, packs = model(input_ids, return_packs=True)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100
        )
        loss.backward()

        apply_local_updates(model, packs, input_ids, local_hp=local_hp)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * labels.numel()
        total_tokens += labels.numel()
        step += 1
        if log_every > 0 and step % log_every == 0:
            print(f"  [step {step}] train loss {loss.item():.4f}")
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    step = 0
    for input_ids, labels in dataloader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100
        )
        total_loss += loss.item() * labels.numel()
        total_tokens += labels.numel()
        step += 1
        # Per-chunk validation logging
        print(f"  [val step {step}] val loss {loss.item():.4f}")
    return total_loss / max(total_tokens, 1)


def apply_local_updates(model: CeptaTransformerLM, packs, input_ids: torch.Tensor, local_hp: dict):
    """Apply BP+Oja/SP local updates for all CEPTA modules, then clear their grads."""
    # Embedding CEPTA (index path)
    emb_pack = packs["embedding"]
    cepta_emb: CeptaEmbedding = model.embedding.cepta
    U = emb_pack["U"].detach()
    F = emb_pack["F"].detach()
    Y = emb_pack["Y"].detach()
    flat_ids = emb_pack["input_ids"].reshape(-1)
    dW_bp = cepta_emb.W.grad
    df_bp = cepta_emb.f.grad
    update_cepta_local(
        cepta_emb.W,
        cepta_emb.f,
        cepta_emb.SP,
        Z=U.reshape(-1, U.size(-1)),
        F=F.reshape(-1, F.size(-1)),
        Y=Y.reshape(-1, Y.size(-2), Y.size(-1)),
        dW_bp=dW_bp,
        df_bp=df_bp,
        hyperparams=local_hp,
        mode="index",
        input_ids=flat_ids,
    )
    # Zero grads to avoid double-stepping via optimizer
    cepta_emb.W.grad = None
    cepta_emb.f.grad = None
    cepta_emb.SP.grad = None

    # Per-block CEPTA (dense path)
    for blk_pack, block in zip(packs["blocks"], model.blocks):
        ctx_pack = blk_pack["context"]
        cepta_dense: CeptaEmbedding = block.context.cepta_dense
        U_b = ctx_pack["U"].detach()
        F_b = ctx_pack["F"].detach()
        Y_b = ctx_pack["Y"].detach()
        X_dense = ctx_pack["X_dense"].detach().reshape(-1, ctx_pack["X_dense"].size(-1))
        update_cepta_local(
            cepta_dense.W,
            cepta_dense.f,
            cepta_dense.SP,
            Z=U_b.reshape(-1, U_b.size(-1)),
            F=F_b.reshape(-1, F_b.size(-1)),
            Y=Y_b.reshape(-1, Y_b.size(-2), Y_b.size(-1)),
            dW_bp=cepta_dense.W.grad,
            df_bp=cepta_dense.f.grad,
            hyperparams=local_hp,
            mode="dense",
            X_dense=X_dense,
        )
        cepta_dense.W.grad = None
        cepta_dense.f.grad = None
        cepta_dense.SP.grad = None


def main():
    parser = argparse.ArgumentParser(description="Train/Test CEPTA Transformer on After_aling dataset.")
    parser.add_argument("--data_dir", type=Path, default=Path("Z:/Final_project/data_set/After_aling"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dtype_store", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--save_path", type=Path, default=Path("cepta_lm.pt"))
    parser.add_argument("--eta_bp_w", type=float, default=1.0)
    parser.add_argument("--eta_bp_f", type=float, default=1.0)
    parser.add_argument("--eta_w", type=float, default=0.0)
    parser.add_argument("--eta_f", type=float, default=0.0)
    parser.add_argument("--eta_sp", type=float, default=0.0)
    parser.add_argument("--w_max", type=float, default=1e3)
    parser.add_argument("--f_max", type=float, default=1e3)
    parser.add_argument("--z_ref", type=float, default=1.0)
    parser.add_argument("--y_ref", type=float, default=1.0)
    parser.add_argument("--beta_r", type=float, default=0.01)
    parser.add_argument("--beta_m", type=float, default=0.01)
    parser.add_argument("--r_star", type=float, default=0.1)
    parser.add_argument("--m_star", type=float, default=0.1)
    parser.add_argument("--lambda_r", type=float, default=1.0)
    parser.add_argument("--lambda_m", type=float, default=1.0)
    parser.add_argument("--custom_order", action="store_true", help="Use 20,1,19,2,... file order.")
    parser.add_argument("--per_file_report", action="store_true", help="Train file by file and report loss per file.")
    parser.add_argument("--log_every", type=int, default=0, help="Log train loss every N steps (0=off).")
    args = parser.parse_args()

    tokenizer = get_deepseek_v3_tokenizer()
    # Avoid tokenizer max_length warnings; we will chunk into blocks ourselves.
    tokenizer.model_max_length = int(1e9)
    model = build_model(tokenizer.vocab_size, max_seq_len=args.block_size, dtype_store=args.dtype_store)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    local_hp = {
        "eta_bp_w": args.eta_bp_w,
        "eta_bp_f": args.eta_bp_f,
        "eta_w": args.eta_w,
        "eta_f": args.eta_f,
        "eta_sp": args.eta_sp,
        "w_max": args.w_max,
        "f_max": args.f_max,
        "z_ref": args.z_ref,
        "y_ref": args.y_ref,
        "beta_r": args.beta_r,
        "beta_m": args.beta_m,
        "r_star": args.r_star,
        "m_star": args.m_star,
        "lambda_r": args.lambda_r,
        "lambda_m": args.lambda_m,
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Use After_aling 01..07 in given directory, chunked
    file_names = [f"{i:02d}.txt" for i in range(1, 8)]
    texts = read_files(Path(args.data_dir), file_names)
    tokens_all = tokenize_texts(tokenizer, texts)
    ds = ChunkedLMDataset(tokens_all, block_size=args.block_size)
    n = len(ds)
    n_train = max(1, int(n * args.train_ratio))
    train_ds = torch.utils.data.Subset(ds, range(0, n_train))
    val_ds = torch.utils.data.Subset(ds, range(n_train, n))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, local_hp=local_hp, log_every=args.log_every
        )
        val_loss = evaluate(model, val_loader, device) if len(val_ds) > 0 else float("nan")
        print(f"Epoch {epoch}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

    torch.save(model.state_dict(), args.save_path)
    print(f"Saved trained model to {args.save_path}")


if __name__ == "__main__":
    main()
