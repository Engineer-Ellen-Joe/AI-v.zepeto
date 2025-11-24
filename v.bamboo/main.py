import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

from embedding import CeptaEmbeddingModel, EmbeddingConfig
from tokenizer import tokenizer


def load_config(path: Path) -> EmbeddingConfig:
    data = json.loads(path.read_text())
    return EmbeddingConfig(
        vocab_size=data["vocab_size"],
        d_model=data["d_model"],
        P=data["P"],
        alpha=data["alpha"],
        P_r=data["P_r"],
        lr=data.get("lr", 1e-3),
        eps_rms=data.get("eps_rms", 1e-6),
    )


def prepare_batch(texts: List[str]):
    token_ids = tokenizer.batch_encode(texts)
    max_len = max(len(t) for t in token_ids)
    batch = np.zeros((len(texts), max_len), dtype=np.int32)
    for i, seq in enumerate(token_ids):
        batch[i, : len(seq)] = np.array(seq, dtype=np.int32)
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="JSON config for model.")
    parser.add_argument("--mode", type=str, choices=["train", "infer"], default="infer")
    parser.add_argument("--text", type=str, default="hello cepta")
    args = parser.parse_args()

    if args.config:
        cfg = load_config(Path(args.config))
    else:
        cfg = EmbeddingConfig(vocab_size=32000, d_model=128, P=64, alpha=4, P_r=32, lr=1e-3)

    model = CeptaEmbeddingModel(cfg)
    batch = prepare_batch([args.text])

    if args.mode == "train":
        targets = np.roll(batch, -1, axis=1)
        loss = model.train_step(batch, targets)
        print(f"training step complete, loss={loss:.6f}")
    else:
        logits, hidden = model.forward(batch)
        print("logits shape:", logits.shape)
        print("hidden embedding sample:", hidden[0, -1, :5])


if __name__ == "__main__":
    main()
