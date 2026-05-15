"""
src/build_index.py
Extract CLS-token embeddings from the fine-tuned DistilBERT model
and build a FAISS flat-L2 index over the full dataset.

Usage:
    python -m src.build_index \
        --dataset  data/dataset.csv \
        --model    models/distilbert_finetuned \
        --out      data
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import faiss
from transformers import DistilBertTokenizerFast, DistilBertModel


# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE = 64     # tune down if CPU / low RAM
MAX_LENGTH = 256    # must match finetune.py


# ── Core ──────────────────────────────────────────────────────────────────────

def extract_embeddings(
    statements: list[str],
    model: DistilBertModel,
    tokenizer: DistilBertTokenizerFast,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """
    Encode statements in batches.
    Returns float32 array of shape (N, 768).
    Uses [CLS] token (index 0) as the sentence representation.
    """
    all_embeddings = []
    model.eval()

    for start in range(0, len(statements), batch_size):
        batch = statements[start : start + batch_size]
        inputs = tokenizer(
            batch,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # last_hidden_state: (batch, seq_len, 768) — take position 0 ([CLS])
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        all_embeddings.append(cls_embeddings)

        if (start // batch_size + 1) % 20 == 0:
            print(f"  encoded {start + len(batch):,} / {len(statements):,}")

    return np.vstack(all_embeddings).astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a flat L2 index.
    Flat = exact nearest-neighbour (no approximation).
    Switch to IndexIVFFlat for >1M rows.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    # Wrap with IDMap so stored IDs match DataFrame row indices
    index_with_ids = faiss.IndexIDMap(index)
    ids = np.arange(len(embeddings), dtype=np.int64)
    index_with_ids.add_with_ids(embeddings, ids)
    return index_with_ids


# ── Public API ────────────────────────────────────────────────────────────────

def build_index(
    dataset_csv: str,
    model_dir: str = "models/distilbert_finetuned",
    output_dir: str = "data",
):
    """
    Full pipeline: load dataset → embed → build FAISS index → save.
    Returns (embeddings np.ndarray, faiss.Index).
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(dataset_csv)
    statements = df["statement"].astype(str).tolist()
    print(f"[build_index] {len(statements):,} statements to embed")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[build_index] Device: {device}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    # Load as base model (no classification head) — we only want embeddings
    model = DistilBertModel.from_pretrained(model_dir).to(device)

    embeddings = extract_embeddings(statements, model, tokenizer, device)
    print(f"[build_index] Embeddings shape: {embeddings.shape}")

    index = build_faiss_index(embeddings)
    print(f"[build_index] FAISS index size: {index.ntotal:,}")

    # ── Save ──────────────────────────────────────────────────────────────────
    emb_path   = os.path.join(output_dir, "embeddings.npy")
    idx_path   = os.path.join(output_dir, "faiss_index.bin")

    np.save(emb_path, embeddings)
    faiss.write_index(index, idx_path)

    print(f"[build_index] Saved {emb_path}")
    print(f"[build_index] Saved {idx_path}")

    return embeddings, index


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/dataset.csv")
    parser.add_argument("--model",   default="models/distilbert_finetuned")
    parser.add_argument("--out",     default="data")
    args = parser.parse_args()
    build_index(args.dataset, args.model, args.out)
