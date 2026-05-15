"""
src/finetune.py
Fine-tune DistilBERT for 7-class mental health classification.

Usage:
    python -m src.finetune \
        --train data/train.csv \
        --val   data/val.csv \
        --out   models/distilbert_finetuned
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)


# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 256          # 256 covers >99 % of statements; 512 is slower with no gain
BATCH_SIZE = 16           # reduce to 8 if CUDA OOM
EPOCHS = 4
WARMUP_STEPS = 300
WEIGHT_DECAY = 0.01


# ── Dataset helpers ───────────────────────────────────────────────────────────

def _encode_labels(train_df, val_df):
    """Fit LabelEncoder on train, transform both splits. Returns (train_df, val_df, le)."""
    le = LabelEncoder()
    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df["label"] = le.fit_transform(train_df["status"])
    val_df["label"] = le.transform(val_df["status"])
    return train_df, val_df, le


def _hf_dataset(df: pd.DataFrame, tokenizer) -> Dataset:
    """Convert DataFrame to tokenized HuggingFace Dataset."""
    ds = Dataset.from_pandas(df[["statement", "label"]].reset_index(drop=True))

    def tokenize(batch):
        return tokenizer(
            batch["statement"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )

    ds = ds.map(tokenize, batched=True, remove_columns=["statement"])
    ds = ds.rename_column("label", "labels")   # Trainer expects 'labels'
    ds.set_format("torch")
    return ds


# ── Metrics ───────────────────────────────────────────────────────────────────

def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }


# ── Main training function ────────────────────────────────────────────────────

def finetune(train_csv: str, val_csv: str, output_dir: str = "models/distilbert_finetuned"):
    """
    Fine-tune DistilBERT on the mental health classification task.

    Saves model + tokenizer + label_mapping.json to output_dir.
    Returns (trainer, label_encoder).
    """
    os.makedirs(output_dir, exist_ok=True)

    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)

    # Encode string labels → integers
    train_df, val_df, le = _encode_labels(train_df, val_df)

    num_labels = len(le.classes_)
    id2label   = {i: c for i, c in enumerate(le.classes_)}
    label2id   = {c: i for i, c in id2label.items()}
    print(f"[finetune] Classes ({num_labels}): {id2label}")

    # Tokeniser
    tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL)

    train_ds = _hf_dataset(train_df, tokenizer)
    val_ds   = _hf_dataset(val_df,   tokenizer)

    # Model — inject label maps so config.json captures them
    model = DistilBertForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=100,
        fp16=torch.cuda.is_available(),   # mixed precision on GPU
        report_to="none",                 # disable wandb / tensorboard noise
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("[finetune] Training…")
    trainer.train()

    # ── Save artefacts ────────────────────────────────────────────────────────
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save label mapping separately for quick loading in pipeline
    label_map = {"id2label": id2label, "label2id": label2id, "classes": list(le.classes_)}
    with open(os.path.join(output_dir, "label_mapping.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"[finetune] Model saved to {output_dir}")

    # ── Final evaluation on validation set ───────────────────────────────────
    preds_out = trainer.predict(val_ds)
    preds     = np.argmax(preds_out.predictions, axis=-1)
    labels    = preds_out.label_ids
    print("\n[finetune] Validation classification report:")
    print(classification_report(labels, preds, target_names=le.classes_, zero_division=0))

    return trainer, le


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--val",   default="data/val.csv")
    parser.add_argument("--out",   default="models/distilbert_finetuned")
    args = parser.parse_args()
    finetune(args.train, args.val, args.out)
