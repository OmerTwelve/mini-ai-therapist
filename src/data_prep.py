"""
src/data_prep.py
Load, clean, and split Combined_Data.csv into train/val/test sets.
Saves splits + full clean dataset + metadata JSON to output_dir.
"""

import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split


# ── Constants ────────────────────────────────────────────────────────────────
MIN_LEN = 5        # chars — filters noise / accidental rows
MAX_LEN = 5000     # chars — prevents tokenizer overflow
RANDOM_STATE = 42
SPLITS = (0.70, 0.15, 0.15)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalise column names to lowercase, strip spaces
    df.columns = df.columns.str.strip().str.lower()
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop nulls and filter by statement length."""
    df = df.dropna(subset=["statement", "status"]).copy()
    df["statement"] = df["statement"].astype(str).str.strip()
    mask = df["statement"].str.len().between(MIN_LEN, MAX_LEN)
    return df[mask].reset_index(drop=True)


def _split(df: pd.DataFrame):
    """Stratified 70 / 15 / 15 split."""
    train, temp = train_test_split(
        df,
        test_size=1 - SPLITS[0],
        random_state=RANDOM_STATE,
        stratify=df["status"],
    )
    val, test = train_test_split(
        temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=temp["status"],
    )
    return train, val, test


def _build_metadata(df, train, val, test) -> dict:
    lengths = df["statement"].str.len()
    return {
        "total_samples": len(df),
        "train_samples": len(train),
        "val_samples": len(val),
        "test_samples": len(test),
        "num_classes": int(df["status"].nunique()),
        "classes": sorted(df["status"].unique().tolist()),
        "label_counts": df["status"].value_counts().to_dict(),
        "avg_statement_length": round(float(lengths.mean()), 1),
        "min_statement_length": int(lengths.min()),
        "max_statement_length": int(lengths.max()),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def prepare_data(input_path: str, output_dir: str = "data"):
    """
    Full pipeline: load → clean → split → save.

    Returns (train_df, val_df, test_df, full_df).
    All DataFrames have columns: statement, status.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"[data_prep] Loading {input_path}")
    df = _load(input_path)
    print(f"  raw shape: {df.shape}")

    df = _clean(df)
    print(f"  clean shape: {df.shape}")

    label_counts = df["status"].value_counts()
    print("  label distribution:")
    for label, n in label_counts.items():
        print(f"    {label}: {n}  ({n / len(df) * 100:.1f}%)")

    train, val, test = _split(df)
    print(f"  splits → train={len(train)}  val={len(val)}  test={len(test)}")

    # Save splits + full clean dataset
    for name, frame in [("train", train), ("val", val), ("test", test), ("dataset", df)]:
        path = os.path.join(output_dir, f"{name}.csv")
        frame.to_csv(path, index=False)
        print(f"  saved {path}")

    # Save metadata for downstream modules to reference class list etc.
    meta = _build_metadata(df, train, val, test)
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  saved {meta_path}")

    return train, val, test, df


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "Combined_Data.csv"
    prepare_data(csv_path)
