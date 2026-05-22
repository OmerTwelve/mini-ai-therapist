"""
run_pipeline.py
Orchestrator — runs every stage in order.

Usage:
    python run_pipeline.py --data Combined_Data.csv

Stages:
    1. data_prep   : clean + split CSV
    2. finetune    : train DistilBERT classifier
    3. build_index : extract embeddings + build FAISS index
    (4. app        : `streamlit run app.py` — must be run separately)
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run all AI-therapist pipeline stages.")
    parser.add_argument("--data",       required=True,  help="Path to Combined_Data.csv")
    parser.add_argument("--data-dir",   default="data",                    help="Output dir for data artefacts")
    parser.add_argument("--model-dir",  default="models/distilbert_finetuned", help="Output dir for trained model")
    parser.add_argument("--eval-dir",   default="model_evaluation",        help="Output dir for model evaluation reports")
    parser.add_argument("--resources",  default="mental_health_resources.json", help="Resources JSON")
    parser.add_argument("--skip-train", action="store_true",  help="Skip fine-tuning (use existing model)")
    parser.add_argument("--skip-index", action="store_true",  help="Skip embedding + FAISS build")
    args = parser.parse_args()

    csv_path = Path(args.data)
    if not csv_path.exists():
        print(f"[error] Dataset not found: {csv_path}")
        sys.exit(1)

    # ── Stage 1: data prep ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 1 / 3 — Data Preparation")
    print("=" * 60)
    from src.data_prep import prepare_data
    prepare_data(str(csv_path), args.data_dir)

    # ── Stage 2: fine-tune ────────────────────────────────────────────────────
    if not args.skip_train:
        print("\n" + "=" * 60)
        print("STAGE 2 / 3 — Fine-tuning DistilBERT")
        print("=" * 60)
        from src.finetune import finetune
        finetune(
            train_csv=f"{args.data_dir}/train.csv",
            val_csv=f"{args.data_dir}/val.csv",
            output_dir=args.model_dir,
            eval_dir=args.eval_dir,
        )
    else:
        print("\n[skip] Fine-tuning skipped — using existing model.")

    # ── Stage 3: build FAISS index ────────────────────────────────────────────
    if not args.skip_index:
        print("\n" + "=" * 60)
        print("STAGE 3 / 3 — Building Embeddings + FAISS Index")
        print("=" * 60)
        from src.build_index import build_index
        build_index(
            dataset_csv=f"{args.data_dir}/dataset.csv",
            model_dir=args.model_dir,
            output_dir=args.data_dir,
        )
    else:
        print("\n[skip] Index build skipped — using existing index.")

    print("\n" + "=" * 60)
    print("ALL STAGES COMPLETE")
    print(f"  Model     : {args.model_dir}/")
    print(f"  Data      : {args.data_dir}/")
    print(f"  Evaluation: {args.eval_dir}/")
    print(f"  Resources : {args.resources}")
    print("\nTo launch the UI:")
    print("  streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
