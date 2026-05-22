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
import inspect
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import plotly.express as px
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_recall_fscore_support,
)
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


def _training_device() -> torch.device:
    """Select the training device and print what will be used."""
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        print(f"[finetune] Training device: {device} ({torch.cuda.get_device_name(0)})")
        return device

    device = torch.device("cpu")
    print("[finetune] Training device: cpu")
    return device


def _device_training_args(device: torch.device) -> dict:
    """
    Build TrainingArguments device flags across Transformers versions.
    Older versions use no_cuda; newer versions use use_cpu.
    """
    params = inspect.signature(TrainingArguments.__init__).parameters
    use_gpu = device.type == "cuda"

    if "use_cpu" in params:
        return {"use_cpu": not use_gpu}
    if "no_cuda" in params:
        return {"no_cuda": not use_gpu}
    return {}


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
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
    }


def _count_parameters(model) -> dict:
    """Return total and learnable parameter counts for the model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": int(total),
        "learnable_parameters": int(trainable),
        "frozen_parameters": int(total - trainable),
    }


def _prediction_metrics(trainer, dataset, split_name: str, class_names: list[str]) -> tuple[dict, np.ndarray, np.ndarray]:
    """Run prediction for one split and return classification metrics."""
    output = trainer.predict(dataset)
    preds = np.argmax(output.predictions, axis=-1)
    labels = output.label_ids
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    report = classification_report(
        labels,
        preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "split": split_name,
        "accuracy": float(accuracy_score(labels, preds)),
        "label_id_mse": float(mean_squared_error(labels, preds)),
        "label_id_rmse": float(np.sqrt(mean_squared_error(labels, preds))),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "labels": labels.tolist(),
        "predictions": preds.tolist(),
        "per_class": {
            class_name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
                "support": int(support[i]),
            }
            for i, class_name in enumerate(class_names)
        },
        "classification_report": report,
    }
    return metrics, labels, preds


def _history_frame(log_history: list[dict]) -> pd.DataFrame:
    """Convert Trainer log history into a tabular form for saving and plotting."""
    rows = []
    for row in log_history:
        clean = {
            key: value
            for key, value in row.items()
            if isinstance(value, (int, float, str, bool))
        }
        if clean:
            rows.append(clean)
    return pd.DataFrame(rows)


def _save_matplotlib_line(df: pd.DataFrame, x_col: str, y_cols: list[str], title: str, ylabel: str, path: str):
    """Save a simple Matplotlib line chart."""
    plot_df = df.dropna(subset=y_cols, how="all")
    if plot_df.empty or x_col not in plot_df:
        return

    plt.figure(figsize=(9, 5))
    for col in y_cols:
        if col in plot_df:
            plt.plot(plot_df[x_col], plot_df[col], marker="o", label=col)
    plt.title(title)
    plt.xlabel(x_col.title())
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _save_confusion_matrix(labels: list[int], preds: list[int], class_names: list[str], path: str):
    """Save a confusion matrix as a Matplotlib PNG."""
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))

    plt.figure(figsize=(9, 7))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Validation Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    threshold = cm.max() / 2 if cm.size and cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _save_per_class_metric_bars(per_class: dict, path: str):
    """Save precision, recall, and F1 per class as grouped Matplotlib bars."""
    class_names = list(per_class.keys())
    precision = [per_class[name]["precision"] for name in class_names]
    recall = [per_class[name]["recall"] for name in class_names]
    f1 = [per_class[name]["f1_score"] for name in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    plt.figure(figsize=(11, 6))
    plt.bar(x - width, precision, width, label="Precision")
    plt.bar(x, recall, width, label="Recall")
    plt.bar(x + width, f1, width, label="F1-score")
    plt.title("Validation Precision, Recall, and F1-score by Class")
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(x, class_names, rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _save_evaluation_artifacts(
    eval_dir: str,
    trainer,
    train_metrics: dict,
    val_metrics: dict,
    param_counts: dict,
    class_names: list[str],
):
    """Save metric reports and training curves to model_evaluation/."""
    os.makedirs(eval_dir, exist_ok=True)

    history_df = _history_frame(trainer.state.log_history)
    history_csv = os.path.join(eval_dir, "training_history.csv")
    history_df.to_csv(history_csv, index=False)

    summary = {
        "task_type": "classification",
        "model_name": BASE_MODEL,
        "num_classes": len(class_names),
        "classes": class_names,
        "model_size": param_counts,
        "regression_metrics": {
            "mse": val_metrics["label_id_mse"],
            "rmse": val_metrics["label_id_rmse"],
            "note": "Computed on encoded class IDs because this model performs classification, not continuous-value regression.",
        },
        "train": train_metrics,
        "validation": val_metrics,
        "artifacts": {
            "summary_json": "model_evaluation.json",
            "summary_txt": "model_evaluation.txt",
            "per_class_csv": "per_class_metrics.csv",
            "training_history_csv": "training_history.csv",
            "accuracy_chart_html": "accuracy_curve.html",
            "accuracy_chart_png": "accuracy_curve.png",
            "loss_chart_html": "loss_curve.html",
            "loss_chart_png": "loss_curve.png",
            "final_split_accuracy_chart_html": "final_split_accuracy.html",
            "final_split_accuracy_chart_png": "final_split_accuracy.png",
            "confusion_matrix_png": "confusion_matrix.png",
            "precision_recall_f1_png": "precision_recall_f1.png",
        },
    }

    with open(os.path.join(eval_dir, "model_evaluation.json"), "w") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "Model Evaluation",
        "================",
        f"Task type: classification",
        f"Base model: {BASE_MODEL}",
        f"Classes: {', '.join(class_names)}",
        "",
        "Model size",
        f"  Total parameters: {param_counts['total_parameters']:,}",
        f"  Learnable parameters: {param_counts['learnable_parameters']:,}",
        f"  Frozen parameters: {param_counts['frozen_parameters']:,}",
        "",
        "Final metrics",
        f"  Train accuracy: {train_metrics['accuracy']:.4f}",
        f"  Train macro F1: {train_metrics['f1_macro']:.4f}",
        f"  Train label-ID MSE: {train_metrics['label_id_mse']:.4f}",
        f"  Validation accuracy: {val_metrics['accuracy']:.4f}",
        f"  Validation macro F1: {val_metrics['f1_macro']:.4f}",
        f"  Validation label-ID MSE: {val_metrics['label_id_mse']:.4f}",
        f"  Validation label-ID RMSE: {val_metrics['label_id_rmse']:.4f}",
        "",
        "MSE / RMSE",
        "  Computed on encoded class IDs because this is a classification model.",
        "",
        "Per-class validation metrics",
    ]
    for class_name, metrics in val_metrics["per_class"].items():
        lines.append(
            f"  {class_name}: precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, f1={metrics['f1_score']:.4f}, "
            f"support={metrics['support']}"
        )
    with open(os.path.join(eval_dir, "model_evaluation.txt"), "w") as f:
        f.write("\n".join(lines))

    per_class_rows = []
    for split_metrics in (train_metrics, val_metrics):
        for class_name, metrics in split_metrics["per_class"].items():
            per_class_rows.append({"split": split_metrics["split"], "class": class_name, **metrics})
    pd.DataFrame(per_class_rows).to_csv(os.path.join(eval_dir, "per_class_metrics.csv"), index=False)

    eval_rows = history_df.dropna(subset=["eval_accuracy"], how="all") if "eval_accuracy" in history_df else pd.DataFrame()
    if not eval_rows.empty:
        fig = px.line(
            eval_rows,
            x="epoch",
            y="eval_accuracy",
            markers=True,
            title="Validation Accuracy During Training",
            labels={"epoch": "Epoch", "eval_accuracy": "Accuracy"},
        )
        fig.write_html(os.path.join(eval_dir, "accuracy_curve.html"))
        _save_matplotlib_line(
            eval_rows,
            "epoch",
            ["eval_accuracy"],
            "Validation Accuracy During Training",
            "Accuracy",
            os.path.join(eval_dir, "accuracy_curve.png"),
        )

    loss_columns = [col for col in ["loss", "eval_loss"] if col in history_df.columns]
    if loss_columns:
        plot_df = history_df[[col for col in ["epoch", "step", *loss_columns] if col in history_df.columns]]
        id_vars = [col for col in ["epoch", "step"] if col in plot_df.columns]
        loss_df = plot_df.melt(id_vars=id_vars, value_vars=loss_columns, var_name="metric", value_name="loss").dropna()
        x_axis = "epoch" if "epoch" in loss_df.columns else "step"
        fig = px.line(
            loss_df,
            x=x_axis,
            y="loss",
            color="metric",
            markers=True,
            title="Training and Validation Loss",
        )
        fig.write_html(os.path.join(eval_dir, "loss_curve.html"))
        x_axis = "epoch" if "epoch" in history_df.columns else "step"
        _save_matplotlib_line(
            history_df,
            x_axis,
            loss_columns,
            "Training and Validation Loss",
            "Loss",
            os.path.join(eval_dir, "loss_curve.png"),
        )

    split_accuracy_df = pd.DataFrame(
        [
            {"split": "train", "accuracy": train_metrics["accuracy"]},
            {"split": "validation", "accuracy": val_metrics["accuracy"]},
        ]
    )
    fig = px.bar(
        split_accuracy_df,
        x="split",
        y="accuracy",
        text="accuracy",
        range_y=[0, 1],
        title="Final Split Accuracy",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.write_html(os.path.join(eval_dir, "final_split_accuracy.html"))

    plt.figure(figsize=(6, 5))
    plt.bar(split_accuracy_df["split"], split_accuracy_df["accuracy"], color=["#4C78A8", "#F58518"])
    plt.title("Final Split Accuracy")
    plt.xlabel("Split")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    for i, value in enumerate(split_accuracy_df["accuracy"]):
        plt.text(i, value + 0.02, f"{value:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "final_split_accuracy.png"), dpi=160)
    plt.close()

    _save_confusion_matrix(
        val_metrics["labels"],
        val_metrics["predictions"],
        class_names,
        os.path.join(eval_dir, "confusion_matrix.png"),
    )
    _save_per_class_metric_bars(
        val_metrics["per_class"],
        os.path.join(eval_dir, "precision_recall_f1.png"),
    )

    print(f"[finetune] Evaluation files saved to {eval_dir}")


# ── Main training function ────────────────────────────────────────────────────

def finetune(
    train_csv: str,
    val_csv: str,
    output_dir: str = "models/distilbert_finetuned",
    eval_dir: str = "model_evaluation",
):
    """
    Fine-tune DistilBERT on the mental health classification task.

    Saves model + tokenizer + label_mapping.json to output_dir.
    Saves evaluation reports and charts to eval_dir.
    Returns (trainer, label_encoder).
    """
    os.makedirs(output_dir, exist_ok=True)
    device = _training_device()

    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)

    # Encode string labels → integers
    train_df, val_df, le = _encode_labels(train_df, val_df)

    num_labels = len(le.classes_)
    class_names = [str(c) for c in le.classes_]
    id2label   = {i: c for i, c in enumerate(class_names)}
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
    ).to(device)
    param_counts = _count_parameters(model)
    print("[finetune] Model size:")
    print(f"  total parameters     : {param_counts['total_parameters']:,}")
    print(f"  learnable parameters : {param_counts['learnable_parameters']:,}")
    print(f"  frozen parameters    : {param_counts['frozen_parameters']:,}")

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
        fp16=device.type == "cuda",       # mixed precision on GPU
        dataloader_pin_memory=device.type == "cuda",
        report_to="none",                 # disable wandb / tensorboard noise
        **_device_training_args(device),
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
    train_metrics, _, _ = _prediction_metrics(trainer, train_ds, "train", class_names)
    val_metrics, labels, preds = _prediction_metrics(trainer, val_ds, "validation", class_names)

    print("\n[finetune] Final train metrics:")
    print(f"  accuracy : {train_metrics['accuracy']:.4f}")
    print(f"  macro F1 : {train_metrics['f1_macro']:.4f}")

    print("\n[finetune] Final validation metrics:")
    print(f"  accuracy : {val_metrics['accuracy']:.4f}")
    print(f"  macro F1 : {val_metrics['f1_macro']:.4f}")

    print("\n[finetune] Validation classification report:")
    print(classification_report(labels, preds, target_names=class_names, zero_division=0))

    _save_evaluation_artifacts(
        eval_dir=eval_dir,
        trainer=trainer,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        param_counts=param_counts,
        class_names=class_names,
    )

    return trainer, le


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--val",   default="data/val.csv")
    parser.add_argument("--out",   default="models/distilbert_finetuned")
    parser.add_argument("--eval-dir", default="model_evaluation")
    args = parser.parse_args()
    finetune(args.train, args.val, args.out, args.eval_dir)
