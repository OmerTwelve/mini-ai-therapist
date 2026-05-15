"""
src/pipeline.py
Unified inference pipeline — given a raw user statement:
  1. Classify  → predicted mental state + per-class probabilities
  2. Retrieve  → top-k semantically similar statements via FAISS
  3. Map       → curated resources from mental_health_resources.json

All heavy objects are loaded once at construction time.
"""

import json
import numpy as np
import pandas as pd
import torch
import faiss
from dataclasses import dataclass, field
from pathlib import Path
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    DistilBertModel,
)


# ── Return types ──────────────────────────────────────────────────────────────

@dataclass
class SimilarStatement:
    statement: str
    mental_state: str
    distance: float   # L2 distance — lower = more similar


@dataclass
class PipelineResult:
    predicted_state: str
    confidence: float                        # probability of top class
    all_probabilities: dict                  # {class_name: prob}
    resources: list[dict]
    similar_statements: list[SimilarStatement]


# ── Pipeline ──────────────────────────────────────────────────────────────────

class MentalHealthPipeline:
    """
    Load-once, run-many inference pipeline.

    Args:
        model_dir      : directory produced by src/finetune.py
        data_dir       : directory produced by src/data_prep.py + src/build_index.py
        resources_path : path to mental_health_resources.json
        top_k          : number of similar statements to retrieve
    """

    MAX_LENGTH = 256

    def __init__(
        self,
        model_dir: str = "models/distilbert_finetuned",
        data_dir: str  = "data",
        resources_path: str = "mental_health_resources.json",
        top_k: int = 5,
    ):
        self.top_k  = top_k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[pipeline] Loading on {self.device}…")
        self._load_models(model_dir)
        self._load_index(data_dir)
        self._load_resources(resources_path)
        print("[pipeline] Ready.")

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _load_models(self, model_dir: str):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)

        # Classification head — used for predict step
        self.classifier = DistilBertForSequenceClassification.from_pretrained(model_dir)
        self.classifier.to(self.device).eval()

        # Base encoder — used for embedding step
        # Shares weights with classifier but without the classification head
        self.encoder = DistilBertModel.from_pretrained(model_dir)
        self.encoder.to(self.device).eval()

        # id2label from model config (saved during fine-tuning)
        self.id2label: dict[int, str] = self.classifier.config.id2label

    def _load_index(self, data_dir: str):
        data_dir = Path(data_dir)
        self.index: faiss.Index = faiss.read_index(str(data_dir / "faiss_index.bin"))
        self.dataset: pd.DataFrame = pd.read_csv(data_dir / "dataset.csv")
        print(f"[pipeline] FAISS index: {self.index.ntotal:,} vectors")

    def _load_resources(self, resources_path: str):
        with open(resources_path, "r") as f:
            self.resources: dict = json.load(f)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> dict:
        """Tokenize a single string, move tensors to device."""
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    def _get_cls_embedding(self, text: str) -> np.ndarray:
        """Return (1, 768) float32 array — CLS token of encoder output."""
        inputs = self._tokenize(text)
        with torch.no_grad():
            out = self.encoder(**inputs)
        return out.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(self, text: str) -> tuple[str, float, dict]:
        """
        Returns (predicted_label, confidence, {label: prob}).
        confidence = softmax probability of the top class.
        """
        inputs = self._tokenize(text)
        with torch.no_grad():
            logits = self.classifier(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        top_idx   = int(np.argmax(probs))
        all_probs = {self.id2label[i]: float(p) for i, p in enumerate(probs)}
        return self.id2label[top_idx], float(probs[top_idx]), all_probs

    def retrieve(self, text: str) -> list[SimilarStatement]:
        """
        Return top-k most semantically similar statements from the dataset.
        """
        embedding = self._get_cls_embedding(text)
        distances, indices = self.index.search(embedding, self.top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.dataset):
                continue   # FAISS can return -1 for empty slots
            row = self.dataset.iloc[idx]
            results.append(
                SimilarStatement(
                    statement=str(row["statement"]),
                    mental_state=str(row["status"]),
                    distance=float(dist),
                )
            )
        return results

    def get_resources(self, mental_state: str) -> list[dict]:
        """Lookup curated resources by mental state label."""
        return self.resources.get(mental_state, [])

    def run(self, user_statement: str) -> PipelineResult:
        """
        Full pipeline: classify → retrieve → map resources.
        Returns a PipelineResult dataclass.
        """
        label, confidence, all_probs = self.classify(user_statement)
        similar                      = self.retrieve(user_statement)
        resources                    = self.get_resources(label)

        return PipelineResult(
            predicted_state=label,
            confidence=confidence,
            all_probabilities=all_probs,
            resources=resources,
            similar_statements=similar,
        )


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = MentalHealthPipeline()
    result   = pipeline.run("I've been feeling really anxious and can't stop worrying")

    print(f"\nPredicted state : {result.predicted_state}  ({result.confidence:.1%})")
    print(f"Resources       : {len(result.resources)}")
    print(f"Similar found   : {len(result.similar_statements)}")
    for s in result.similar_statements:
        print(f"  [{s.mental_state}] dist={s.distance:.2f}  {s.statement[:80]}")
