"""Calibrate semantic similarity using labeled topic pairs and real embeddings."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# This script is an explicit calibration job, so it uses its own larger call budget.
os.environ["EMBEDDING_CALLS_PER_RUN"] = "64"
from src.embed import embed_text, reset_embedding_budget


def cosine(left, right):
    import math
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    return dot / (norm_left * norm_right) if norm_left and norm_right else 0.0


def main() -> int:
    pairs = json.loads((ROOT / "data" / "threshold_pairs.json").read_text(encoding="utf-8"))
    reset_embedding_budget()
    scores = []
    for group, label in (("related", 1), ("unrelated", 0)):
        for pair in pairs[group]:
            left, right = embed_text(pair[0]), embed_text(pair[1])
            if not left or not right:
                print("CALIBRATION UNAVAILABLE: embedding service returned no vector; no thresholds or recommendations computed.")
                return 2
            scores.append((cosine(left, right), label, pair))

    best = None
    print("threshold precision recall f1")
    for threshold in (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        tp = sum(score >= threshold and label == 1 for score, label, _ in scores)
        fp = sum(score >= threshold and label == 0 for score, label, _ in scores)
        fn = sum(score < threshold and label == 1 for score, label, _ in scores)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(f"{threshold:.2f} {precision:.3f} {recall:.3f} {f1:.3f}")
        candidate = (f1, precision, threshold)
        if best is None or candidate > best:
            best = candidate
    print(f"RECOMMENDATION: SEMANTIC_SIMILARITY_THRESHOLD={best[2]:.2f} (best F1; review labeled pairs before adopting).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
