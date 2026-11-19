"""Pick one demo ride per class that the served model classifies correctly (issue #7).

    python scripts/pick_demo_rides.py                                   # model in artifacts/
    python scripts/pick_demo_rides.py --model-uri models:/ride-classifier@production   # MLflow model

For each class, scores the rides of the hold-out *test* split (never seen in training), keeps the
correctly predicted ride with the highest confidence and writes it to examples/ride_<class>.csv.
Classes with no correct test ride are reported, not faked: their file is written from the test ride
with the highest probability for its true class and flagged "misclassified" in examples/demo_rides.json.
Re-run it whenever @production changes, so the demo matches the model the API serves.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from trajclf.config import ARTIFACTS_DIR, CLASSES, PROCESSED_DIR, ROOT, SIGNAL_COLS
from trajclf.data import load_processed
from trajclf.predict import RideClassifier

EXAMPLES_DIR = ROOT / "examples"
SAMPLING_MS = 100


def slug(label: str) -> str:
    return label.lower().replace(" ", "_")


def load_classifier(model_dir: Path, model_uri: str | None) -> tuple[RideClassifier, str]:
    if model_uri:
        import mlflow

        return mlflow.pyfunc.load_model(model_uri).unwrap_python_model().clf, model_uri
    return RideClassifier.load(model_dir), str(model_dir)


def score_rides(clf: RideClassifier, rides: dict[str, np.ndarray], split: pd.DataFrame) -> pd.DataFrame:
    """One row per ride: true label, predicted label, confidence and probability of the true label."""
    rows = []
    for ride_id, label in zip(split.ride_id, split.label, strict=True):
        out = clf.predict(pd.DataFrame(rides[ride_id], columns=SIGNAL_COLS))
        rows.append(
            {
                "ride_id": ride_id,
                "label": label,
                "predicted": out["label"],
                "confidence": out["probabilities"][out["label"]],
                "p_true": out["probabilities"][label],
            }
        )
    return pd.DataFrame(rows)


def choose(scores: pd.DataFrame) -> dict[str, dict]:
    """Per class: best correct ride, else the ride closest to being correct (flagged)."""
    picks = {}
    for label in CLASSES:
        of_class = scores[scores.label == label]
        if of_class.empty:
            continue
        correct = of_class[of_class.predicted == label]
        if not correct.empty:
            row, status = correct.sort_values("confidence", ascending=False).iloc[0], "correct"
        else:
            row, status = of_class.sort_values("p_true", ascending=False).iloc[0], "misclassified"
        picks[label] = {
            "ride_id": row.ride_id,
            "predicted": row.predicted,
            "confidence": round(float(row.confidence), 4),
            "status": status,
        }
    return picks


def write_ride(signal: np.ndarray, path: Path) -> None:
    df = pd.DataFrame(signal, columns=SIGNAL_COLS).round(5)
    df.insert(0, "time_index", np.arange(len(df)) * SAMPLING_MS)
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--model-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--model-uri", default=None, help="e.g. models:/ride-classifier@production")
    parser.add_argument("--out-dir", type=Path, default=EXAMPLES_DIR)
    args = parser.parse_args()

    import torch  # only for the environment record

    clf, source = load_classifier(args.model_dir, args.model_uri)
    rides, split = load_processed(args.data_dir)
    test = split[split.split == "test"]
    scores = score_rides(clf, rides, test)
    picks = choose(scores)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for label, pick in picks.items():
        write_ride(rides[pick["ride_id"]], args.out_dir / f"ride_{slug(label)}.csv")
    manifest = {
        "model": source,
        "split": "test (hold-out rides, never used for training)",
        "test_accuracy_on_these_rides": round(float((scores.label == scores.predicted).mean()), 3),
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "os": platform.platform()},
        "rides": {f"ride_{slug(label)}.csv": {"label": label, **pick} for label, pick in picks.items()},
    }
    (args.out_dir / "demo_rides.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Model: {source}   test rides: {len(test)}")
    for label, pick in picks.items():
        mark = "✓" if pick["status"] == "correct" else "✗"
        print(f"  {mark} {label:<13} {pick['ride_id']}  predicted {pick['predicted']} ({pick['confidence']:.0%})")
    print(f"Wrote {len(picks)} rides + demo_rides.json to {args.out_dir}")


if __name__ == "__main__":
    main()
