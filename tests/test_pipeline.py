"""End-to-end smoke test: raw files -> prepare -> train -> reload -> predict, on synthetic rides."""

import json

import numpy as np
import pandas as pd
import pytest

from trajclf.config import CLASSES, SIGNAL_COLS
from trajclf.data import LABELS_FILE, RIDES_FILE, load_processed, prepare
from trajclf.predict import RideClassifier
from trajclf.train import fit_evaluate
from trajclf.validation import DataValidationError


@pytest.fixture
def raw_dir(tmp_path):
    rng = np.random.default_rng(0)
    frames, labels = [], []
    for i in range(50):
        label = CLASSES[i % len(CLASSES)]
        length = int(rng.integers(300, 600))
        sig = rng.normal(scale=1 + (i % len(CLASSES)), size=(length, 6))  # each class has its own variance
        f = pd.DataFrame(sig, columns=SIGNAL_COLS)
        f.insert(0, "time_index", np.arange(length) * 100)
        f.insert(0, "ride_id", f"ride{i:02d}")
        frames.append(f)
        labels.append({"RIDE_ID": f"ride{i:02d}", "LABEL": label})
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.concat(frames).to_parquet(raw / RIDES_FILE)
    pd.DataFrame(labels).to_csv(raw / LABELS_FILE, index=False)
    return raw


def test_full_pipeline(raw_dir, tmp_path):
    processed, artifacts = tmp_path / "processed", tmp_path / "artifacts"
    summary = prepare(raw_dir, processed, test_size=0.2)
    assert summary["n_rides"] == 50
    assert json.loads((processed / "validation_report.json").read_text())["passed"]

    rides, split = load_processed(processed)
    _, result = fit_evaluate(rides, split, artifacts, epochs=2, width=8, window=64, stride=32)
    assert 0 <= result["metrics"]["test_balanced_accuracy"] <= 1
    for name in ["model.pt", "model_config.json", "metrics.json", "confusion_matrix.csv"]:
        assert (artifacts / name).exists()

    clf = RideClassifier.load(artifacts)
    out = clf.predict(pd.DataFrame(rides["ride00"], columns=SIGNAL_COLS))
    assert out["label"] in CLASSES


def test_prepare_stops_on_invalid_data(raw_dir, tmp_path):
    labels = pd.read_csv(raw_dir / LABELS_FILE)
    labels.loc[0, "LABEL"] = "Crash"
    labels.to_csv(raw_dir / LABELS_FILE, index=False)
    with pytest.raises(DataValidationError, match="unknown labels"):
        prepare(raw_dir, tmp_path / "processed")
    report = json.loads((tmp_path / "processed" / "validation_report.json").read_text())
    assert not report["passed"]
