"""Choice of demo rides (issue #7): prefer correct predictions, never hide a misclassified class."""

import importlib.util
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location(
    "pick_demo_rides", Path(__file__).resolve().parents[1] / "scripts" / "pick_demo_rides.py"
)
pick = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pick)


def test_choose_prefers_most_confident_correct_ride():
    scores = pd.DataFrame(
        [
            {"ride_id": "a", "label": "Safe", "predicted": "Safe", "confidence": 0.6, "p_true": 0.6},
            {"ride_id": "b", "label": "Safe", "predicted": "Safe", "confidence": 0.9, "p_true": 0.9},
            {"ride_id": "c", "label": "Safe", "predicted": "Tandem", "confidence": 0.95, "p_true": 0.02},
        ]
    )
    assert pick.choose(scores)["Safe"] == {"ride_id": "b", "predicted": "Safe", "confidence": 0.9, "status": "correct"}


def test_choose_flags_class_without_correct_ride():
    scores = pd.DataFrame(
        [
            {"ride_id": "r1", "label": "Reckless", "predicted": "Safe", "confidence": 0.7, "p_true": 0.10},
            {"ride_id": "r2", "label": "Reckless", "predicted": "Tandem", "confidence": 0.5, "p_true": 0.30},
        ]
    )
    out = pick.choose(scores)["Reckless"]
    assert out["ride_id"] == "r2" and out["status"] == "misclassified"


def test_write_ride_round_trips_through_the_api_format(tmp_path):
    import numpy as np

    path = tmp_path / "ride.csv"
    pick.write_ride(np.ones((60, 6), dtype=np.float32), path)
    df = pd.read_csv(path)
    assert list(df.columns) == ["time_index", "ax", "ay", "az", "rx", "ry", "rz"]
    assert df.time_index.iloc[-1] == 5900 and len(df) == 60
