import numpy as np
import pandas as pd
import pytest

from trajclf.config import SIGNAL_COLS
from trajclf.validation import DataValidationError, validate_dataset, validate_ride


def make_raw(n_rides=3, length=200, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_rides):
        f = pd.DataFrame(rng.normal(size=(length, 6)), columns=SIGNAL_COLS)
        f.insert(0, "time_index", np.arange(length) * 100)
        f.insert(0, "ride_id", f"ride{i}")
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def make_labels(ids, label="Safe"):
    return pd.DataFrame({"RIDE_ID": ids, "LABEL": label})


def test_valid_dataset_passes():
    raw = make_raw()
    report = validate_dataset(raw, make_labels(raw.ride_id.unique()))
    assert report.passed, report.errors
    assert report.stats["n_rides"] == 3


def test_missing_column_fails():
    report = validate_dataset(make_raw().drop(columns="rz"))
    assert not report.passed and "missing columns" in report.errors[0]


def test_too_many_missing_values_fails():
    raw = make_raw()
    raw.loc[raw.sample(frac=0.1, random_state=0).index, "ax"] = np.nan
    assert not validate_dataset(raw).passed


def test_wrong_sampling_rate_fails():
    raw = make_raw()
    raw["time_index"] = raw["time_index"] * 2  # 5 Hz instead of 10 Hz
    assert not validate_dataset(raw).passed


def test_out_of_range_values_fail():
    raw = make_raw()
    raw.loc[0, "ry"] = 1e5
    report = validate_dataset(raw)
    assert not report.passed and "rotation" in report.errors[0]


def test_label_problems_are_reported():
    raw = make_raw()
    labels = make_labels(["ride0", "ride1", "ghost"], label="Crash")
    errors = " ".join(validate_dataset(raw, labels).errors)
    assert "unknown labels" in errors
    assert "without label" in errors
    assert "labels without ride" in errors


def test_validate_ride_rejects_bad_input():
    ride = make_raw(1).drop(columns="ride_id")
    assert len(validate_ride(ride)) == 200
    with pytest.raises(DataValidationError, match="samples"):
        validate_ride(ride.head(10))
    with pytest.raises(DataValidationError, match="missing columns"):
        validate_ride(ride.drop(columns="ax"))
    bad = ride.copy()
    bad.loc[: len(bad) // 2, "az"] = np.nan
    with pytest.raises(DataValidationError, match="missing values"):
        validate_ride(bad)
    bad = ride.copy()
    bad.loc[3, "ax"] = np.inf
    with pytest.raises(DataValidationError, match="infinity"):
        validate_ride(bad)
