import numpy as np
import pandas as pd
import pytest
import torch

from trajclf.config import CLASSES, SIGNAL_COLS
from trajclf.data import clean_ride, normalize_label, split_rides
from trajclf.features import Normalizer, make_windows
from trajclf.model import RideCNN
from trajclf.predict import RideClassifier


def test_normalize_label_handles_raw_spellings():
    assert normalize_label("Safe Overall") == "Safe overall"
    assert normalize_label(" reckless ") == "Reckless"
    with pytest.raises(ValueError):
        normalize_label("Crash")


def test_clean_ride_sorts_dedups_and_fills_nans():
    ride = pd.DataFrame({"time_index": [200, 0, 100, 100], **{c: [3.0, 1.0, np.nan, np.nan] for c in SIGNAL_COLS}})
    out = clean_ride(ride)
    assert out.time_index.tolist() == [0, 100, 200]
    assert out[SIGNAL_COLS].isna().sum().sum() == 0
    assert out.ax.tolist() == [1.0, 2.0, 3.0]  # linear interpolation


def test_split_is_stratified_and_disjoint():
    labels = pd.Series(["Safe"] * 80 + ["Reckless"] * 10 + ["Tandem"] * 10, index=[f"r{i}" for i in range(100)])
    split = split_rides(labels, test_size=0.2)
    test = split[split.split == "test"]
    assert len(test) == 20
    assert (test.label == "Reckless").sum() == 2
    assert set(test.ride_id).isdisjoint(split[split.split == "train"].ride_id)


@pytest.mark.parametrize("length,expected", [(100, 1), (256, 1), (512, 3), (600, 4)])
def test_make_windows_shapes(length, expected):
    w = make_windows(np.ones((length, 6), dtype=np.float32), window=256, stride=128)
    assert w.shape == (expected, 256, 6)


def test_normalizer_roundtrip():
    sig = [np.random.randn(500, 6).astype(np.float32) * 5 + 3]
    norm = Normalizer.fit(sig)
    z = norm.transform(sig[0])
    assert np.allclose(z.mean(axis=0), 0, atol=1e-4)
    assert Normalizer.from_dict(norm.to_dict()) == norm


def test_model_output_shape():
    logits = RideCNN(6, len(CLASSES), width=8)(torch.randn(4, 256, 6))
    assert logits.shape == (4, len(CLASSES))


def test_classifier_save_load_predict(tmp_path):
    config = {
        "classes": CLASSES,
        "signal_cols": SIGNAL_COLS,
        "window": 64,
        "stride": 32,
        "width": 8,
        "n_features": 8,
        "normalizer": {"mean": [0.0] * 8, "std": [1.0] * 8},
    }
    RideClassifier(RideCNN(8, len(CLASSES), width=8), config).save(tmp_path)
    clf = RideClassifier.load(tmp_path)

    ride = pd.DataFrame(np.random.randn(300, 6), columns=SIGNAL_COLS)
    ride.iloc[5, 2] = np.nan  # missing values must be handled
    out = clf.predict(ride)
    assert out["label"] in CLASSES
    assert abs(sum(out["probabilities"].values()) - 1) < 1e-3
    assert out["n_windows"] == 9

    with pytest.raises(ValueError):
        clf.predict(ride.drop(columns="rz"))
