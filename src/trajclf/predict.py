"""Inference on a full ride. This is the entry point the API should call.

Example:
    from trajclf.predict import RideClassifier
    clf = RideClassifier.load("artifacts")
    clf.predict(df_ride)   # DataFrame with columns ax, ay, az, rx, ry, rz (one row per 100 ms)
    -> {"label": "Safe", "probabilities": {...}, "n_windows": 23}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trajclf.config import SIGNAL_COLS
from trajclf.features import Normalizer, make_windows, preprocess
from trajclf.model import RideCNN
from trajclf.validation import validate_ride

MODEL_FILE = "model.pt"
CONFIG_FILE = "model_config.json"


class RideClassifier:
    def __init__(self, model: RideCNN, config: dict):
        self.model = model.eval()
        self.config = config
        self.classes: list[str] = config["classes"]
        self.normalizer = Normalizer.from_dict(config["normalizer"])

    # ---------- persistence ----------
    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), out_dir / MODEL_FILE)
        (out_dir / CONFIG_FILE).write_text(json.dumps(self.config, indent=2))

    @classmethod
    def load(cls, model_dir: str | Path) -> RideClassifier:
        model_dir = Path(model_dir)
        config = json.loads((model_dir / CONFIG_FILE).read_text())
        model = RideCNN(
            n_channels=config.get("n_features", len(config["signal_cols"])),
            n_classes=len(config["classes"]),
            width=config["width"],
        )
        model.load_state_dict(torch.load(model_dir / MODEL_FILE, map_location="cpu", weights_only=True))
        return cls(model, config)

    # ---------- inference ----------
    @staticmethod
    def to_signal(ride: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Accept a DataFrame (columns ax..rz) or a (T, 6) array; fill gaps like in training."""
        if not isinstance(ride, pd.DataFrame):
            arr = np.asarray(ride, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != len(SIGNAL_COLS):
                raise ValueError(f"Expected an array of shape (T, {len(SIGNAL_COLS)}), got {arr.shape}")
            ride = pd.DataFrame(arr, columns=SIGNAL_COLS)
        ride = validate_ride(ride)[SIGNAL_COLS].astype(np.float32)
        ride = ride.interpolate(limit_direction="both").fillna(0.0)
        return ride.to_numpy(dtype=np.float32)

    @torch.no_grad()
    def predict_proba(self, ride: np.ndarray | pd.DataFrame) -> np.ndarray:
        signal = self.normalizer.transform(preprocess(self.to_signal(ride)))
        windows = make_windows(signal, self.config["window"], self.config["stride"])
        probs = torch.softmax(self.model(torch.from_numpy(windows)), dim=1)
        return probs.mean(dim=0).numpy()  # ride score = average over its windows

    def predict(self, ride: np.ndarray | pd.DataFrame) -> dict:
        signal = self.to_signal(ride)
        proba = self.predict_proba(signal)
        n_windows = len(make_windows(signal, self.config["window"], self.config["stride"]))
        return {
            "label": self.classes[int(proba.argmax())],
            "probabilities": {c: round(float(p), 4) for c, p in zip(self.classes, proba)},
            "n_windows": n_windows,
        }
