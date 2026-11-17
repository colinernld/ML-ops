"""Turn variable-length rides into fixed-size, normalized windows."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from trajclf.config import WINDOW_SIZE, WINDOW_STRIDE

N_FEATURES = 8  # 6 centered IMU channels + acceleration magnitude + rotation magnitude


def preprocess(signal: np.ndarray) -> np.ndarray:
    """(T, 6) raw IMU -> (T, 8) orientation-robust features.

    Each phone is mounted differently, so the raw axes (gravity is ~9.8 on `ay` for one ride and
    split across axes for another) let the model memorise rides instead of riding styles.
    We remove each ride's median per channel and add the norms of acceleration and rotation,
    which do not depend on how the phone is held.
    """
    signal = signal.astype(np.float32)
    acc_norm = np.linalg.norm(signal[:, :3], axis=1, keepdims=True)
    rot_norm = np.linalg.norm(signal[:, 3:6], axis=1, keepdims=True)
    feats = np.concatenate([signal, acc_norm - np.median(acc_norm), rot_norm], axis=1)
    feats[:, :6] -= np.median(signal, axis=0)
    return feats


def make_windows(signal: np.ndarray, window: int = WINDOW_SIZE, stride: int = WINDOW_STRIDE) -> np.ndarray:
    """Cut a (T, C) signal into (n_windows, window, C). Short rides are zero-padded."""
    if signal.ndim != 2:
        raise ValueError(f"Expected a 2D (T, C) signal, got shape {signal.shape}")
    t, c = signal.shape
    if t < window:
        padded = np.zeros((window, c), dtype=signal.dtype)
        padded[:t] = signal
        return padded[None]
    starts = list(range(0, t - window + 1, stride))
    if starts[-1] != t - window:  # keep the end of the ride
        starts.append(t - window)
    return np.stack([signal[s : s + window] for s in starts])


@dataclass
class Normalizer:
    """Per-channel standardization, fitted on training rides only."""

    mean: list[float]
    std: list[float]

    @classmethod
    def fit(cls, signals: list[np.ndarray]) -> Normalizer:
        stacked = np.concatenate(signals, axis=0)
        std = stacked.std(axis=0)
        std[std == 0] = 1.0
        return cls(mean=stacked.mean(axis=0).tolist(), std=std.tolist())

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - np.asarray(self.mean, dtype=np.float32)) / np.asarray(self.std, dtype=np.float32)).astype(
            np.float32
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Normalizer:
        return cls(mean=list(d["mean"]), std=list(d["std"]))
