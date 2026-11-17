"""Shared constants for the data + model part of the project."""

from pathlib import Path

# IMU channels: accelerometer (m/s^2) and gyroscope (deg/s), sampled every 100 ms (10 Hz)
SIGNAL_COLS = ["ax", "ay", "az", "rx", "ry", "rz"]

# Order matters: index i in the model output == CLASSES[i]
CLASSES = ["Safe", "Safe overall", "Bad weather", "Tandem", "Reckless"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Windowing: 256 samples = 25.6 s of riding, 50 % overlap
WINDOW_SIZE = 256
WINDOW_STRIDE = 128

SEED = 42

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"
