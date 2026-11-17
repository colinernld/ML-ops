"""Load, clean and split the GBG500 rides.

Usage:
    python -m trajclf.data --raw-dir data/raw --out-dir data/processed
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from trajclf.config import CLASSES, PROCESSED_DIR, RAW_DIR, SEED, SIGNAL_COLS

log = logging.getLogger(__name__)

RIDES_FILE = "GBG500.parquet"
LABELS_FILE = "GBG500_labels.csv"


def normalize_label(label: str) -> str:
    """Map raw label spellings ('Safe Overall', 'safe overall'...) to CLASSES."""
    lookup = {c.lower(): c for c in CLASSES}
    key = str(label).strip().lower()
    if key not in lookup:
        raise ValueError(f"Unknown label {label!r}, expected one of {CLASSES}")
    return lookup[key]


def load_labels(path: str | Path) -> pd.Series:
    """Return a Series ride_id -> normalized label."""
    df = pd.read_csv(path)
    return df.set_index("RIDE_ID")["LABEL"].map(normalize_label).rename("label")


def clean_ride(ride: pd.DataFrame) -> pd.DataFrame:
    """Sort by time, drop duplicated timestamps and fill missing sensor values."""
    ride = ride.sort_values("time_index").drop_duplicates("time_index")
    ride[SIGNAL_COLS] = ride[SIGNAL_COLS].interpolate(limit_direction="both").fillna(0.0)
    return ride


def load_rides(path: str | Path) -> dict[str, np.ndarray]:
    """Return {ride_id: array of shape (T, 6)} with cleaned float32 signals."""
    df = pd.read_parquet(path, columns=["ride_id", "time_index", *SIGNAL_COLS])
    rides = {}
    for ride_id, ride in df.groupby("ride_id", sort=True):
        rides[ride_id] = clean_ride(ride)[SIGNAL_COLS].to_numpy(dtype=np.float32)
    log.info("Loaded %d rides from %s", len(rides), path)
    return rides


def split_rides(labels: pd.Series, test_size: float = 0.2, seed: int = SEED) -> pd.DataFrame:
    """Stratified train/test split at ride level (no window leakage between sets)."""
    _, test_ids = train_test_split(
        labels.index.to_numpy(), test_size=test_size, stratify=labels.to_numpy(), random_state=seed
    )
    split = pd.Series("train", index=labels.index, name="split")
    split.loc[test_ids] = "test"
    return pd.concat([labels.rename("label"), split], axis=1).rename_axis("ride_id").reset_index()


def prepare(raw_dir: Path = RAW_DIR, out_dir: Path = PROCESSED_DIR, test_size: float = 0.2) -> dict:
    """Full pipeline: raw parquet + csv -> processed/rides.npz + processed/split.csv."""
    rides = load_rides(raw_dir / RIDES_FILE)
    labels = load_labels(raw_dir / LABELS_FILE)

    missing = set(labels.index) ^ set(rides)
    if missing:
        raise ValueError(f"{len(missing)} rides have a signal but no label (or the reverse)")

    split = split_rides(labels, test_size=test_size)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "rides.npz", **rides)
    split.to_csv(out_dir / "split.csv", index=False)

    summary = {
        "n_rides": len(rides),
        "n_samples": int(sum(len(r) for r in rides.values())),
        "class_counts": split.groupby(["split", "label"]).size().unstack(0).fillna(0).astype(int).to_dict(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Wrote processed data to %s", out_dir)
    return summary


def load_processed(out_dir: Path = PROCESSED_DIR) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    with np.load(out_dir / "rides.npz") as npz:
        rides = {k: npz[k] for k in npz.files}
    return rides, pd.read_csv(out_dir / "split.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(prepare(args.raw_dir, args.out_dir, args.test_size), indent=2))


if __name__ == "__main__":
    main()
