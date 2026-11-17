"""Data quality checks.

Two levels:
- `validate_dataset`: run on the raw training data before every (re)training. A failing check
  stops the pipeline instead of silently training on broken data.
- `validate_ride`: run on every ride sent to the model (API, batch scoring) to reject bad input
  with a clear message.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from trajclf.config import CLASSES, SIGNAL_COLS

SAMPLING_MS = 100  # 10 Hz
# Physical limits of phone sensors: accelerometer ~±16 g, gyroscope ~±2000 deg/s
ACC_LIMIT = 16 * 9.81
GYRO_LIMIT = 2000.0
MAX_MISSING_RATIO = 0.05  # dataset level
MAX_MISSING_RATIO_RIDE = 0.2  # single ride
MIN_RIDE_SAMPLES = 50  # 5 s
MAX_RIDE_SAMPLES = 36_000  # 1 h


class DataValidationError(ValueError):
    pass


@dataclass
class ValidationReport:
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def error(self, msg: str) -> None:
        self.passed = False
        self.errors.append(msg)

    def to_dict(self) -> dict:
        return asdict(self)


def _check_ranges(values: pd.DataFrame, report: ValidationReport) -> None:
    acc = values[["ax", "ay", "az"]].abs().max().max()
    gyro = values[["rx", "ry", "rz"]].abs().max().max()
    if acc > ACC_LIMIT:
        report.error(f"acceleration out of physical range: max |a| = {acc:.1f} m/s² (limit {ACC_LIMIT:.0f})")
    if gyro > GYRO_LIMIT:
        report.error(f"rotation out of physical range: max |r| = {gyro:.1f} deg/s (limit {GYRO_LIMIT:.0f})")


def validate_dataset(rides: pd.DataFrame, labels: pd.DataFrame | None = None) -> ValidationReport:
    """Checks on the raw GBG500-like dataframe (+ optional labels with RIDE_ID, LABEL)."""
    report = ValidationReport()
    missing_cols = [c for c in ["ride_id", "time_index", *SIGNAL_COLS] if c not in rides.columns]
    if missing_cols:
        report.error(f"missing columns: {missing_cols}")
        return report
    non_numeric = [c for c in ["time_index", *SIGNAL_COLS] if not pd.api.types.is_numeric_dtype(rides[c])]
    if non_numeric:
        report.error(f"non-numeric columns: {non_numeric}")
        return report

    missing_ratio = float(rides[SIGNAL_COLS].isna().any(axis=1).mean())
    if missing_ratio > MAX_MISSING_RATIO:
        report.error(f"{missing_ratio:.1%} rows with missing sensor values (limit {MAX_MISSING_RATIO:.0%})")

    steps = rides.sort_values(["ride_id", "time_index"]).groupby("ride_id").time_index.diff().dropna()
    regular = float((steps == SAMPLING_MS).mean()) if len(steps) else 0.0
    if regular < 0.99:
        report.error(f"only {regular:.1%} of time steps are {SAMPLING_MS} ms (expected 10 Hz sampling)")

    lengths = rides.groupby("ride_id").size()
    too_short = int((lengths < MIN_RIDE_SAMPLES).sum())
    if too_short:
        report.warnings.append(f"{too_short} rides shorter than {MIN_RIDE_SAMPLES} samples")
    _check_ranges(rides[SIGNAL_COLS], report)

    report.stats = {
        "n_rows": len(rides),
        "n_rides": int(lengths.size),
        "ride_length_min": int(lengths.min()),
        "ride_length_max": int(lengths.max()),
        "missing_row_ratio": round(missing_ratio, 4),
        "regular_step_ratio": round(regular, 4),
    }

    if labels is not None:
        unknown = sorted(set(labels.LABEL.str.strip().str.lower()) - {c.lower() for c in CLASSES})
        if unknown:
            report.error(f"unknown labels: {unknown}")
        if labels.RIDE_ID.duplicated().any():
            report.error(f"{int(labels.RIDE_ID.duplicated().sum())} duplicated ride ids in labels")
        unlabeled = set(lengths.index) - set(labels.RIDE_ID)
        orphan = set(labels.RIDE_ID) - set(lengths.index)
        if unlabeled:
            report.error(f"{len(unlabeled)} rides without label")
        if orphan:
            report.error(f"{len(orphan)} labels without ride")
        counts = labels.LABEL.value_counts()
        report.stats["label_counts"] = counts.to_dict()
        rare = counts[counts < 5]
        if len(rare):
            report.warnings.append(f"classes with < 5 rides: {rare.to_dict()}")
    return report


def validate_ride(ride: pd.DataFrame) -> pd.DataFrame:
    """Validate one ride before inference. Returns the ride (sorted) or raises DataValidationError."""
    missing = [c for c in SIGNAL_COLS if c not in ride.columns]
    if missing:
        raise DataValidationError(f"missing columns: {missing} (expected {SIGNAL_COLS})")
    if not (MIN_RIDE_SAMPLES <= len(ride) <= MAX_RIDE_SAMPLES):
        raise DataValidationError(
            f"ride has {len(ride)} samples, expected between {MIN_RIDE_SAMPLES} and {MAX_RIDE_SAMPLES} (10 Hz)"
        )
    try:
        values = ride[SIGNAL_COLS].astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"sensor columns must be numeric: {exc}") from exc
    if np.isinf(values.to_numpy()).any():
        raise DataValidationError("sensor values contain infinity")
    missing_ratio = float(values.isna().any(axis=1).mean())
    if missing_ratio > MAX_MISSING_RATIO_RIDE:
        raise DataValidationError(
            f"{missing_ratio:.0%} of rows have missing values (limit {MAX_MISSING_RATIO_RIDE:.0%})"
        )
    report = ValidationReport()
    _check_ranges(values, report)
    if not report.passed:
        raise DataValidationError("; ".join(report.errors))
    if "time_index" in ride.columns:
        ride = ride.sort_values("time_index")
    return ride
