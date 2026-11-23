"""Traffic simulator (scripts/simulate_traffic.py): each drift scenario changes what it claims, and the API's
validation rules accept or reject the result as the matching alert expects (monitoring/prometheus/alerts.yml)."""

import importlib.util
import random
from pathlib import Path

import pandas as pd
import pytest

from trajclf.validation import DataValidationError, validate_ride

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("simulate_traffic", ROOT / "scripts" / "simulate_traffic.py")
sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sim)


@pytest.fixture(scope="module")
def ride():
    return sim.load_ride(ROOT / "examples" / "ride_safe.csv")


def mean_norm(ride, cols) -> float:
    values = pd.DataFrame(ride)[cols].dropna()
    return float(((values**2).sum(axis=1) ** 0.5).mean())


def drifted(ride, drift, seed=0):
    return sim.apply_drift(ride, drift, random.Random(seed))


def test_load_ride(ride):
    assert list(ride) == sim.SIGNAL_COLS
    assert len(ride["ax"]) == len(pd.read_csv(ROOT / "examples" / "ride_safe.csv"))
    validate_ride(pd.DataFrame(ride))


def test_class_mix_covers_the_demo_rides():
    assert set(sim.CLASS_MIX) == {path.name for path in (ROOT / "examples").glob("*.csv")}
    assert sum(sim.CLASS_MIX.values()) == 500  # GBG500


def test_no_drift_sends_the_ride_unchanged(ride):
    assert drifted(ride, "none") == ride


def test_units_drift_sends_acceleration_in_g(ride):
    out = drifted(ride, "units")
    assert mean_norm(ride, sim.ACC_COLS) == pytest.approx(9.5, abs=0.5)  # like the training rides
    assert mean_norm(out, sim.ACC_COLS) < 8  # AccelerationDrift: outside 8 - 11 m/s²
    assert out["rx"] == ride["rx"]
    validate_ride(pd.DataFrame(out))  # still accepted: only the monitoring can see it


def test_rotation_drift_triples_the_gyroscope(ride):
    out = drifted(ride, "rotation")
    assert mean_norm(out, sim.GYRO_COLS) == pytest.approx(3 * mean_norm(ride, sim.GYRO_COLS), rel=1e-6)
    assert mean_norm(out, sim.GYRO_COLS) > 35  # RotationDrift
    assert out["ax"] == ride["ax"]
    validate_ride(pd.DataFrame(out))


def test_missing_drift_stays_under_the_rejection_limit(ride):
    missing = pd.DataFrame(drifted(ride, "missing")).isna().any(axis=1).mean()
    assert 0.05 < missing <= 0.2  # MissingValuesDrift, but the API still accepts the ride
    validate_ride(pd.DataFrame(drifted(ride, "missing")))


def test_short_drift(ride):
    out = drifted(ride, "short")
    assert len(out["ax"]) == sim.SHORT_RIDE < 600  # ShortRides: under 1 minute
    validate_ride(pd.DataFrame(out))


def test_invalid_drift_is_rejected_about_half_of_the_time(ride):
    rng = random.Random(0)
    rejected = 0
    for _ in range(40):
        try:
            validate_ride(pd.DataFrame(sim.apply_drift(ride, "invalid", rng)))
        except DataValidationError:
            rejected += 1
    assert 10 <= rejected <= 30  # HighRejectionRate: above 20 %


def test_drifts_do_not_modify_the_ride(ride):
    before = {c: list(values) for c, values in ride.items()}
    for drift in sim.DRIFTS:
        drifted(ride, drift)
    assert ride == before


def test_unknown_drift(ride):
    with pytest.raises(ValueError, match="unknown drift"):
        drifted(ride, "tsunami")
