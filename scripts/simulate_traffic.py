"""Send rides to the API to watch the monitoring live, with or without a simulated drift.

    python scripts/simulate_traffic.py                                  # dev API (Compose), normal traffic
    python scripts/simulate_traffic.py --api http://localhost:30800     # production API (Kubernetes)
    python scripts/simulate_traffic.py --drift units --duration 300     # 5 min of accelerometer data in g

Normal traffic replays the demo rides of examples/ with the class mix of the training data (83 % Safe,
8 % Safe overall, ...), so the Grafana dashboard looks like real traffic and no alert fires.
Each drift scenario makes one alert of monitoring/prometheus/alerts.yml fire after about 3 minutes:

    units      accelerometer divided by 9.81 (sent in g, not m/s²)    -> AccelerationDrift
    rotation   gyroscope x3 (much more aggressive riding)             -> RotationDrift
    missing    15 % of the rows empty (still accepted: limit 20 %)    -> MissingValuesDrift
    short      rides cut to 30 s                                      -> ShortRides
    invalid    1 ride in 2 cut to 2 s, rejected with HTTP 422         -> HighRejectionRate

Standard library only: runs with any Python 3.10+, without installing the project.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

SIGNAL_COLS = ["ax", "ay", "az", "rx", "ry", "rz"]
ACC_COLS, GYRO_COLS = SIGNAL_COLS[:3], SIGNAL_COLS[3:]
EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
# Rides per class in the 500 training rides (docs/data.md); examples/ holds one demo ride per class.
CLASS_MIX = {
    "ride_safe.csv": 416,
    "ride_safe_overall.csv": 38,
    "ride_bad_weather.csv": 22,
    "ride_tandem.csv": 18,
    "ride_reckless.csv": 6,
}
DRIFTS = ("none", "units", "rotation", "missing", "short", "invalid")
SHORT_RIDE = 300  # samples = 30 s at 10 Hz (still valid: the API accepts 50 to 36,000 samples)
INVALID_RIDE = 20  # samples = 2 s: rejected by the API (HTTP 422)

Ride = dict[str, list]  # one list per channel, None = missing sample (sent as JSON null)


def _number(value: str | None) -> float | None:
    value = (value or "").strip()
    return float(value) if value else None


def load_ride(path: Path) -> Ride:
    """Read a ride CSV (columns ax..rz, optional time_index) into one list per channel."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {c: [_number(row.get(c)) for row in rows] for c in SIGNAL_COLS}


def apply_drift(ride: Ride, drift: str, rng: random.Random) -> Ride:
    """Return a modified copy of the ride (the ride passed in is left unchanged)."""
    out = {c: list(values) for c, values in ride.items()}
    n = len(out["ax"])
    if drift == "units":
        for c in ACC_COLS:
            out[c] = [None if v is None else round(v / 9.81, 4) for v in out[c]]
    elif drift == "rotation":
        for c in GYRO_COLS:
            out[c] = [None if v is None else v * 3 for v in out[c]]
    elif drift == "missing":
        for i in rng.sample(range(n), k=int(0.15 * n)):
            for c in SIGNAL_COLS:
                out[c][i] = None
    elif drift == "short":
        out = {c: values[:SHORT_RIDE] for c, values in out.items()}
    elif drift == "invalid":
        if rng.random() < 0.5:
            out = {c: values[:INVALID_RIDE] for c, values in out.items()}
    elif drift != "none":
        raise ValueError(f"unknown drift {drift!r}, expected one of {DRIFTS}")
    return out


def send(api: str, ride: Ride, timeout: float = 60) -> tuple[int, dict]:
    """POST the ride to /predict. Returns (HTTP status, JSON body)."""
    request = urllib.request.Request(
        f"{api}/predict", data=json.dumps(ride).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, {"detail": exc.read().decode(errors="replace")}


def summary(sent: int, statuses: Counter, labels: Counter) -> str:
    codes = ", ".join(f"{code or 'unreachable'}: {n}" for code, n in sorted(statuses.items()))
    predicted = ", ".join(f"{label} {n}" for label, n in labels.most_common())
    return f"{sent} rides sent | HTTP {codes} | predicted: {predicted or '-'}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default="http://localhost:8000", help="API URL (production: http://localhost:30800)")
    parser.add_argument("--drift", choices=DRIFTS, default="none", help="simulated drift (default: none)")
    parser.add_argument("--rate", type=float, default=1.0, help="rides per second (default: 1)")
    parser.add_argument("--duration", type=float, default=600, help="seconds, 0 = until Ctrl+C (default: 600)")
    parser.add_argument("--rides-dir", type=Path, default=EXAMPLES_DIR, help="folder of ride CSV files")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.rate <= 0:
        parser.error("--rate must be positive")

    rides = {path.name: load_ride(path) for path in sorted(args.rides_dir.glob("*.csv"))}
    if not rides:
        parser.error(f"no CSV ride in {args.rides_dir}")
    names = list(rides)
    weights = [CLASS_MIX.get(name, 1) for name in names]
    rng = random.Random(args.seed)
    api = args.api.rstrip("/")

    print(f"Sending rides to {api}/predict: drift={args.drift}, {args.rate:g} ride/s. Ctrl+C to stop.")
    statuses: Counter = Counter()
    labels: Counter = Counter()
    sent, start = 0, time.monotonic()
    try:
        while not args.duration or time.monotonic() - start < args.duration:
            name = rng.choices(names, weights)[0]
            try:
                status, body = send(api, apply_drift(rides[name], args.drift, rng))
            except OSError as exc:  # API unreachable or timeout: count it and keep going
                status, body = 0, {"detail": str(exc)}
            statuses[status] += 1
            if status == 200:
                labels[body["label"]] += 1
            sent += 1
            if sent % 20 == 0:
                print(summary(sent, statuses, labels), flush=True)
            time.sleep(max(0.0, start + sent / args.rate - time.monotonic()))  # keep the requested rate
    except KeyboardInterrupt:
        pass
    print(summary(sent, statuses, labels))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
