"""Prometheus metrics exposed on GET /metrics (scraped by the monitoring part).

Three layers, following the course's monitoring session:
  - service:  request count, latency and status per endpoint
  - input:    ride length, missing values, mean |a| and |r|  -> input drift (see docs/model_card.md)
  - output:   predicted class counts and confidence          -> output drift / model health
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from prometheus_client import Counter, Gauge, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# ---------- service ----------
REQUESTS = Counter("api_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram(
    "api_request_duration_seconds",
    "HTTP request latency",
    ["method", "path"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)

# ---------- model ----------
MODEL_LOADED = Gauge("model_loaded", "1 if a model is loaded and serving")
MODEL_INFO = Gauge("model_info", "Model currently served (value is always 1)", ["source", "version"])

# ---------- output ----------
PREDICTIONS = Counter("predictions_total", "Predictions by predicted class", ["label"])
PREDICTION_ERRORS = Counter("prediction_errors_total", "Rejected prediction requests", ["reason"])
CONFIDENCE = Histogram(
    "prediction_confidence", "Probability of the predicted class", buckets=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
)

# ---------- input ----------
RIDE_SAMPLES = Histogram(
    "ride_samples", "Samples per ride (10 Hz)", buckets=(50, 300, 600, 1200, 3000, 6000, 12000, 36000)
)
RIDE_MISSING = Histogram("ride_missing_ratio", "Share of rows with a missing value", buckets=(0, 0.01, 0.05, 0.1, 0.2))
RIDE_ACC = Histogram(
    "ride_acc_magnitude_mean", "Mean |a| over the ride (m/s²)", buckets=(5, 8, 9, 9.5, 10, 10.5, 11, 13, 20)
)
RIDE_ROT = Histogram(
    "ride_rot_magnitude_mean", "Mean |r| over the ride (deg/s)", buckets=(1, 2, 5, 10, 20, 40, 80, 160)
)


def set_model(source: str | None, version: str | None) -> None:
    MODEL_INFO.clear()
    if source is None:
        MODEL_LOADED.set(0)
        return
    MODEL_LOADED.set(1)
    MODEL_INFO.labels(source=source, version=version or "local").set(1)


def observe_ride(ride: pd.DataFrame) -> None:
    values = ride[["ax", "ay", "az", "rx", "ry", "rz"]].astype(float)
    RIDE_SAMPLES.observe(len(values))
    RIDE_MISSING.observe(float(values.isna().any(axis=1).mean()))
    acc = np.sqrt((values[["ax", "ay", "az"]] ** 2).sum(axis=1, min_count=3))
    rot = np.sqrt((values[["rx", "ry", "rz"]] ** 2).sum(axis=1, min_count=3))
    if acc.notna().any():
        RIDE_ACC.observe(float(acc.mean()))
    if rot.notna().any():
        RIDE_ROT.observe(float(rot.mean()))


def observe_prediction(label: str, confidence: float) -> None:
    PREDICTIONS.labels(label=label).inc()
    CONFIDENCE.observe(confidence)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Count and time every request, labelled by route template (/predict, not the raw URL)."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            path = getattr(route, "path", "unmatched")
            if path != "/metrics":
                LATENCY.labels(request.method, path).observe(time.perf_counter() - start)
                REQUESTS.labels(request.method, path, str(status)).inc()
