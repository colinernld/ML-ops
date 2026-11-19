"""API integration tests: real FastAPI app + a small untrained model saved to disk (no MLflow server)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("fastapi")  # optional dependency: pip install -e ".[api]"
from fastapi.testclient import TestClient

from api.main import create_app
from api.model_store import ModelStore
from api.settings import Settings
from trajclf.config import CLASSES, SIGNAL_COLS
from trajclf.features import N_FEATURES, Normalizer
from trajclf.model import RideCNN
from trajclf.predict import RideClassifier

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    torch.manual_seed(0)
    config = {
        "classes": CLASSES,
        "signal_cols": SIGNAL_COLS,
        "window": 64,
        "stride": 32,
        "width": 4,
        "n_features": N_FEATURES,
        "normalizer": Normalizer(mean=[0.0] * N_FEATURES, std=[1.0] * N_FEATURES).to_dict(),
    }
    out = tmp_path_factory.mktemp("model")
    RideClassifier(RideCNN(N_FEATURES, len(CLASSES), width=4), config).save(out)
    return out


@pytest.fixture
def client(model_dir):
    with TestClient(create_app(Settings(model_dir=str(model_dir)))) as c:  # `with` runs the startup
        yield c


def ride_json(n=120, seed=0):
    rng = np.random.default_rng(seed)
    return {c: rng.normal(size=n).round(4).tolist() for c in SIGNAL_COLS}


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True, "model_version": None, "error": None}
    assert client.get("/ready").status_code == 200


def test_predict_json(client):
    r = client.post("/predict", json=ride_json())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["label"] in CLASSES
    assert set(body["probabilities"]) == set(CLASSES)
    assert sum(body["probabilities"].values()) == pytest.approx(1, abs=1e-3)
    assert body["confidence"] == body["probabilities"][body["label"]]
    assert body["n_samples"] == 120 and body["n_windows"] >= 1


def test_predict_accepts_missing_values_and_timestamps(client):
    ride = ride_json()
    ride["ax"][3] = None
    ride["time_index"] = list(range(0, 12000, 100))
    assert client.post("/predict", json=ride).status_code == 200


@pytest.mark.parametrize(
    "change,message",
    [
        (lambda r: r.update(ax=r["ax"][:10]), "same length"),  # caught by the request schema
        (lambda r: [r.update({c: v[:20]}) for c, v in list(r.items())], "between 50"),  # too short
        (lambda r: r.update(ax=[1e6] * 120), "physical range"),  # sensor values impossible
        (lambda r: r.update(ay=[None] * 120), "missing values"),
    ],
)
def test_predict_rejects_invalid_rides(client, change, message):
    ride = ride_json()
    change(ride)
    r = client.post("/predict", json=ride)
    assert r.status_code == 422
    assert message in r.text


def test_predict_file_with_example_csv(client):
    with open(EXAMPLES / "ride_safe.csv", "rb") as f:
        r = client.post("/predict/file", files={"file": ("ride_safe.csv", f, "text/csv")})
    assert r.status_code == 200, r.text
    assert r.json()["n_samples"] == len(pd.read_csv(EXAMPLES / "ride_safe.csv"))


def test_predict_file_rejects_bad_files(client):
    r = client.post("/predict/file", files={"file": ("x.csv", b"foo,bar\n1,2\n", "text/csv")})
    assert r.status_code == 422 and "missing columns" in r.text
    r = client.post("/predict/file", files={"file": ("x.csv", b"", "text/csv")})
    assert r.status_code == 422


def test_predict_file_size_limit(model_dir):
    app = create_app(Settings(model_dir=str(model_dir), max_upload_mb=0.001))
    with TestClient(app) as c, open(EXAMPLES / "ride_safe.csv", "rb") as f:
        assert c.post("/predict/file", files={"file": ("r.csv", f, "text/csv")}).status_code == 413


def test_model_info(client):
    body = client.get("/model").json()
    assert body["source"] == "local" and body["classes"] == CLASSES and body["window"] == 64


def test_metrics_exposed(client):
    client.post("/predict", json=ride_json())
    text = client.get("/metrics").text
    assert "predictions_total" in text
    assert 'api_requests_total{method="POST",path="/predict"' in text
    assert "ride_acc_magnitude_mean" in text


def test_starts_without_model_then_recovers(model_dir):
    """MLflow down at startup: API is alive but not ready, then serves once the model can be loaded."""
    calls = {"n": 0}

    def flaky_loader():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("MLflow unreachable")
        from api.model_store import load_from_dir

        return load_from_dir(str(model_dir))

    settings = Settings(reload_retry_seconds=0)
    with TestClient(create_app(settings, ModelStore(settings, loader=flaky_loader))) as c:
        assert c.get("/health").json()["model_loaded"] is False
        assert c.get("/ready").status_code == 200  # retried and loaded
        assert c.post("/predict", json=ride_json()).status_code == 200


def test_unavailable_model_returns_503(model_dir):
    settings = Settings(reload_retry_seconds=3600)

    def broken():
        raise ConnectionError("MLflow unreachable")

    with TestClient(create_app(settings, ModelStore(settings, loader=broken))) as c:
        assert c.get("/ready").status_code == 503
        r = c.post("/predict", json=ride_json())
        assert r.status_code == 503 and "MLflow unreachable" in r.text


def test_reload_requires_token(model_dir):
    with TestClient(create_app(Settings(model_dir=str(model_dir), reload_token="s3cret"))) as c:
        assert c.post("/model/reload").status_code == 401
        assert c.post("/model/reload", headers={"X-Reload-Token": "s3cret"}).status_code == 200


def test_swagger_available(client):
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/predict", "/predict/file", "/model", "/health", "/ready"} <= set(paths)


def test_loads_production_alias_from_mlflow(model_dir, tmp_path, monkeypatch):
    """Production path: resolve models:/ride-classifier@production to a version and serve it."""
    mlflow = pytest.importorskip("mlflow")
    from trajclf.mlflow_model import RidePyfunc

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("api-test")
    for _ in range(2):  # two versions, production points at v1 to check the alias is really followed
        with mlflow.start_run():
            mlflow.pyfunc.log_model(
                name="model",
                python_model=RidePyfunc(),
                artifacts={"model_dir": str(model_dir)},
                registered_model_name="ride-classifier",
            )
    mlflow.MlflowClient().set_registered_model_alias("ride-classifier", "production", "1")

    with TestClient(create_app(Settings(model_uri="models:/ride-classifier@production"))) as c:
        info = c.get("/model").json()
        assert (info["source"], info["version"], info["alias"]) == ("mlflow", "1", "production")
        assert c.post("/predict", json=ride_json()).json()["model_version"] == "1"

        mlflow.MlflowClient().set_registered_model_alias("ride-classifier", "production", "2")  # promote
        assert c.post("/model/reload").json()["version"] == "2"
        assert c.post("/predict", json=ride_json()).json()["model_version"] == "2"
