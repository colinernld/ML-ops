"""Training pipeline steps (trajclf.pipeline) end to end on synthetic rides, with a local MLflow store.

The Airflow DAG only calls these steps; its structure is tested in test_airflow_dag.py when Airflow is installed.
"""

import json

import numpy as np
import pandas as pd
import pytest

mlflow = pytest.importorskip("mlflow")  # optional dependency: pip install -e ".[mlflow]"

from trajclf import pipeline
from trajclf.config import CLASSES, SIGNAL_COLS
from trajclf.data import LABELS_FILE, RIDES_FILE

FAST = {"epochs": 2, "width": 8, "window": 64, "stride": 32}


@pytest.fixture
def raw_dir(tmp_path):
    rng = np.random.default_rng(0)
    frames, labels = [], []
    for i in range(50):
        length = int(rng.integers(300, 600))
        f = pd.DataFrame(rng.normal(scale=1 + (i % 5), size=(length, 6)), columns=SIGNAL_COLS)
        f.insert(0, "time_index", np.arange(length) * 100)
        f.insert(0, "ride_id", f"ride{i:02d}")
        frames.append(f)
        labels.append({"RIDE_ID": f"ride{i:02d}", "LABEL": CLASSES[i % 5]})
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.concat(frames).to_parquet(raw / RIDES_FILE)
    pd.DataFrame(labels).to_csv(raw / LABELS_FILE, index=False)
    return raw


@pytest.fixture
def tracking(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    return uri


def test_steps_end_to_end(raw_dir, tmp_path, tracking, monkeypatch):
    processed, artifacts = tmp_path / "processed", tmp_path / "artifacts"

    assert pipeline.prepare_data(raw_dir, processed)["n_rides"] == 50

    first = pipeline.train_and_register(processed, artifacts, **FAST)
    assert first["version"] == "1" and first["run_id"]
    assert "test_f1_macro" in mlflow.MlflowClient().get_run(first["run_id"]).data.metrics

    promoted = pipeline.promote_model()
    # Whatever the decision, the result must match the registry
    assert promoted["promoted"] == (promoted["production"] == "1")

    second = pipeline.train_and_register(processed, artifacts, **FAST)
    assert second["version"] == "2"

    calls = []

    class FakeResponse:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return self.body

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, request.get_method(), request.get_header("X-reload-token")))
        return FakeResponse(json.dumps({"version": "2", "source": "mlflow"}).encode())

    monkeypatch.setattr(pipeline.urllib.request, "urlopen", fake_urlopen)
    assert pipeline.reload_api("http://api:8000/", "s3cret") == {"reloaded": True, "version": "2", "source": "mlflow"}
    assert calls == [("http://api:8000/model/reload", "POST", "s3cret")]


def test_prepare_stops_on_invalid_data(raw_dir, tmp_path):
    labels = pd.read_csv(raw_dir / LABELS_FILE)
    labels.loc[0, "LABEL"] = "Crash"
    labels.to_csv(raw_dir / LABELS_FILE, index=False)
    with pytest.raises(ValueError, match="unknown labels"):
        pipeline.prepare_data(raw_dir, tmp_path / "processed")


def test_reload_is_skipped_without_api_url(monkeypatch):
    monkeypatch.delenv("API_URL", raising=False)
    assert pipeline.reload_api() == {"reloaded": False, "reason": "API_URL not set"}


def test_train_requires_mlflow(monkeypatch, tmp_path):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(RuntimeError, match="MLFLOW_TRACKING_URI"):
        pipeline.train_and_register(tmp_path, tmp_path)


def test_promote_exit_code_marks_task_skipped(monkeypatch, capsys):
    monkeypatch.setattr(
        pipeline, "promote_model", lambda: {"promoted": False, "production": "1", "message": "worse than production"}
    )
    assert pipeline.main(["promote"]) == pipeline.NOT_PROMOTED_EXIT_CODE
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["promoted"] is False

    monkeypatch.setattr(pipeline, "promote_model", lambda: {"promoted": True, "production": "2", "message": "ok"})
    assert pipeline.main(["promote"]) == 0
