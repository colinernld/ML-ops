# API (FastAPI) + web app (Streamlit)

The API serves the model registered in MLflow; the web app is a thin client of the API to test it by hand.

```
Streamlit (webapp/app.py) ──HTTP──▶ FastAPI (src/api/) ──▶ MLflow registry  models:/ride-classifier@production
     :8501                            :8000                    └─▶ MinIO (model.pt, config)
                                        └─ /metrics ◀── Prometheus (monitoring part)
```

## Run it (dev)

```bash
pip install -e ".[mlflow,api,webapp,dev]"

# Option A: model from the MLflow registry (the dev stack must be up and a version must be @production)
export MLFLOW_TRACKING_URI=http://localhost:5050
uvicorn api.main:app --app-dir src --port 8000

# Option B: model from a local folder, no MLflow needed (after python -m trajclf.train)
MODEL_DIR=artifacts uvicorn api.main:app --app-dir src --port 8000

# Web app, in another terminal
API_URL=http://localhost:8000 streamlit run webapp/app.py
```

Swagger UI: <http://localhost:8000/docs> · web app: <http://localhost:8501>

## Endpoints

| Method | Path | Role | Errors |
|---|---|---|---|
| `POST` | `/predict` | Score one ride sent as JSON: one array per channel (`ax`…`rz`, optional `time_index`) | 422 invalid ride · 503 no model |
| `POST` | `/predict/file` | Score one ride uploaded as CSV (same columns, e.g. `examples/ride_safe.csv`) | 413 file too large · 422 · 503 |
| `GET` | `/model` | Model served: source, registry version and alias, run id, classes, window | 503 |
| `POST` | `/model/reload` | Reload after `python -m trajclf.promote`; the old model keeps serving if loading fails | 401 wrong token · 503 |
| `GET` | `/health` | Liveness probe: the process answers (even without a model) | – |
| `GET` | `/ready` | Readiness probe: 200 only when a model is loaded | 503 |
| `GET` | `/metrics` | Prometheus metrics | – |

Response of `/predict` and `/predict/file`:

```json
{"label": "Safe", "probabilities": {"Safe": 0.81, "Safe overall": 0.1, "...": 0.0},
 "confidence": 0.81, "n_samples": 2659, "n_windows": 20, "model_version": "3"}
```

A ride must contain 50 to 36,000 samples (5 s to 1 h at 10 Hz), numeric values within the physical range of
phone sensors, and at most 20 % of rows with missing values (`null` in JSON, empty in CSV). The rules come
from `trajclf.validation.validate_ride`, the same code used before training, and every rejection returns
HTTP 422 with the reason.

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `MLFLOW_TRACKING_URI` | – | MLflow server (`http://localhost:5050` laptop, `http://mlflow:5000` Compose, the Service name in Kubernetes) |
| `MODEL_URI` | `models:/ride-classifier@production` | Model to serve. An alias is resolved to a fixed version at load time. |
| `MODEL_DIR` | empty | Local folder with `model.pt` + `model_config.json`. Overrides `MODEL_URI`. |
| `RELOAD_RETRY_SECONDS` | `15` | If no model could be loaded (MLflow not up yet), retry at most this often. |
| `MAX_UPLOAD_MB` | `10` | Max CSV size for `/predict/file`. |
| `RELOAD_TOKEN` | empty | If set, `POST /model/reload` needs the header `X-Reload-Token`. Set it in production. |
| `API_URL` (web app) | `http://localhost:8000` | API the web app calls. |
| `EXAMPLES_DIR` (web app) | `examples/` | Sample rides offered in the web app. |

## Design choices

- **The version is resolved, never hard-coded.** `@production` is resolved to a version number, that exact
  version is loaded and its number comes back in every response, so a prediction can always be traced to a
  registry version and its training run.
- **The API starts even if MLflow is down.** `/health` stays green (the pod is not killed), `/ready` stays 503
  (no traffic is sent) and the model is loaded as soon as MLflow answers. This matters on Kubernetes,
  where the API pod can start before the MLflow pod.
- **Reloading does not drop requests.** The new model is loaded first and then swapped in; if loading fails,
  the current model keeps serving. Rolling back is `python -m trajclf.promote --version N` then a reload.
- **Validation is shared with training** (`validate_ride`), so the API rejects exactly what the pipeline
  would reject, with the same messages.
- **Inference endpoints are synchronous** (`def`, not `async def`): PyTorch is CPU-bound, so FastAPI runs it
  in its thread pool instead of blocking the event loop.
- **The web app holds no model and no logic.** It only calls the API and displays its answer, so it always
  shows what production answers, and its image does not need PyTorch.

## Metrics for monitoring (`GET /metrics`)

| Layer | Metric | Use |
|---|---|---|
| Service | `api_requests_total{method,path,status}`, `api_request_duration_seconds` | traffic, error rate, latency |
| Model | `model_loaded`, `model_info{source,version}` | which version serves, alert if 0 |
| Output | `predictions_total{label}`, `prediction_confidence` | output drift: share of each class (≈ 83 % Safe expected), falling confidence |
| Input | `ride_samples`, `ride_missing_ratio`, `ride_acc_magnitude_mean`, `ride_rot_magnitude_mean` | input drift (suggested in the [model card](model_card.md)) |
| Errors | `prediction_errors_total{reason}` | `validation`, `bad_file`, `file_too_large`, `model_unavailable` |

## Tests

`tests/test_api.py` (17 tests) runs the real app with FastAPI's `TestClient`: predictions in JSON and CSV,
every rejection case, upload limit, metrics, Swagger, startup without MLflow then recovery, reload token, and
loading `models:/ride-classifier@production` from a temporary MLflow registry, including a promotion followed
by a reload.

```bash
pytest tests/test_api.py
```

## For the Docker / Kubernetes part

- API image: `pip install -e ".[mlflow,api]"` with the **CPU-only PyTorch wheel**
  (`--index-url https://download.pytorch.org/whl/cpu`), otherwise the image carries CUDA (~2 GB more).
  Command: `uvicorn api.main:app --app-dir src --host 0.0.0.0 --port 8000`.
- Web app image: only `streamlit`, `requests`, `pandas` + `webapp/` and `examples/`; no PyTorch.
  Command: `streamlit run webapp/app.py --server.address 0.0.0.0 --server.port 8501`.
- Probes: `livenessProbe` → `GET /health`, `readinessProbe` → `GET /ready` (allow ~30 s for the first load).
- Env: `MLFLOW_TRACKING_URI=http://<mlflow service>:5000` (add that name to `MLFLOW_ALLOWED_HOSTS`),
  `RELOAD_TOKEN` from a Secret.
- Prometheus: scrape `<api service>:8000/metrics`.
