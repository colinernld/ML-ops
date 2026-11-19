"""Ride classification API.

    # dev, with the MLflow stack running (docker compose up -d) and a model @production
    MLFLOW_TRACKING_URI=http://localhost:5050 uvicorn api.main:app --port 8000
    # or with a local model folder (python -m trajclf.train), no MLflow needed
    MODEL_DIR=artifacts uvicorn api.main:app --port 8000

Swagger UI: http://localhost:8000/docs   Prometheus metrics: http://localhost:8000/metrics
"""

# No `from __future__ import annotations` here: FastAPI must evaluate the Annotated dependencies
# defined inside create_app().
import io
import logging
import secrets
from contextlib import asynccontextmanager
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api import monitoring
from api.model_store import LoadedModel, ModelStore
from api.schemas import Error, Health, ModelInfoOut, Prediction, Ride
from api.settings import Settings
from trajclf.validation import DataValidationError

log = logging.getLogger("api")

DESCRIPTION = """
Classifies a bike/scooter ride from the phone's **accelerometer and gyroscope** (10 Hz) into
**Safe, Safe overall, Bad weather, Tandem, Reckless**.

Two ways to send a ride: **JSON** (`POST /predict`, one array per channel) or a **CSV file**
(`POST /predict/file`, columns `ax, ay, az, rx, ry, rz`, optional `time_index`; samples in `examples/`).
A ride must have between 50 samples (5 s) and 36,000 samples (1 h). Invalid rides get **HTTP 422**
with the reason.

The model is loaded from the MLflow Model Registry (`models:/ride-classifier@production` by default);
every response says which version answered.
"""

TAGS = [
    {"name": "prediction", "description": "Score a ride"},
    {"name": "model", "description": "Model currently served"},
    {"name": "ops", "description": "Health checks (Kubernetes probes)"},
]

ERRORS = {422: {"model": Error, "description": "Invalid ride"}, 503: {"model": Error, "description": "No model"}}


def create_app(settings: Settings | None = None, store: ModelStore | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = store or ModelStore(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if store.try_load():
            monitoring.set_model(store.model.info.source, store.model.info.version)
        else:
            monitoring.set_model(None, None)
            log.warning("Starting without a model, will retry: %s", store.last_error)
        yield

    app = FastAPI(
        title="Ride classification API",
        version="1.0.0",
        description=DESCRIPTION,
        openapi_tags=TAGS,
        lifespan=lifespan,
    )
    app.state.store = store
    app.add_middleware(monitoring.MetricsMiddleware)

    def current_model() -> LoadedModel:
        had_model = store.model is not None
        model = store.ensure_loaded()
        if model is None:
            monitoring.PREDICTION_ERRORS.labels(reason="model_unavailable").inc()
            raise HTTPException(503, f"No model loaded yet: {store.last_error}")
        if not had_model:
            monitoring.set_model(model.info.source, model.info.version)
        return model

    CurrentModel = Annotated[LoadedModel, Depends(current_model)]
    CsvFile = Annotated[UploadFile, File(description="CSV with columns ax, ay, az, rx, ry, rz (time_index optional)")]

    def score(ride: pd.DataFrame, model: LoadedModel) -> Prediction:
        try:
            out = model.clf.predict(ride)
        except DataValidationError as exc:
            monitoring.PREDICTION_ERRORS.labels(reason="validation").inc()
            raise HTTPException(422, f"Invalid ride: {exc}") from exc
        confidence = out["probabilities"][out["label"]]
        monitoring.observe_ride(ride)
        monitoring.observe_prediction(out["label"], confidence)
        log.info("prediction label=%s confidence=%.2f samples=%d", out["label"], confidence, len(ride))
        return Prediction(
            label=out["label"],
            probabilities=out["probabilities"],
            confidence=confidence,
            n_samples=len(ride),
            n_windows=out["n_windows"],
            model_version=model.info.version,
        )

    # ---------- prediction ----------
    # Plain `def` (not async): PyTorch inference is CPU-bound, FastAPI runs it in a worker thread.
    @app.post("/predict", response_model=Prediction, tags=["prediction"], responses=ERRORS)
    def predict(ride: Ride, model: CurrentModel) -> Prediction:
        """Score one ride sent as JSON (one array per channel, same length)."""
        return score(pd.DataFrame(ride.model_dump(exclude_none=True)), model)

    @app.post(
        "/predict/file",
        response_model=Prediction,
        tags=["prediction"],
        responses={**ERRORS, 413: {"model": Error, "description": "File too large"}},
    )
    def predict_file(file: CsvFile, model: CurrentModel) -> Prediction:
        """Score one ride uploaded as a CSV file (e.g. `examples/ride_safe.csv`)."""
        limit = int(settings.max_upload_mb * 1024 * 1024)
        content = file.file.read(limit + 1)
        if len(content) > limit:
            monitoring.PREDICTION_ERRORS.labels(reason="file_too_large").inc()
            raise HTTPException(413, f"File larger than {settings.max_upload_mb:g} MB")
        try:
            ride = pd.read_csv(io.BytesIO(content))
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
            monitoring.PREDICTION_ERRORS.labels(reason="bad_file").inc()
            raise HTTPException(422, f"Could not read the CSV file: {exc}") from exc
        ride.columns = [str(c).strip() for c in ride.columns]
        return score(ride, model)

    # ---------- model ----------
    @app.get("/model", response_model=ModelInfoOut, tags=["model"], responses={503: ERRORS[503]})
    def model_info(model: CurrentModel) -> ModelInfoOut:
        """Which model version is served, and what it expects."""
        cfg = model.clf.config
        return ModelInfoOut(
            **model.info.__dict__, classes=model.clf.classes, window=cfg["window"], stride=cfg["stride"]
        )

    @app.post("/model/reload", response_model=ModelInfoOut, tags=["model"], responses={503: ERRORS[503]})
    def reload_model(x_reload_token: Annotated[str | None, Header()] = None) -> ModelInfoOut:
        """Reload the model (e.g. after `python -m trajclf.promote` moved `@production`).

        If loading fails, the current model keeps serving. Requires the `X-Reload-Token` header when
        `RELOAD_TOKEN` is set.
        """
        if settings.reload_token and not secrets.compare_digest(x_reload_token or "", settings.reload_token):
            raise HTTPException(401, "Missing or wrong X-Reload-Token")
        try:
            loaded = store.reload()
        except Exception as exc:  # any loading error: keep serving the current model
            raise HTTPException(503, f"Reload failed, previous model kept: {store.last_error}") from exc
        monitoring.set_model(loaded.info.source, loaded.info.version)
        return model_info(loaded)

    # ---------- ops ----------
    @app.get("/health", response_model=Health, tags=["ops"])
    def health() -> Health:
        """Liveness: the process answers (even without a model)."""
        m = store.model
        return Health(status="ok", model_loaded=m is not None, model_version=m.info.version if m else None)

    @app.get("/ready", response_model=Health, tags=["ops"], responses={503: {"model": Health}})
    def ready():
        """Readiness: a model is loaded and the API can predict. Kubernetes only sends traffic when 200."""
        had_model = store.model is not None
        m = store.ensure_loaded()
        if m is None:
            body = Health(status="not ready", model_loaded=False, error=store.last_error)
            return JSONResponse(body.model_dump(), status_code=503)
        if not had_model:
            monitoring.set_model(m.info.source, m.info.version)
        return Health(status="ready", model_loaded=True, model_version=m.info.version)

    @app.get("/metrics", tags=["ops"], response_class=Response)
    def metrics() -> Response:
        """Prometheus metrics: requests, latency, predictions per class, confidence, input drift."""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
app = create_app()
