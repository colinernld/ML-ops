"""Load the classifier (MLflow registry or local folder) and keep it in memory, thread-safe.

The API never hard-codes a model version: by default it resolves `models:/ride-classifier@production`
to a version number, loads that exact version and reports it in every response. After
`python -m trajclf.promote`, call POST /model/reload (or restart the pod) to serve the new version.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from api.settings import Settings
from trajclf.predict import RideClassifier

log = logging.getLogger(__name__)

_ALIAS_URI = re.compile(r"^models:/(?P<name>[^/@]+)@(?P<alias>[^/@]+)$")
_VERSION_URI = re.compile(r"^models:/(?P<name>[^/@]+)/(?P<version>\d+)$")


@dataclass(frozen=True)
class ModelInfo:
    source: str  # "mlflow" or "local"
    uri: str
    name: str | None = None
    version: str | None = None
    alias: str | None = None
    run_id: str | None = None
    loaded_at: str = ""


@dataclass(frozen=True)
class LoadedModel:
    clf: RideClassifier
    info: ModelInfo


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_from_dir(model_dir: str) -> LoadedModel:
    clf = RideClassifier.load(model_dir)
    return LoadedModel(clf, ModelInfo(source="local", uri=model_dir, loaded_at=_now()))


def load_from_mlflow(model_uri: str) -> LoadedModel:
    """Resolve an alias to a fixed version first, so the version we report is the one we loaded."""
    import mlflow
    from mlflow import MlflowClient

    name = alias = version = run_id = None
    resolved = model_uri
    if m := _ALIAS_URI.match(model_uri):
        name, alias = m["name"], m["alias"]
        mv = MlflowClient().get_model_version_by_alias(name, alias)
        version, run_id = str(mv.version), mv.run_id
        resolved = f"models:/{name}/{version}"
    elif m := _VERSION_URI.match(model_uri):
        name, version = m["name"], m["version"]
        run_id = MlflowClient().get_model_version(name, version).run_id

    clf = mlflow.pyfunc.load_model(resolved).unwrap_python_model().clf
    info = ModelInfo(
        source="mlflow", uri=model_uri, name=name, version=version, alias=alias, run_id=run_id, loaded_at=_now()
    )
    return LoadedModel(clf, info)


class ModelStore:
    """Holds the current model. Loading failures are kept (not raised) so the API can start before MLflow."""

    def __init__(self, settings: Settings, loader=None):
        self.settings = settings
        self._loader = loader or self._default_loader
        self._lock = threading.Lock()
        self._model: LoadedModel | None = None
        self._last_attempt = 0.0
        self.last_error: str | None = None

    def _default_loader(self) -> LoadedModel:
        if self.settings.model_dir:
            return load_from_dir(self.settings.model_dir)
        return load_from_mlflow(self.settings.model_uri)

    @property
    def model(self) -> LoadedModel | None:
        return self._model

    def reload(self) -> LoadedModel:
        """Load (again) and swap atomically. Raises if loading fails; the previous model stays in service."""
        with self._lock:
            self._last_attempt = time.monotonic()
            try:
                loaded = self._loader()
            except Exception as exc:  # any loading problem is reported, never crashes the API
                self.last_error = f"{type(exc).__name__}: {exc}"
                log.error("Model loading failed: %s", self.last_error)
                raise
            self._model, self.last_error = loaded, None
            log.info("Model loaded: %s", loaded.info)
            return loaded

    def try_load(self) -> bool:
        try:
            self.reload()
            return True
        except Exception:  # error kept in last_error, shown by /ready
            log.debug("Model not loaded", exc_info=True)
            return False

    def ensure_loaded(self) -> LoadedModel | None:
        """Return the model; if none is loaded yet, retry loading at most every `reload_retry_seconds`."""
        if self._model is None and time.monotonic() - self._last_attempt >= self.settings.reload_retry_seconds:
            self.try_load()
        return self._model
