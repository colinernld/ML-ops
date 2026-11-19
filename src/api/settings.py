"""API configuration, read from environment variables only (same image for dev, Compose and Kubernetes)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Where the model comes from. MODEL_DIR (a local folder with model.pt + model_config.json) wins
    # over MODEL_URI, which is useful for dev and tests without an MLflow server.
    model_uri: str = "models:/ride-classifier@production"
    model_dir: str = ""
    # If the model cannot be loaded at startup (MLflow not up yet), retry at most this often.
    reload_retry_seconds: float = 15.0
    max_upload_mb: float = 10.0
    # Optional shared secret for POST /model/reload. Empty = endpoint open (dev only).
    reload_token: str = ""

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            model_uri=os.getenv("MODEL_URI", cls.model_uri),
            model_dir=os.getenv("MODEL_DIR", cls.model_dir),
            reload_retry_seconds=float(os.getenv("RELOAD_RETRY_SECONDS", cls.reload_retry_seconds)),
            max_upload_mb=float(os.getenv("MAX_UPLOAD_MB", cls.max_upload_mb)),
            reload_token=os.getenv("RELOAD_TOKEN", cls.reload_token),
        )
