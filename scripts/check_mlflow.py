"""Check that the MLflow server and its MinIO artifact store work end to end.

    MLFLOW_TRACKING_URI=http://localhost:5050 python scripts/check_mlflow.py

Logs a small run (one metric + one file) in the "smoke-test" experiment, downloads the file back
through the server, compares it, then deletes the run. Also lists the registered models.
Exit code 0 means the stack is usable for training and for the API.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import mlflow
from mlflow import MlflowClient


def main() -> int:
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        print("Set MLFLOW_TRACKING_URI first, e.g. export MLFLOW_TRACKING_URI=http://localhost:5050")
        return 1

    content = "ride-mlops smoke test\n"
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "smoke.txt"
        src.write_text(content)

        mlflow.set_experiment("smoke-test")
        with mlflow.start_run(run_name="smoke-test") as run:
            mlflow.log_metric("ok", 1.0)
            mlflow.log_artifact(str(src))

        back = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id, artifact_path="smoke.txt", dst_path=str(Path(tmp) / "download")
        )
        if Path(back).read_text() != content:
            print("FAIL: the downloaded artifact differs from the uploaded one")
            return 1

    client = MlflowClient()
    client.delete_run(run.info.run_id)
    print(f"OK  tracking + artifact upload/download work ({uri})")

    models = client.search_registered_models()
    if not models:
        print("    no registered model yet: run  python -m trajclf.train --register-name ride-classifier")
    for m in models:
        versions = sorted(int(v.version) for v in client.search_model_versions(f"name='{m.name}'"))
        aliases = ", ".join(f"@{a}->v{v}" for a, v in (m.aliases or {}).items()) or "no alias"
        print(f"    model {m.name}: versions {versions}, {aliases}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
