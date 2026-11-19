"""Training pipeline: the steps the Airflow DAG (dags/ride_pipeline.py) orchestrates.

    validate + prepare data ─▶ train + register in MLflow ─▶ promote to @production ─▶ reload the API
                          └──▶ cross-validate (optional, logged to MLflow)

Each step is a plain function and a CLI sub-command, so the same code runs from Airflow, from a
GitHub Actions retraining job or by hand:

    export MLFLOW_TRACKING_URI=http://localhost:5050
    python -m trajclf.pipeline prepare
    python -m trajclf.pipeline train
    python -m trajclf.pipeline promote      # exit code 99 if production is unchanged (Airflow: "skipped")
    python -m trajclf.pipeline reload       # POST $API_URL/model/reload, skipped if API_URL is not set
    python -m trajclf.pipeline all          # the four steps in order

Every sub-command prints its result as one JSON line (Airflow keeps it as the task's XCom).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from trajclf.config import ARTIFACTS_DIR, PROCESSED_DIR, RAW_DIR, load_params

log = logging.getLogger(__name__)

MODEL_NAME = "ride-classifier"
TRAIN_EXPERIMENT = "ride-classification"
CV_EXPERIMENT = "ride-classification-cv"
NOT_PROMOTED_EXIT_CODE = 99  # BashOperator(skip_on_exit_code=99) marks the task "skipped", not "failed"


def _require_mlflow() -> None:
    if not os.getenv("MLFLOW_TRACKING_URI"):
        raise RuntimeError("MLFLOW_TRACKING_URI is not set: the pipeline registers models in MLflow")


def _train_kwargs(params: dict) -> dict:
    tp = params["train"]
    return {
        "epochs": tp["epochs"],
        "batch_size": tp["batch_size"],
        "lr": tp["lr"],
        "width": tp["width"],
        "window": params["features"]["window"],
        "stride": params["features"]["stride"],
        "seed": params["seed"],
        "noise": tp["noise"],
        "dropout": tp["dropout"],
    }


def prepare_data(raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR) -> dict:
    """Validate the raw data (stops the pipeline on failure), clean it and split it by ride."""
    from trajclf.data import prepare

    params = load_params()
    summary = prepare(Path(raw_dir), Path(processed_dir), params["data"]["test_size"], params["seed"])
    return {"n_rides": summary["n_rides"], "n_samples": summary["n_samples"]}


def train_and_register(processed_dir: Path = PROCESSED_DIR, artifacts_dir: Path = ARTIFACTS_DIR, **overrides) -> dict:
    """Train on the train split, evaluate on the test split, log the run and register a new version."""
    import mlflow
    from mlflow import MlflowClient

    from trajclf.train import log_to_mlflow, train

    _require_mlflow()
    kwargs = {**_train_kwargs(load_params()), **overrides}
    _, result = train(Path(processed_dir), Path(artifacts_dir), **kwargs)
    log_to_mlflow(Path(artifacts_dir), result, TRAIN_EXPERIMENT, MODEL_NAME)
    run_id = mlflow.last_active_run().info.run_id
    versions = MlflowClient().search_model_versions(f"name='{MODEL_NAME}' and run_id='{run_id}'")
    metrics = result["metrics"]
    return {
        "run_id": run_id,
        "version": str(versions[0].version) if versions else None,
        "test_f1_macro": round(float(metrics["test_f1_macro"]), 4),
        "test_balanced_accuracy": round(float(metrics["test_balanced_accuracy"]), 4),
    }


def cross_validate(processed_dir: Path = PROCESSED_DIR, run_name: str = "pipeline", **overrides) -> dict:
    """5-fold CV with the current params.yaml, logged as one run in the CV experiment."""
    from trajclf.evaluate import cross_validate as run_cv
    from trajclf.evaluate import log_cv_to_mlflow

    _require_mlflow()
    params = load_params()
    kwargs = {**_train_kwargs(params), **overrides}
    seed = kwargs.pop("seed")
    res = run_cv(Path(processed_dir), folds=params["evaluate"]["folds"], seed=seed, **kwargs)
    log_cv_to_mlflow(res, {**kwargs, "seed": seed}, run_name, CV_EXPERIMENT)
    return {m: round(float(res[m].mean()), 4) for m in ("test_balanced_accuracy", "test_f1_macro")}


def promote_model(metric: str = "f1_macro") -> dict:
    """Move @production to the newest version if it beats the baselines and the current production."""
    from mlflow import MlflowClient

    from trajclf.promote import promote

    _require_mlflow()
    client = MlflowClient()
    before = _production_version(client)
    _, message = promote(client, MODEL_NAME, metric=metric)
    after = _production_version(client)
    return {"promoted": after is not None and after != before, "production": after, "message": message}


def _production_version(client) -> str | None:
    from mlflow.exceptions import MlflowException

    try:
        return str(client.get_model_version_by_alias(MODEL_NAME, "production").version)
    except MlflowException:
        return None


def reload_api(api_url: str | None = None, token: str | None = None, timeout: float = 120) -> dict:
    """Ask the API to load the new @production version (POST /model/reload). No-op without API_URL."""
    api_url = (api_url if api_url is not None else os.getenv("API_URL", "")).rstrip("/")
    token = token if token is not None else os.getenv("RELOAD_TOKEN", "")
    if not api_url:
        return {"reloaded": False, "reason": "API_URL not set"}
    request = urllib.request.Request(f"{api_url}/model/reload", method="POST")
    if token:
        request.add_header("X-Reload-Token", token)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            served = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"API reload failed: HTTP {exc.code} {exc.read().decode(errors='replace')}") from exc
    return {"reloaded": True, "version": served.get("version"), "source": served.get("source")}


def run_all(raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR, artifacts_dir: Path = ARTIFACTS_DIR) -> dict:
    out = {"prepare": prepare_data(raw_dir, processed_dir)}
    out["train"] = train_and_register(processed_dir, artifacts_dir)
    out["promote"] = promote_model()
    out["reload"] = reload_api() if out["promote"]["promoted"] else {"reloaded": False, "reason": "not promoted"}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["prepare", "train", "cv", "promote", "reload", "all"])
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--epochs", type=int, default=None, help="override params.yaml (quick runs)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    overrides = {"epochs": args.epochs} if args.epochs else {}

    if args.step == "prepare":
        result = prepare_data(args.raw_dir, args.processed_dir)
    elif args.step == "train":
        result = train_and_register(args.processed_dir, args.artifacts_dir, **overrides)
    elif args.step == "cv":
        result = cross_validate(args.processed_dir, **overrides)
    elif args.step == "promote":
        result = promote_model()
    elif args.step == "reload":
        result = reload_api()
    else:
        result = run_all(args.raw_dir, args.processed_dir, args.artifacts_dir)

    print(json.dumps(result))  # last stdout line = Airflow XCom
    if args.step == "promote" and not result["promoted"]:
        log.info("Production unchanged: %s", result["message"])
        return NOT_PROMOTED_EXIT_CODE
    return 0


if __name__ == "__main__":
    sys.exit(main())
