"""Stratified k-fold cross-validation at ride level.

With only 500 rides (6 Reckless), a single 100-ride test set is noisy: one ride flips the
Reckless recall from 0 to 100 %. Cross-validation gives a mean +/- std for the report.

Usage:
    python -m trajclf.evaluate                     # hyper-parameters from params.yaml
    python -m trajclf.evaluate --epochs 25 --run-name more-epochs
With MLFLOW_TRACKING_URI set, each evaluation is logged as one MLflow run (mean/std metrics)
so that experiments can be compared side by side in the MLflow UI.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from trajclf.config import ARTIFACTS_DIR, PROCESSED_DIR, SEED, load_params
from trajclf.data import load_processed
from trajclf.train import fit_evaluate

log = logging.getLogger(__name__)


def cross_validate(data_dir: Path = PROCESSED_DIR, folds: int = 5, seed: int = SEED, **train_kwargs) -> pd.DataFrame:
    rides, split = load_processed(data_dir)
    labels = split[["ride_id", "label"]].reset_index(drop=True)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows = []
    for k, (_, test_idx) in enumerate(skf.split(labels.ride_id, labels.label), start=1):
        fold = labels.assign(split="train")
        fold.loc[test_idx, "split"] = "test"
        _, result = fit_evaluate(rides, fold, out_dir=None, seed=seed, **train_kwargs)
        rows.append({"fold": k, **result["metrics"]})
        log.info("fold %d: %s", k, {m: round(v, 3) for m, v in result["metrics"].items() if m.startswith("test_")})
    return pd.DataFrame(rows).set_index("fold")


def log_cv_to_mlflow(res: pd.DataFrame, params: dict, run_name: str | None, experiment: str) -> None:
    import mlflow

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name, tags={"type": "cross-validation"}):
        mlflow.log_params(params)
        summary = res.agg(["mean", "std"])
        for metric in res.columns:
            mlflow.log_metric(f"cv_{metric}_mean", float(summary.loc["mean", metric]))
            mlflow.log_metric(f"cv_{metric}_std", float(summary.loc["std", metric]))
        for fold, row in res.iterrows():
            mlflow.log_metric("fold_test_balanced_accuracy", float(row["test_balanced_accuracy"]), step=int(fold))
            mlflow.log_metric("fold_test_f1_macro", float(row["test_f1_macro"]), step=int(fold))


def main() -> None:
    params = load_params()
    tp = params["train"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--folds", type=int, default=params["evaluate"]["folds"])
    parser.add_argument("--epochs", type=int, default=tp["epochs"])
    parser.add_argument("--lr", type=float, default=tp["lr"])
    parser.add_argument("--width", type=int, default=tp["width"])
    parser.add_argument("--window", type=int, default=params["features"]["window"])
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--experiment", default="ride-classification-cv")
    parser.add_argument("--out", type=Path, default=ARTIFACTS_DIR / "cv_results.csv")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run_params = {
        "folds": args.folds,
        "epochs": args.epochs,
        "lr": args.lr,
        "width": args.width,
        "window": args.window,
        "stride": args.window // 2,
        "batch_size": tp["batch_size"],
        "dropout": tp["dropout"],
        "noise": tp["noise"],
    }
    res = cross_validate(
        args.data_dir,
        args.folds,
        seed=params["seed"],
        epochs=args.epochs,
        lr=args.lr,
        width=args.width,
        window=args.window,
        stride=args.window // 2,
        batch_size=tp["batch_size"],
        dropout=tp["dropout"],
        noise=tp["noise"],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out)
    summary = res.agg(["mean", "std"]).T.round(3)
    print(summary.to_string())
    (args.out.with_suffix(".json")).write_text(json.dumps(summary.to_dict(), indent=2))
    if os.getenv("MLFLOW_TRACKING_URI"):
        log_cv_to_mlflow(res, run_params, args.run_name, args.experiment)


if __name__ == "__main__":
    main()
