"""Train the ride classifier and evaluate it against naive baselines.

Usage:
    python -m trajclf.train                       # local run, artifacts in ./artifacts
    MLFLOW_TRACKING_URI=http://localhost:5000 python -m trajclf.train --register-name ride-classifier
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from trajclf.config import (
    ARTIFACTS_DIR,
    CLASS_TO_IDX,
    CLASSES,
    PROCESSED_DIR,
    SEED,
    SIGNAL_COLS,
    WINDOW_SIZE,
    WINDOW_STRIDE,
)
from trajclf.data import load_processed
from trajclf.features import N_FEATURES, Normalizer, make_windows, preprocess
from trajclf.model import RideCNN
from trajclf.predict import RideClassifier

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_windows(rides, ride_ids, labels, normalizer, window, stride):
    xs, ys = [], []
    for rid, lab in zip(ride_ids, labels):
        w = make_windows(normalizer.transform(preprocess(rides[rid])), window, stride)
        xs.append(w)
        ys.append(np.full(len(w), CLASS_TO_IDX[lab]))
    return np.concatenate(xs), np.concatenate(ys)


def compute_metrics(y_true, y_pred, prefix: str = "") -> dict:
    reckless = CLASS_TO_IDX["Reckless"]
    return {
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        f"{prefix}f1_macro": f1_score(y_true, y_pred, average="macro", labels=range(len(CLASSES)), zero_division=0),
        f"{prefix}reckless_recall": float(np.mean(np.asarray(y_pred)[np.asarray(y_true) == reckless] == reckless))
        if reckless in y_true
        else float("nan"),
    }


def baselines(y_train, y_test, seed: int) -> dict:
    out = {}
    for strategy in ["most_frequent", "stratified"]:
        dummy = DummyClassifier(strategy=strategy, random_state=seed).fit(np.zeros((len(y_train), 1)), y_train)
        out.update(compute_metrics(y_test, dummy.predict(np.zeros((len(y_test), 1))), prefix=f"baseline_{strategy}_"))
    return out


def train(
    data_dir: Path = PROCESSED_DIR,
    out_dir: Path = ARTIFACTS_DIR,
    epochs: int = 12,
    batch_size: int = 64,
    lr: float = 1e-3,
    width: int = 16,
    window: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
    seed: int = SEED,
    noise: float = 0.05,
    dropout: float = 0.5,
) -> tuple[RideClassifier, dict]:
    rides, split = load_processed(data_dir)
    return fit_evaluate(rides, split, out_dir, epochs, batch_size, lr, width, window, stride, seed, noise, dropout)


def fit_evaluate(
    rides: dict[str, np.ndarray],
    split: pd.DataFrame,
    out_dir: Path | None = ARTIFACTS_DIR,
    epochs: int = 12,
    batch_size: int = 64,
    lr: float = 1e-3,
    width: int = 16,
    window: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
    seed: int = SEED,
    noise: float = 0.05,
    dropout: float = 0.5,
) -> tuple[RideClassifier, dict]:
    """Train on split=='train' rides, evaluate on split=='test' rides. Saves to out_dir if given."""
    set_seed(seed)
    tr, te = split[split.split == "train"], split[split.split == "test"]

    normalizer = Normalizer.fit([preprocess(rides[r]) for r in tr.ride_id])
    x_tr, y_tr = build_windows(rides, tr.ride_id, tr.label, normalizer, window, stride)
    log.info("Train: %d rides -> %d windows | Test: %d rides", len(tr), len(x_tr), len(te))

    # Rare classes (Reckless = 5 rides in train) are oversampled so each class is seen equally often
    class_counts = np.bincount(y_tr, minlength=len(CLASSES))
    sample_w = 1.0 / class_counts[y_tr]
    sampler = WeightedRandomSampler(torch.as_tensor(sample_w, dtype=torch.double), len(y_tr), replacement=True)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)), batch_size, sampler=sampler)

    model = RideCNN(N_FEATURES, len(CLASSES), width=width, dropout=dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for xb, yb in loader:
            xb = xb * (1 + noise * torch.randn(xb.size(0), 1, xb.size(2))) + noise * torch.randn_like(xb)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(yb)
            n += len(yb)
        history.append(total / n)
        log.info("epoch %02d  train_loss=%.4f", epoch, history[-1])

    config = {
        "classes": CLASSES,
        "signal_cols": SIGNAL_COLS,
        "window": window,
        "stride": stride,
        "width": width,
        "n_features": N_FEATURES,
        "normalizer": normalizer.to_dict(),
    }
    clf = RideClassifier(model, config)

    # Ride-level evaluation on held-out rides
    y_true = te.label.map(CLASS_TO_IDX).to_numpy()
    y_pred = np.array([int(clf.predict_proba(rides[r]).argmax()) for r in te.ride_id])
    metrics = {
        **compute_metrics(y_true, y_pred, prefix="test_"),
        **baselines(tr.label.map(CLASS_TO_IDX).to_numpy(), y_true, seed),
    }

    report = classification_report(
        y_true, y_pred, labels=range(len(CLASSES)), target_names=CLASSES, zero_division=0, output_dict=True
    )
    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=range(len(CLASSES))),
        index=[f"true_{c}" for c in CLASSES],
        columns=[f"pred_{c}" for c in CLASSES],
    )

    result = {
        "metrics": metrics,
        "history": history,
        "confusion_matrix": cm,
        "params": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "width": width,
            "window": window,
            "stride": stride,
            "seed": seed,
            "noise": noise,
            "dropout": dropout,
            "n_train_rides": len(tr),
            "n_test_rides": len(te),
        },
    }
    if out_dir is None:
        return clf, result
    out_dir = Path(out_dir)
    clf.save(out_dir)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "classification_report.json").write_text(json.dumps(report, indent=2))
    cm.to_csv(out_dir / "confusion_matrix.csv")
    (out_dir / "train_history.json").write_text(json.dumps(history))
    log.info("Saved model + reports to %s", out_dir)
    return clf, result


def log_to_mlflow(out_dir: Path, result: dict, experiment: str, register_name: str | None) -> None:
    """Log params/metrics/artifacts + a pyfunc model. Only runs if MLflow is configured."""
    import mlflow

    from trajclf.mlflow_model import RidePyfunc

    mlflow.set_experiment(experiment)
    with mlflow.start_run() as run:
        mlflow.log_params(result["params"])
        for step, loss in enumerate(result["history"], start=1):
            mlflow.log_metric("train_loss", loss, step=step)
        mlflow.log_metrics({k: v for k, v in result["metrics"].items() if not np.isnan(v)})
        mlflow.log_artifacts(str(out_dir), artifact_path="reports")
        mlflow.pyfunc.log_model(
            name="model",
            python_model=RidePyfunc(),
            artifacts={"model_dir": str(out_dir)},
            code_paths=[str(Path(__file__).resolve().parent)],
            registered_model_name=register_name,
        )
        log.info("MLflow run %s logged", run.info.run_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--out-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--experiment", default="ride-classification")
    parser.add_argument("--register-name", default=None, help="Register the model in the MLflow Model Registry")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    _, result = train(args.data_dir, args.out_dir, args.epochs, args.batch_size, args.lr, args.width, seed=args.seed)
    print(json.dumps(result["metrics"], indent=2))
    print(result["confusion_matrix"].to_string())

    if os.getenv("MLFLOW_TRACKING_URI"):
        log_to_mlflow(args.out_dir, result, args.experiment, args.register_name)
    else:
        log.info("MLFLOW_TRACKING_URI not set: skipping MLflow logging")


if __name__ == "__main__":
    main()
