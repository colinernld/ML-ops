"""Stratified k-fold cross-validation at ride level.

With only 500 rides (6 Reckless), a single 100-ride test set is noisy: one ride flips the
Reckless recall from 0 to 100 %. Cross-validation gives a mean +/- std for the report.

Usage:
    python -m trajclf.evaluate --folds 5
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from trajclf.config import ARTIFACTS_DIR, PROCESSED_DIR, SEED
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--out", type=Path, default=ARTIFACTS_DIR / "cv_results.csv")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    res = cross_validate(args.data_dir, args.folds, epochs=args.epochs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out)
    summary = res.agg(["mean", "std"]).T.round(3)
    print(summary.to_string())
    (args.out.with_suffix(".json")).write_text(json.dumps(summary.to_dict(), indent=2))


if __name__ == "__main__":
    main()
