"""Point the `production` alias of the registered model at the version the API should serve.

    export MLFLOW_TRACKING_URI=http://localhost:5050
    python -m trajclf.promote                 # newest version, only if it is good enough
    python -m trajclf.promote --version 3     # manual choice or rollback, no checks

The newest version is promoted only if, on its hold-out test rides, it
  1. beats both naive baselines logged with it ("always Safe" and random), and
  2. is at least as good as the current @production version on the same metric (default macro-F1).
The replaced version gets the alias `previous`, so a rollback is `--version <previous version>`.
The API loads models:/ride-classifier@production and never needs a version number.
"""

from __future__ import annotations

import argparse

from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

MODEL_NAME = "ride-classifier"
PRODUCTION = "production"
PREVIOUS = "previous"
BASELINES = ("baseline_most_frequent", "baseline_stratified")


def decide(candidate: dict, production: dict | None, metric: str = "f1_macro") -> tuple[bool, str]:
    """Promotion rule on the metrics of two training runs. Returns (promote?, reason)."""
    key = f"test_{metric}"
    score = candidate.get(key)
    if score is None:
        return False, f"the candidate run has no '{key}' metric"
    for baseline in BASELINES:
        base = candidate.get(f"{baseline}_{metric}")
        if base is not None and score <= base:
            return False, f"{key}={score:.3f} does not beat {baseline} ({base:.3f})"
    if production is None:
        return True, f"{key}={score:.3f} beats the baselines (no production model yet)"
    current = production.get(key)
    if current is not None and score < current:
        return False, f"{key}={score:.3f} is below the production model ({current:.3f})"
    return True, f"{key}={score:.3f} beats the baselines and is >= production ({current:.3f})"


def _production(client: MlflowClient, name: str) -> ModelVersion | None:
    try:
        return client.get_model_version_by_alias(name, PRODUCTION)
    except MlflowException:
        return None


def _run_metrics(client: MlflowClient, version: ModelVersion) -> dict:
    return client.get_run(version.run_id).data.metrics if version.run_id else {}


def promote(
    client: MlflowClient, name: str = MODEL_NAME, version: int | None = None, metric: str = "f1_macro"
) -> tuple[bool, str]:
    """Move the `production` alias if allowed. Returns (production changed or already set, message)."""
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise SystemExit(f"No version of '{name}' registered: run python -m trajclf.train --register-name {name}")
    current = _production(client, name)

    if version is None:
        candidate = max(versions, key=lambda v: int(v.version))
        if current is not None and str(current.version) == str(candidate.version):
            return True, f"v{candidate.version} is already @{PRODUCTION}"
        ok, reason = decide(
            _run_metrics(client, candidate), _run_metrics(client, current) if current else None, metric
        )
    else:
        candidate = client.get_model_version(name, str(version))
        ok, reason = True, "chosen manually"
        if current is not None and str(current.version) == str(candidate.version):
            return True, f"v{candidate.version} is already @{PRODUCTION}"

    # The decision is kept on the version itself (tag "promotion"), visible in the registry UI.
    client.set_model_version_tag(name, candidate.version, "promotion", f"{'promoted' if ok else 'refused'}: {reason}")
    if not ok:
        kept = f"@{PRODUCTION} stays on v{current.version}" if current else f"no @{PRODUCTION} model yet"
        return False, f"v{candidate.version} not promoted: {reason}; {kept}"
    if current is not None:
        client.set_registered_model_alias(name, PREVIOUS, current.version)
    client.set_registered_model_alias(name, PRODUCTION, candidate.version)
    before = f" (was v{current.version}, now @{PREVIOUS})" if current else ""
    return True, f"v{candidate.version} is now @{PRODUCTION}{before}: {reason}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default=MODEL_NAME, help="registered model name")
    parser.add_argument("--version", type=int, default=None, help="promote this version without checks")
    parser.add_argument("--metric", default="f1_macro", choices=["f1_macro", "balanced_accuracy"])
    args = parser.parse_args()
    _, message = promote(MlflowClient(), args.name, args.version, args.metric)
    print(message)


if __name__ == "__main__":
    main()
