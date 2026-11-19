"""Airflow DAG: retrain the ride classifier and put it in production if it is better.

    validate_and_prepare ─▶ train_and_register ─▶ promote ─▶ reload_api
                       └──▶ cross_validate (only when the "cross_validate" param is true)

Each task runs one step of `trajclf.pipeline` (the same code as `python -m trajclf.pipeline <step>`)
with the project's own Python, so Airflow and the ML libraries never share dependencies.
`promote` ends "skipped" (not "failed") when the new model is not better: `reload_api` is then
skipped too and production stays on the current version.

Runs on demand (Trigger DAG in the UI) and every Monday at 03:00. See docs/pipeline.md.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, Param

PROJECT_DIR = os.getenv("PROJECT_DIR", "/opt/project")  # repository root (mounted in the Airflow container)
PROJECT_PYTHON = os.getenv("PROJECT_PYTHON", "python")  # interpreter with the project's dependencies
SKIP = 99  # = trajclf.pipeline.NOT_PROMOTED_EXIT_CODE; exit code that Airflow shows as "skipped"
EPOCHS = "{% if params.epochs %} --epochs {{ params.epochs }}{% endif %}"

ENV = {
    "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", ""),
    "API_URL": os.getenv("API_URL", ""),
    "RELOAD_TOKEN": os.getenv("RELOAD_TOKEN", ""),
    "PYTHONPATH": f"{PROJECT_DIR}/src",
}


def step(task_id: str, name: str, doc: str, args: str = "", before: str = "", **kwargs) -> BashOperator:
    """One pipeline step = one CLI call; its last stdout line (JSON) becomes the task's XCom."""
    return BashOperator(
        task_id=task_id,
        bash_command=f"{before}cd {PROJECT_DIR} && {PROJECT_PYTHON} -m trajclf.pipeline {name}{args}",
        env=ENV,
        append_env=True,
        skip_on_exit_code=SKIP,
        doc_md=doc,
        **kwargs,
    )


with DAG(
    dag_id="ride_classifier_training",
    description="Validate data → train → register in MLflow → promote if better → reload the API",
    schedule="0 3 * * 1",
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=2)},
    params={
        "cross_validate": Param(False, type="boolean", description="Also run 5-fold CV (about 5x training time)"),
        "epochs": Param(None, type=["null", "integer"], description="Override params.yaml (quick test runs)"),
    },
    tags=["mlops", "training"],
    doc_md=__doc__,
) as dag:
    prepare = step(
        "validate_and_prepare",
        "prepare",
        "Data quality gate (schema, 10 Hz sampling, physical ranges, labels), then clean and split by ride. "
        "Fails the run on bad data.",
    )
    train = step(
        "train_and_register",
        "train",
        "Train the CNN, evaluate it on hold-out rides against the baselines, log to MLflow, register a new version.",
        args=EPOCHS,
    )
    promote = step(
        "promote",
        "promote",
        "Move @production to the new version only if it beats both baselines and the current production. "
        "Skipped otherwise, which also skips reload_api.",
        retries=0,
    )
    reload_api = step("reload_api", "reload", "POST /model/reload so the API serves the new @production version.")
    cross_validate = step(
        "cross_validate",
        "cv",
        "5-fold cross-validation logged in the ride-classification-cv experiment. Skipped unless the "
        "`cross_validate` param is true.",
        args=EPOCHS,
        before=f"{{% if not params.cross_validate %}}exit {SKIP}{{% endif %}}\n",
    )

    prepare >> train >> promote >> reload_api
    prepare >> cross_validate
