"""Structure of the Airflow DAG (runs only where Airflow is installed, e.g. in the Airflow image)."""

from pathlib import Path

import pytest

pytest.importorskip("airflow")
from airflow.dag_processing.dagbag import DagBag

from trajclf.pipeline import NOT_PROMOTED_EXIT_CODE

DAGS = Path(__file__).resolve().parents[1] / "dags"


@pytest.fixture(scope="module")
def dag():
    bag = DagBag(dag_folder=str(DAGS))
    assert bag.import_errors == {}
    return bag.get_dag("ride_classifier_training")


def test_dependencies(dag):
    assert dag.get_task("validate_and_prepare").downstream_task_ids == {"train_and_register", "cross_validate"}
    assert dag.get_task("train_and_register").downstream_task_ids == {"promote"}
    assert dag.get_task("promote").downstream_task_ids == {"reload_api"}


def test_not_promoted_means_skipped(dag):
    assert NOT_PROMOTED_EXIT_CODE in dag.get_task("promote").skip_on_exit_code
    assert dag.get_task("promote").retries == 0  # a refused promotion must not be retried


def test_each_task_calls_one_pipeline_step(dag):
    steps = {t.task_id: t.bash_command for t in dag.tasks}
    assert "trajclf.pipeline prepare" in steps["validate_and_prepare"]
    assert "trajclf.pipeline train" in steps["train_and_register"]
    assert "trajclf.pipeline reload" in steps["reload_api"]
    assert "params.cross_validate" in steps["cross_validate"]
