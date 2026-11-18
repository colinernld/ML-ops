"""Promotion rule + alias handling in the MLflow Model Registry (local SQLite store, no server needed)."""

import pytest

mlflow = pytest.importorskip("mlflow")  # optional dependency: pip install -e ".[mlflow]"
promotion = pytest.importorskip("trajclf.promote")
decide, promote = promotion.decide, promotion.promote
PRODUCTION, PREVIOUS = promotion.PRODUCTION, promotion.PREVIOUS

BASE = {"baseline_most_frequent_f1_macro": 0.18, "baseline_stratified_f1_macro": 0.26}


def test_decide_needs_to_beat_baselines():
    assert decide({"test_f1_macro": 0.36, **BASE}, None)[0]
    ok, reason = decide({"test_f1_macro": 0.20, **BASE}, None)
    assert not ok and "baseline_stratified" in reason
    assert not decide({**BASE}, None)[0]  # metric missing


def test_decide_compares_with_production():
    assert decide({"test_f1_macro": 0.36, **BASE}, {"test_f1_macro": 0.30})[0]
    assert decide({"test_f1_macro": 0.36, **BASE}, {"test_f1_macro": 0.36})[0]  # tie: newer wins
    assert not decide({"test_f1_macro": 0.30, **BASE}, {"test_f1_macro": 0.36})[0]
    assert decide({"test_balanced_accuracy": 0.5, "baseline_most_frequent_balanced_accuracy": 0.2}, None,
                  metric="balanced_accuracy")[0]


@pytest.fixture
def client(tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    c = mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)
    c.create_experiment("test", artifact_location=str(tmp_path / "artifacts"))
    c.create_registered_model("ride-classifier")
    return c


def add_version(client, f1):
    exp = client.get_experiment_by_name("test").experiment_id
    run = client.create_run(exp)
    for key, value in {"test_f1_macro": f1, **BASE}.items():
        client.log_metric(run.info.run_id, key, value)
    return client.create_model_version("ride-classifier", f"{run.info.artifact_uri}/model", run_id=run.info.run_id)


def alias(client, name):
    return str(client.get_model_version_by_alias("ride-classifier", name).version)


def test_promote_flow(client):
    add_version(client, 0.20)  # v1 below the random baseline
    assert not promote(client)[0]
    assert client.get_model_version("ride-classifier", "1").tags["promotion"].startswith("refused")

    add_version(client, 0.30)  # v2 first good model
    assert promote(client)[0] and alias(client, PRODUCTION) == "2"
    assert client.get_model_version("ride-classifier", "2").tags["promotion"].startswith("promoted")
    assert promote(client)[0]  # already production: no change

    add_version(client, 0.28)  # v3 worse than production
    ok, message = promote(client)
    assert not ok and "stays on v2" in message and alias(client, PRODUCTION) == "2"

    add_version(client, 0.36)  # v4 better
    assert promote(client)[0]
    assert alias(client, PRODUCTION) == "4" and alias(client, PREVIOUS) == "2"

    assert promote(client, version=2)[0]  # manual rollback
    assert alias(client, PRODUCTION) == "2" and alias(client, PREVIOUS) == "4"
