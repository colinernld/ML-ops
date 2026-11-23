"""Monitoring configuration (monitoring/): the Grafana dashboard and the alert rules only use metrics the API
really exposes, every alert is tested and documented, and the dev (Compose) and production (Kubernetes)
setups stay consistent. The PromQL of the alerts is unit-tested by promtool in CI (alerts_test.yml)."""

import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

pytest.importorskip("fastapi")  # optional dependency: pip install -e ".[api]"
from fastapi.testclient import TestClient

from api.main import create_app
from api.settings import Settings
from trajclf.config import CLASSES, SIGNAL_COLS
from trajclf.features import N_FEATURES, Normalizer
from trajclf.model import RideCNN
from trajclf.predict import RideClassifier

ROOT = Path(__file__).resolve().parents[1]
MONITORING = ROOT / "monitoring"
DASHBOARD = MONITORING / "grafana" / "dashboards" / "ride-api.json"
PROMETHEUS = MONITORING / "prometheus"
# Metrics produced by the API itself (src/api/monitoring.py). The others come from Prometheus (up, ALERTS),
# the blackbox exporter (probe_*) or the Python client's process collector (process_*, Linux only).
API_PREFIXES = ("api_", "model_", "prediction", "ride_")
LAYERS = {"infrastructure", "service", "data", "model"}


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def alert_rules() -> list[dict]:
    return [rule for group in load_yaml(PROMETHEUS / "alerts.yml")["groups"] for rule in group["rules"]]


def promql_expressions() -> list[str]:
    dashboard = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    exprs = [target["expr"] for panel in dashboard["panels"] for target in panel.get("targets", [])]
    return exprs + [rule["expr"] for rule in alert_rules()]


def api_metric_names(expr: str) -> set[str]:
    expr = re.sub(r'"[^"]*"', "", expr)  # label values: "Safe", "/predict.*"
    expr = re.sub(r"\{[^}]*\}|\[[^\]]*\]", "", expr)  # label matchers and ranges: {...}, [5m]
    return {word for word in re.findall(r"[a-zA-Z_:][a-zA-Z0-9_:]*", expr) if word.startswith(API_PREFIXES)}


@pytest.fixture(scope="module")
def exposed_metrics(tmp_path_factory) -> set[str]:
    """Names of the samples on GET /metrics after one valid and one rejected prediction."""
    torch.manual_seed(0)
    config = {
        "classes": CLASSES,
        "signal_cols": SIGNAL_COLS,
        "window": 64,
        "stride": 32,
        "width": 4,
        "n_features": N_FEATURES,
        "normalizer": Normalizer(mean=[0.0] * N_FEATURES, std=[1.0] * N_FEATURES).to_dict(),
    }
    model_dir = tmp_path_factory.mktemp("model")
    RideClassifier(RideCNN(N_FEATURES, len(CLASSES), width=4), config).save(model_dir)
    rng = np.random.default_rng(0)
    ride = {c: rng.normal(size=120).round(4).tolist() for c in SIGNAL_COLS}
    with TestClient(create_app(Settings(model_dir=str(model_dir)))) as client:
        assert client.post("/predict", json=ride).status_code == 200
        assert client.post("/predict", json={c: v[:10] for c, v in ride.items()}).status_code == 422
        text = client.get("/metrics").text
    return {line.split("{")[0].split(" ")[0] for line in text.splitlines() if line and not line.startswith("#")}


def test_dashboard_and_alerts_only_use_metrics_the_api_exposes(exposed_metrics):
    used = set().union(*(api_metric_names(expr) for expr in promql_expressions()))
    assert {"predictions_total", "ride_acc_magnitude_mean_sum", "api_request_duration_seconds_bucket"} <= used
    assert used <= exposed_metrics, f"used but not exposed by GET /metrics: {sorted(used - exposed_metrics)}"


def test_dashboard_is_valid():
    dashboard = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    ids = [panel["id"] for panel in dashboard["panels"]]
    assert len(ids) == len(set(ids)), "duplicated panel id"
    datasources = load_yaml(MONITORING / "grafana" / "provisioning" / "datasources" / "prometheus.yml")
    uid = datasources["datasources"][0]["uid"]
    for panel in dashboard["panels"]:
        assert panel["gridPos"]["x"] + panel["gridPos"]["w"] <= 24, panel["title"]
        for target in panel.get("targets", []):
            assert target["datasource"]["uid"] == uid, panel["title"]
    rows = [panel["title"] for panel in dashboard["panels"] if panel["type"] == "row"]
    assert [title.split(" · ")[0] for title in rows] == ["1", "2", "3", "4"]  # the 4 monitoring layers


def test_every_alert_is_labelled_tested_and_documented():
    rules = alert_rules()
    assert len(rules) >= 10
    for rule in rules:
        assert rule["labels"]["severity"] in {"warning", "critical"}, rule["alert"]
        assert rule["labels"]["layer"] in LAYERS, rule["alert"]
        assert rule["annotations"]["summary"] and rule["annotations"]["description"], rule["alert"]
    assert {rule["labels"]["layer"] for rule in rules} == LAYERS

    tests = load_yaml(PROMETHEUS / "alerts_test.yml")["tests"]
    cases = [case for test in tests for case in test["alert_rule_test"]]
    fired = {case["alertname"] for case in cases if case.get("exp_alerts")}
    untested = sorted({rule["alert"] for rule in rules} - fired)
    assert not untested, f"no promtool test where these alerts fire: {untested}"

    docs = (ROOT / "docs" / "monitoring.md").read_text(encoding="utf-8")
    undocumented = [rule["alert"] for rule in rules if rule["alert"] not in docs]
    assert not undocumented, f"alerts missing from docs/monitoring.md: {undocumented}"


def _health_checks(config: dict) -> dict:
    jobs = config["scrape_configs"]
    return {job["job_name"]: job["static_configs"] for job in jobs if job.get("metrics_path") == "/probe"}


def test_dev_and_production_prometheus_are_consistent():
    dev, prod = load_yaml(PROMETHEUS / "prometheus.yml"), load_yaml(PROMETHEUS / "prometheus-k8s.yml")
    assert [job["job_name"] for job in dev["scrape_configs"]] == [job["job_name"] for job in prod["scrape_configs"]]
    assert dev["rule_files"] == prod["rule_files"] == ["/etc/prometheus/alerts.yml"]
    # Compose service names = Kubernetes Service names, so the health checks are identical
    assert _health_checks(dev) and _health_checks(dev) == _health_checks(prod)


def test_kubernetes_wiring():
    docs = yaml.safe_load_all((ROOT / "k8s" / "api.yaml").read_text(encoding="utf-8"))
    api = next(doc for doc in docs if doc["kind"] == "Deployment")
    annotations = api["spec"]["template"]["metadata"]["annotations"]
    assert annotations["prometheus.io/scrape"] == "true"  # pod discovery in prometheus-k8s.yml
    assert annotations["prometheus.io/port"] == "8000"

    kustomization = load_yaml(MONITORING / "kustomization.yaml")
    for resource in kustomization["resources"]:
        assert (MONITORING / resource).is_file(), resource
    for generator in kustomization["configMapGenerator"]:
        for entry in generator["files"]:
            assert (MONITORING / entry.split("=")[-1]).is_file(), entry
    assert "../monitoring" in load_yaml(ROOT / "k8s" / "kustomization.yaml")["resources"]
