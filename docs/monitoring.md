# Monitoring (Prometheus + Grafana)

The API measures itself (`GET /metrics`, [src/api/monitoring.py](../src/api/monitoring.py)), Prometheus collects
the metrics every 15 s and evaluates the alert rules, Grafana shows everything on one dashboard. The same
configuration files ([`monitoring/`](../monitoring)) run in dev (Docker Compose) and in production (Kubernetes).

```
API pods ── GET /metrics ─────────────────────────┐
(service, input data and model metrics)           │  scrape every 15 s
                                                  ▼
blackbox exporter ── health checks ──────────▶ Prometheus ── alert rules (alerts.yml) ──▶ Alerts tab
(api /ready, webapp, mlflow, minio, postgres)     │
                                                  ▼  PromQL
                                               Grafana ── dashboard "Ride classifier - monitoring"
```

| | Dev (Docker Compose) | Production (Kubernetes) |
|---|---|---|
| Start | `docker compose --profile monitoring up -d` | deployed with the rest: `kubectl apply -k k8s/` or the `cd` workflow |
| Grafana (dashboard on the home page) | <http://localhost:3000> | <http://localhost:30300> |
| Prometheus (Alerts tab) | <http://localhost:9090/alerts> | <http://localhost:30909/alerts> |
| How the API is found | static target `api:8000` ([prometheus.yml](../monitoring/prometheus/prometheus.yml)) | every API pod annotated `prometheus.io/scrape` ([prometheus-k8s.yml](../monitoring/prometheus/prometheus-k8s.yml)) |

Grafana opens the dashboard without login (read-only). To edit, log in as `admin` with
`GRAFANA_ADMIN_PASSWORD` from `.env` (copied to Kubernetes by `k8s/create-secrets.sh`).

## Four layers

The dashboard has one row per layer, and every alert belongs to one layer (label `layer`).

| Layer | Question | Metrics | Alerts |
|---|---|---|---|
| **1 · Infrastructure** | Are all the services up? Does the API have enough memory? | `probe_success`, `probe_duration_seconds` (blackbox), `up`, `process_cpu_seconds_total`, `process_resident_memory_bytes`, `process_start_time_seconds` | `ServiceDown`, `TargetDown`, `ApiMemoryHigh` |
| **2 · Service** | Does the API answer, fast and without errors? | `api_requests_total{path,status}`, `api_request_duration_seconds` | `ApiHighErrorRate`, `ApiSlowPredictions` |
| **3 · Input data** | Do the rides sent still look like the training data? (input drift) | `ride_acc_magnitude_mean`, `ride_rot_magnitude_mean`, `ride_samples`, `ride_missing_ratio`, `prediction_errors_total{reason}` | `HighRejectionRate`, `AccelerationDrift`, `RotationDrift`, `MissingValuesDrift`, `ShortRides` |
| **4 · Model** | Which version serves, what does it predict, how sure is it? (output drift) | `model_loaded`, `model_info{version}`, `predictions_total{label}`, `prediction_confidence` | `ModelNotLoaded`, `PredictionDistributionDrift`, `LowPredictionConfidence` |

**Why no accuracy in production?** The true label of a ride is not known when it is scored. Model quality is
measured on hold-out rides at every (re)training, logged in MLflow, and used as the promotion gate
([mlflow.md](mlflow.md)). In production we watch what can be watched without labels: whether the **inputs**
drift away from the training data, and whether the **outputs** (class mix, confidence) change.

## Alerts

Rules: [`monitoring/prometheus/alerts.yml`](../monitoring/prometheus/alerts.yml). Thresholds come from the
training data ([data.md](data.md)) and the demo rides (`examples/`: mean |a| 9.4 – 9.6 m/s², mean |r|
16 – 40 °/s, 2,000 – 2,900 samples).

| Alert | Layer | Severity | Fires when (for 2 – 5 min) | Reference | First thing to check |
|---|---|---|---|---|---|
| `ServiceDown` | infrastructure | critical | a health check fails (api `/ready`, webapp, mlflow, minio, postgres) | – | `kubectl -n ride-mlops get pods`, logs of that service |
| `TargetDown` | infrastructure | critical | Prometheus cannot scrape a target (API `/metrics`, blackbox) | – | is the API container / pod running? |
| `ApiMemoryHigh` | infrastructure | warning | API memory > 1.2 GB | K8s limit 1.5 GiB | very long rides, memory leak: restart, raise the limit |
| `ApiHighErrorRate` | service | critical | > 5 % of `/predict*` requests answer 5xx | 0 % | usually no model loaded (503): `GET /ready`, alias `@production` in MLflow |
| `ApiSlowPredictions` | service | warning | 95th percentile of prediction latency > 2 s | ~0.1 – 0.3 s | API CPU (row 1), number of replicas |
| `HighRejectionRate` | data | warning | > 20 % of the rides sent are rejected (413 / 422) | ~0 % | `prediction_errors_total` by reason, API logs: which client sends bad rides? |
| `AccelerationDrift` | data | warning | mean \|a\| per ride outside 8 – 11 m/s² | ≈ 9.5 m/s² (gravity) | accelerometer sent in g (≈ 1) or without gravity |
| `RotationDrift` | data | warning | mean \|r\| per ride outside 3 – 35 °/s | 16 °/s (Safe) – 40 °/s (Reckless) | gyroscope in rad/s, new phones, new population → retrain |
| `MissingValuesDrift` | data | warning | > 5 % of rows have a missing value | 0.7 % | sensor or app problem on the client |
| `ShortRides` | data | warning | rides shorter than 1 min on average | 3 – 8 min | the model scores 25.6 s windows: few windows = unreliable |
| `ModelNotLoaded` | model | critical | the API has no model | – | MLflow up? a version promoted to `@production`? |
| `PredictionDistributionDrift` | model | warning | share of rides predicted Safe outside 50 – 97 % | ≈ 83 % | input panels; > 97 % = collapse to "always Safe" (our first model did that) |
| `LowPredictionConfidence` | model | warning | mean confidence < 35 % | ≈ 60 % (5 classes: 20 % = random) | input panels: rides unlike the training data |

Data and model alerts need a minimum volume (10 rides, 20 predictions in 5 min), so a single odd ride
does not fire them. Their windows are **5 minutes so that a drift can be shown live**; on real traffic they
would be 1 hour to 1 day.

When an input drift is confirmed and is not a client bug (new phones, new cities, new riders), the answer is
to **retrain on recent data**: Actions → *retrain* → *Run workflow* ([cicd.md](cicd.md#retraining-workflow)).
The new version only reaches production if it beats the current one.

## Demo: simulate traffic and drifts

[`scripts/simulate_traffic.py`](../scripts/simulate_traffic.py) sends the demo rides of `examples/` to the
API with the class mix of the training data (83 % Safe), optionally with a drift. Standard library only:
any Python 3.10+ works, no install needed.

```bash
python scripts/simulate_traffic.py                                   # dev API, normal traffic, 10 min
python scripts/simulate_traffic.py --api http://localhost:30800 --duration 0   # production, until Ctrl+C
python scripts/simulate_traffic.py --api http://localhost:30800 --drift units  # in another terminal
```

| `--drift` | What it changes | What you see | Alert (~3 min) |
|---|---|---|---|
| `none` | nothing (default) | all green, ≈ 83 % Safe | none |
| `units` | accelerometer divided by 9.81 (sent in g) | row 3: mean \|a\| drops from 9.5 to ≈ 1 m/s² | `AccelerationDrift` |
| `rotation` | gyroscope × 3 | row 3: mean \|r\| ≈ 50 °/s; row 4: class mix changes | `RotationDrift` |
| `missing` | 15 % of rows empty (accepted: limit 20 %) | row 3: missing values 15 % | `MissingValuesDrift` |
| `short` | rides cut to 30 s | row 3: ride length 0.5 min | `ShortRides` |
| `invalid` | 1 ride in 2 cut to 2 s | row 2: 4xx; row 3: rejected rides (validation) | `HighRejectionRate` |

Demo plan for the presentation (production on Kubernetes):

1. Before the talk: normal traffic running (`--duration 0`), Grafana open on <http://localhost:30300>.
2. About 5 minutes before the monitoring slide, start `--drift units` in a second terminal.
3. On the slide: the acceleration panel left its green band, `AccelerationDrift` is firing (top-right counter,
   table at the bottom, Prometheus → Alerts) while the API still answers 200: **validation cannot catch it,
   only monitoring can**.
4. Stop the drift (Ctrl+C): the panel goes back to 9.5 m/s² and the alert resolves after a few minutes.

The normal traffic is calibrated on the model of [`examples/demo_rides.json`](../examples/demo_rides.json)
(`ride_safe.csv` predicted Safe). If the production model predicts it differently, the Safe share moves away
from 83 % and `PredictionDistributionDrift` fires: exactly what it is meant to detect.

## Design choices

- **The API measures itself** with `prometheus_client`: counters and histograms only, labelled by route
  template (`/predict`, not the raw URL). No ride data is stored, only aggregates (privacy, low cost).
- **Pull, with discovery in Kubernetes.** Prometheus finds every API pod through its annotations
  (`k8s/api.yaml`) with a Role that can only read pods of the `ride-mlops` namespace. Each replica is scraped on
  its own, so counters stay correct with several replicas and during rolling updates.
- **Blackbox exporter** for the services that expose no metrics (MLflow, MinIO, Postgres, web app): their health
  checks become `probe_success`. Service names are the same in Compose and Kubernetes, so the checks are identical.
- **Configuration as code, one source of truth.** Compose mounts `monitoring/`; in Kubernetes,
  [`monitoring/kustomization.yaml`](../monitoring/kustomization.yaml) turns the same files into ConfigMaps
  (hash suffix: changing a file restarts the pods that use it). The data source and the dashboard are
  provisioned at startup; edits made in the Grafana UI are not saved, change the JSON in git instead.
- **Tested.** CI runs `promtool check config` on both Prometheus configs and `promtool test rules`
  ([alerts_test.yml](../monitoring/prometheus/alerts_test.yml)): every alert has a scenario where it fires with
  the expected labels and text, and a normal-traffic scenario checks that **no** alert fires.
  `tests/test_monitoring.py` checks that the dashboard and the rules only use metrics that `GET /metrics`
  really exposes, that every alert is tested and documented here, and that dev and production stay consistent.
- **Light enough for a laptop cluster**: Prometheus 192 Mi, Grafana 128 Mi, blackbox exporter 32 Mi requested;
  7 days of history. Prometheus keeps its data in an `emptyDir` (lost when the pod is replaced): fine for a laptop,
  a real production would use a PersistentVolumeClaim.
- **Not done (next steps)**: Alertmanager to send alerts (Slack, e-mail) or to call the `retrain` workflow
  (`repository_dispatch` event "retrain", already accepted); Kubernetes-level metrics (kube-state-metrics,
  cAdvisor) and log aggregation (Loki or ELK), e.g. with the kube-prometheus-stack Helm chart on a bigger cluster.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Grafana shows "No data" everywhere | Prometheus → Status → Targets: is `ride-api` UP? In dev with the API started by `uvicorn` on the laptop, set the target to `host.docker.internal:8000` in `prometheus.yml`. |
| Rows 3 and 4 are empty | no prediction yet: run `scripts/simulate_traffic.py` or use the web app |
| Health check `api` DOWN while the API answers | `/ready` answers 503: no model promoted to `@production` (`ModelNotLoaded` fires too) |
| Health check `mlflow` DOWN, probe gets HTTP 403 | MLflow rejects the `Host` header: add the name to `MLFLOW_SERVER_ALLOWED_HOSTS` |
| `localhost:30300` does not answer | `kubectl -n ride-mlops get pods`: grafana Running? Docker Desktop cluster type must be kubeadm (NodePorts on localhost) |
| Metrics history gone | the Prometheus pod was replaced (new config or image): its data is in an `emptyDir` |
| Dashboard change lost after a restart | provisioned dashboards are read from `monitoring/grafana/dashboards/ride-api.json`: edit that file |
