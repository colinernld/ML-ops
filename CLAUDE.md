# Project context (read first)

Shared context for everyone working on this repo, and for their AI assistants (Claude Code loads this
file automatically). **Keep it current**: when your part changes status or you find something the
others must know, update the tables below in your PR.

## What this is

MLOps course project (Albert School, group of 5). A deep learning model classifies a bike/scooter ride
from phone IMU data (accelerometer + gyroscope, 10 Hz) into **Safe, Safe overall, Bad weather, Tandem,
Reckless**, with a full MLOps pipeline around it. Data: `Project.zip` on Inside → `data/raw/` (never in git).

## Course requirements (summary of the project brief on Inside)

- **Deadline 25/11/2026 20:00** (late = penalty). Presentation 10 min + 10 min questions on **26/11**.
  The list of group members must reach the instructors by **20/11**.
- **Two environments**: dev = local Docker Compose; production = *all* services (API, MLflow, ...) on a
  local Kubernetes cluster (Docker Desktop; Helm charts allowed). Cloud is optional.
- **Objectives**: 1 extract data · 2 deep learning classifier · 3 store the model in S3 (MinIO) ·
  4 track experiments and model versions with MLflow · 5 prediction API · 6 Dockerize the API and deploy
  it on Kubernetes with GitHub Actions CI/CD (self-hosted runner) · 7 version and document on GitHub.
- **Bonus**: retraining pipeline in GitHub Actions (get data → retrain → redeploy) · monitoring
  (Prometheus + Grafana, ELK, ...) · feature store.
- **Repo must contain**: all code (preprocessing, training, API, interactive web app, optional monitoring),
  pipeline/workflow DAGs (Airflow or other), GitHub Actions CI/CD, tests (unit, integration, end-to-end),
  documentation, a README with technical choices, results and screenshots of the production environment,
  and the **Docker Hub URLs** of our images.
- **Grading**: code quality (best practices, modularity, docs) · CI/CD automation · MLflow tracking and
  model management · API functionality and usability (Swagger) · consistent architecture, model better
  than random (accuracy itself matters little) · collaboration and versioning on GitHub.

## Parts and status

| Part | Status | Where |
|---|---|---|
| Data + model | done | `src/trajclf/`, [docs/data.md](docs/data.md), [docs/model_card.md](docs/model_card.md) |
| MLflow + MinIO | done | `docker-compose.yml`, `docker/mlflow/`, `src/trajclf/promote.py`, [docs/mlflow.md](docs/mlflow.md) |
| API (FastAPI) + web app (Streamlit) | done | `src/api/`, `webapp/`, [docs/api.md](docs/api.md) |
| Docker + Kubernetes + CI/CD | to do | `docker/`, `k8s/`, `.github/workflows/` |
| Training pipeline (Airflow DAG) | done (Airflow image: first build on Docker Desktop to confirm) | `dags/`, `src/trajclf/pipeline.py`, `docker/airflow/`, [docs/pipeline.md](docs/pipeline.md) |
| Monitoring + README + slides | to do | `monitoring/`, `README.md` |

## Interfaces to build on

- **Run the dev stack**: `cp .env.example .env && docker compose up -d` → MLflow <http://localhost:5050>,
  MinIO console <http://localhost:9001>. Check it: `python scripts/check_mlflow.py`.
- **Model to serve**: `mlflow.pyfunc.load_model("models:/ride-classifier@production")`. Never hard-code a
  version. For the plain classifier: `.unwrap_python_model().clf` → `.predict(df)` returns
  `{"label", "probabilities", "n_windows"}` and raises `trajclf.validation.DataValidationError` on bad
  input (→ HTTP 422). Loading needs the package and PyTorch: `pip install -e ".[mlflow]"`.
- **Input**: one ride as a DataFrame with columns `ax, ay, az, rx, ry, rz` at 10 Hz, 50 – 36,000 rows.
  Samples: `examples/ride_safe.csv`, `examples/ride_tandem.csv`.
- **`MLFLOW_TRACKING_URI`**: laptop `http://localhost:5050` · container in the Compose project
  `http://mlflow:5000` · Kubernetes: the MLflow Service name (also add it to `MLFLOW_ALLOWED_HOSTS`).
- **New model version** (also for retraining): `python -m trajclf.train --register-name ride-classifier`
  then `python -m trajclf.promote` (moves `@production` only if the new version beats both baselines and
  the current production version; the old one becomes `@previous`; rollback: `--version N`).
- **API** (details in [docs/api.md](docs/api.md)): `uvicorn api.main:app --app-dir src --port 8000`, Swagger at
  `/docs`. Probes: liveness `GET /health`, readiness `GET /ready`. Prometheus metrics at `GET /metrics`
  (requests, latency, predictions per class, confidence, input drift). Config by env only: `MLFLOW_TRACKING_URI`,
  `MODEL_URI` (default `models:/ride-classifier@production`), `MODEL_DIR` (local model, overrides), `RELOAD_TOKEN`.
  After `trajclf.promote`, `POST /model/reload` (or restart the pod) serves the new version.
- **Training pipeline**: Airflow DAG `ride_classifier_training` (validate → train + register → promote if
  better → reload the API). `docker compose --profile airflow up -d` → <http://localhost:8080>. Same steps
  without Airflow: `python -m trajclf.pipeline <prepare|train|cv|promote|reload|all>` (reuse it for the GitHub
  Actions retraining job). Details in [docs/pipeline.md](docs/pipeline.md).
- **Web app**: `API_URL=http://localhost:8000 streamlit run webapp/app.py` (port 8501). Calls the API only: its
  image needs no PyTorch.
- **Monitoring ideas**: end of [docs/model_card.md](docs/model_card.md); metrics already exposed are listed in
  [docs/api.md](docs/api.md#metrics-for-monitoring-get-metrics).

## Gotchas

- macOS uses port 5000 (AirPlay), so MLflow is on **5050** on the host.
- The official `minio/minio` image no longer exists: we use `pgsty/minio` (console branded "SILO").
- MLflow 3.18 rejects unknown `Host` headers: a new hostname for MLflow must be added to `MLFLOW_ALLOWED_HOSTS`.
- Postgres URL uses `postgresql+psycopg2://` (plain `postgresql://` looks for psycopg 3, not installed).

## Open items

Tracked as GitHub issues (details and "done when" checklists there). Assign yourself on the issue, link
it from your PR with `Closes #N`, and remove the line here once it is merged.

- [#4](https://github.com/colinernld/ML-ops/issues/4) **MLflow, MinIO and Postgres must also run on
  Kubernetes** (production). Docker + K8s part.
- [#5](https://github.com/colinernld/ML-ops/issues/5) **Push our images to Docker Hub** (API, web app,
  MLflow server) and list the URLs in the README. Needs a team Docker Hub account.
- [#7](https://github.com/colinernld/ML-ops/issues/7) **Results differ by environment** (5-fold CV with
  torch 2.14.1: balanced accuracy 0.36, macro-F1 0.31 vs 0.44 / 0.36 in the README, still above the
  baselines) and **both example rides are predicted "Safe overall"**: keep in mind for the API demo.

## Conventions

See [CONTRIBUTING.md](CONTRIBUTING.md): one branch per task (`feat/<part>-...`), pull request with one
review, squash merge, commit messages prefixed by the part (`api: ...`). Run `ruff check . && pytest`
before pushing. Never commit `data/`, `artifacts/`, `mlruns/` or `.env`.
