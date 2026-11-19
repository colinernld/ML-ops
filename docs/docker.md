# Docker images

Three images make up the application. They are built and published by GitHub Actions
(`.github/workflows/docker-images.yml`), so nobody needs to build them locally.

| Image | Dockerfile | Content | Port | Health check |
|---|---|---|---|---|
| `ride-api` | `docker/api/Dockerfile` | FastAPI + PyTorch (CPU) + MLflow client + `trajclf` | 8000 | `GET /health` (liveness), `GET /ready` (model loaded) |
| `ride-webapp` | `docker/webapp/Dockerfile` | Streamlit + sample rides, no PyTorch | 8501 | `GET /_stcore/health` |
| `ride-mlflow` | `docker/mlflow/Dockerfile` | MLflow 3.18 server + Postgres and S3 drivers | 5000 | `GET /health` |

Docker Hub:

- https://hub.docker.com/r/colinernld/ride-api
- https://hub.docker.com/r/colinernld/ride-webapp
- https://hub.docker.com/r/colinernld/ride-mlflow

Tags: `latest` (last commit on `main`) and `sha-<commit>` (immutable, use this one in Kubernetes for
rollbacks). Images are multi-architecture: `linux/amd64` and `linux/arm64` (Apple Silicon).

## Design choices

- **Same image everywhere** (Compose, Kubernetes): configuration only through environment variables.
- **CPU-only PyTorch** in the API image: several GB smaller than the default CUDA build. PyTorch is pinned
  (`TORCH_VERSION=2.14.1`, build argument) to the version that trained the registered model, since results
  depend on it (issue #7).
- **The model is not inside the image**: the API downloads `models:/ride-classifier@production` from
  MLflow at startup. A new model version does not need a new image (`POST /model/reload` or restart).
- **Web app without PyTorch**: it only calls the API, so its image is small and starts fast.
- **Non-root user** (uid 10001) in both images, ready for Kubernetes `runAsNonRoot`.
- **Layer order**: PyTorch is installed in its own layer before the code is copied, so a code change
  rebuilds only the last layers.
- **`.dockerignore`** keeps `data/`, `artifacts/`, `mlruns/`, `.env` and `.git` out of every image.

## Environment variables

| Image | Variable | Default | Meaning |
|---|---|---|---|
| api | `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow server (Service name in Compose and Kubernetes) |
| api | `MLFLOW_HTTP_REQUEST_MAX_RETRIES` | `2` | few retries, so the API starts quickly even if MLflow is down |
| api | `MODEL_URI` | `models:/ride-classifier@production` | model to serve |
| api | `MODEL_DIR` | — | local model folder (overrides `MODEL_URI`, for tests) |
| api | `RELOAD_TOKEN` | — | secret for `POST /model/reload` |
| webapp | `API_URL` | `http://api:8000` | where the API is |

## Dev environment (Docker Compose)

```bash
cp .env.example .env
docker compose up -d --build     # builds api + webapp locally the first time (a few minutes, a few GB of disk)
```

| Service | URL |
|---|---|
| API (Swagger) | http://localhost:8000/docs |
| Web app | http://localhost:8501 |
| MLflow | http://localhost:5050 |
| MinIO console | http://localhost:9001 |

The API starts even before a model is promoted: `/health` answers 200, `/ready` answers 503 until
`python -m trajclf.promote` has set `@production`, then `POST /model/reload` (or restart the container).

**Not enough disk space to build?** Use the images published by the CI instead:

```bash
IMAGE_NAMESPACE=colinernld IMAGE_TAG=latest docker compose pull api webapp
IMAGE_NAMESPACE=colinernld IMAGE_TAG=latest docker compose up -d --no-build
```

## Publishing to Docker Hub

Done automatically by GitHub Actions on every merge to `main` that touches the images. One-time setup:

1. Create a Docker Hub account (or an organisation) for the team. Ours: `colinernld`.
2. Docker Hub → Account settings → **Personal access tokens** → *Generate new token*, access *Read & Write*.
3. GitHub repo → Settings → Secrets and variables → Actions:
   - **Secrets**: `DOCKERHUB_USERNAME` (the Docker Hub user) and `DOCKERHUB_TOKEN` (the token).
   - **Variables** (optional): `DOCKERHUB_NAMESPACE` if images go to an organisation instead of the user.
4. Actions tab → *docker-images* → **Run workflow** (or merge any change to `main`).
5. If the namespace changes, update it in this page, the README and `k8s/kustomization.yaml`.

On pull requests the workflow only builds the images (amd64) and checks that the API and web app
containers start and answer their health check. Nothing is pushed.

## Building locally (optional)

```bash
docker build -f docker/api/Dockerfile -t ride-api .
docker build -f docker/webapp/Dockerfile -t ride-webapp .
# run it next to the Compose stack (MLflow only accepts the host names listed in MLFLOW_ALLOWED_HOSTS)
docker run --rm -p 8000:8000 --network ride-mlops_default -e MLFLOW_TRACKING_URI=http://mlflow:5000 ride-api
```
