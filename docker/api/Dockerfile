# Ride classification API (FastAPI + PyTorch + MLflow client).
#   docker build -f docker/api/Dockerfile -t ride-api .        (build context = repository root)
# Configuration by environment variables only, see src/api/settings.py and docs/docker.md.
FROM python:3.12-slim

# Same PyTorch as the environment that trained the registered model (issue #7: results depend on it).
ARG TORCH_VERSION=2.14.1
ARG MLFLOW_VERSION=3.18.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 1. Heavy dependencies first, in their own layer (rebuilt only when versions change).
#    CPU-only wheel: several GB smaller than the CUDA build, the API never uses a GPU.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}"

# 2. Project code + its API dependencies.
COPY pyproject.toml ./
COPY src/ src/
# hadolint ignore=DL3013
RUN pip install ".[mlflow,api]" "mlflow==${MLFLOW_VERSION}" \
    && useradd --create-home --uid 10001 app

# 3. Run as an unprivileged user (numeric uid for Kubernetes runAsNonRoot; MLflow caches models in $HOME).
USER 10001

# Defaults for Compose and Kubernetes (Service name "mlflow"); override with -e / env.
# Few MLflow HTTP retries: if MLflow is down the API starts anyway (/health 200, /ready 503) instead of
# blocking its startup for minutes. Without a tracking URI MLflow would try to create a local SQLite
# database in a read-only folder and retry for minutes, so a URI is always set.
ENV MODEL_URI=models:/ride-classifier@production \
    MLFLOW_TRACKING_URI=http://mlflow:5000 \
    MLFLOW_HTTP_REQUEST_MAX_RETRIES=2 \
    MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR=1 \
    MLFLOW_HTTP_REQUEST_TIMEOUT=60
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
