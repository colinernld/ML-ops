# Production on Kubernetes

Production = **all** services on the local Kubernetes cluster of Docker Desktop, in the namespace
`ride-mlops`. Manifests: [`k8s/`](../k8s) (plain YAML + kustomize, no extra tool needed:
`kubectl apply -k` understands kustomize).

| Component | Kind | Storage | Exposed on |
|---|---|---|---|
| Postgres (MLflow metadata) | StatefulSet | 2 Gi volume | in-cluster only |
| MinIO (MLflow artifacts, S3) | StatefulSet | 5 Gi volume | console http://localhost:30901 |
| MLflow server + registry | Deployment | — | http://localhost:30550 |
| API (FastAPI) | Deployment | — | http://localhost:30800/docs |
| Web app (Streamlit) | Deployment | — | http://localhost:30851 |
| Prometheus (metrics, alerts) | Deployment | `emptyDir`, 7 days | http://localhost:30909 |
| Grafana (dashboard) | Deployment | — (provisioned) | http://localhost:30300 |
| Blackbox exporter (health checks) | Deployment | — | in-cluster only |

```
browser ──► webapp :30851 ──► api :30800 ──► mlflow :30550 ──► postgres
                                │                             └──────► minio (bucket "mlflow")
                                └── /metrics ◀── prometheus :30909 ◀── grafana :30300
```

Monitoring manifests live in [`monitoring/`](../monitoring) (included by `k8s/kustomization.yaml`), see
[monitoring.md](monitoring.md).

- Images come from Docker Hub (built by GitHub Actions, see [docker.md](docker.md)): nothing is built on the Mac.
- Passwords live in the Secret `ride-secrets`, created from your `.env` by `k8s/create-secrets.sh`, never in git.
- **Probes**: the API is *live* when the process answers (`/health`) and *ready* only when a model is
  loaded (`/ready`), so Kubernetes never sends traffic to an API without a model. Rolling updates keep the
  old pods until the new ones are ready (`maxUnavailable: 0`): zero-downtime deploys.
- Ports 30xxx are NodePorts: Docker Desktop exposes them on `localhost`, so production runs next to the
  Compose dev stack (ports 5050 / 8000 / 8501) without conflicts.

## First deployment (once, on the production Mac)

**0. Prerequisites**: Docker Desktop with Kubernetes enabled (Settings → Kubernetes → Enable, cluster type
kubeadm), at least 6 GB of memory (Settings → Resources), ~10 GB free disk. Check:

```bash
kubectl config use-context docker-desktop
kubectl get nodes          # docker-desktop   Ready
```

**1. Secrets** (from the repository root):

```bash
cp .env.example .env       # if not done yet, then change the passwords (letters and digits only)
./k8s/create-secrets.sh
```

**2. Deploy everything** (images from Docker Hub `colinernld`, set in `k8s/kustomization.yaml`).

```bash
kubectl apply -k k8s/
kubectl -n ride-mlops get pods -w      # wait until postgres, minio, mlflow and webapp are Running/Ready
```

The API pod stays `0/1 Ready`: normal, there is no model in the production MLflow yet.

**3. Train, register and promote a model inside the cluster** (no Python setup needed on the laptop).
`k8s/train-in-cluster.sh` runs a one-off Job with the API image (PyTorch + MLflow + trajclf), the code
of this repository and the raw data folder of this Mac mounted read-only. It validates the data, trains,
registers the model in the production MLflow, promotes it to `@production` if it beats the baselines,
then restarts the API. On an 8 GB Mac, pause the API during training to free memory:

```bash
kubectl -n ride-mlops scale deployment/api --replicas=0
./k8s/train-in-cluster.sh "/path/to/Project"     # folder with GBG500.parquet + GBG500_labels.csv
kubectl -n ride-mlops scale deployment/api --replicas=1
kubectl -n ride-mlops rollout status deployment/api     # 1/1 ready
```

Alternative from the laptop (needs `pip install -e ".[mlflow]"` and the data in `data/raw/`):
`MLFLOW_TRACKING_URI=http://localhost:30550 python -m trajclf.train --register-name ride-classifier`,
then `python -m trajclf.promote` and `kubectl -n ride-mlops rollout restart deployment/api`.

**4. Check**: http://localhost:30851 (web app), http://localhost:30800/docs (Swagger),
http://localhost:30550 (MLflow, model `ride-classifier` with alias `production`),
http://localhost:30300 (Grafana dashboard; send traffic with `python scripts/simulate_traffic.py --api http://localhost:30800`).

```bash
curl -F "file=@examples/ride_safe.csv" http://localhost:30800/predict/file
```

After that, deployments are automatic (see [cicd.md](cicd.md)).

## Everyday commands

```bash
kubectl -n ride-mlops get pods,svc                 # what runs
kubectl -n ride-mlops logs deploy/api -f           # API logs
kubectl -n ride-mlops rollout history deploy/api   # deployments (change-cause = image tag + commit)
kubectl -n ride-mlops rollout undo deploy/api      # roll back the API to the previous version
```

**New model** (after retraining): `python -m trajclf.promote` then
`kubectl -n ride-mlops rollout restart deployment/api`. **Model rollback**:
`python -m trajclf.promote --version <N>` then restart the API.

**Image rollback**: Actions → *cd* → *Run workflow* with tag `sha-<old commit>`, or `kubectl rollout undo`.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `CreateContainerConfigError` | Secret missing: run `./k8s/create-secrets.sh` |
| `ImagePullBackOff` | image not on Docker Hub yet, or wrong namespace in `k8s/kustomization.yaml` |
| API `0/1 Ready` | no model `@production` in MLflow: step 3 |
| `OOMKilled` (pod restarts) | not enough memory: Docker Desktop → Resources → Memory ≥ 6 GB, stop other containers, pause the API while training |
| MLflow `Invalid Host header` | add the host name to `MLFLOW_SERVER_ALLOWED_HOSTS` in `k8s/config.yaml` |
| `localhost:30xxx` does not answer | cluster type kind instead of kubeadm: use `kubectl -n ride-mlops port-forward svc/webapp 30851:8501` |
| Start from scratch | `kubectl delete namespace ride-mlops` (⚠️ deletes the MLflow data), then steps 1–3 |
