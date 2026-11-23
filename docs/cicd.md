# CI/CD

Four GitHub Actions workflows in [`.github/workflows/`](../.github/workflows):

```
pull request ──► ci (lint + tests + manifests + monitoring) ─┐
             └─► docker-images (build + smoke test)          ├─► review + approval ─► merge to main
                                                             ┘
main ──► ci
     └─► docker-images: build amd64+arm64, push to Docker Hub (latest, sha-<commit>)
              └─► cd (self-hosted runner on the production Mac):
                  kubectl apply -k k8s/ → images sha-<commit> → rollout → smoke test → rollback if it fails

manual / weekly / drift ──► retrain (self-hosted runner): data → validate → train → register
                            → promote only if better → API restarted on the new @production version
```

| Workflow | Runs on | Does |
|---|---|---|
| `ci.yml` | GitHub (ubuntu) | `ruff check`, `pytest` (synthetic data, no dataset needed), hadolint on every Dockerfile, kustomize + kubeconform on `k8s/` (monitoring included), `promtool` on the Prometheus configs and alert-rule unit tests, `docker compose config` |
| `docker-images.yml` | GitHub (ubuntu) | PR: build + start the API and web app containers and hit their health checks. `main`: multi-arch build and push to Docker Hub ([docker.md](docker.md)) |
| `cd.yml` | **self-hosted runner** (production Mac) | deploy the images of the merged commit to Kubernetes, wait for the rollout, smoke-test the API (`/ready` + a real prediction), web app and MLflow, **roll back automatically** if anything fails; then check Prometheus and Grafana (reported, never blocking) |
| `retrain.yml` | **self-hosted runner** (production Mac) | retraining pipeline: see [below](#retraining-workflow) |

`cd.yml` also runs when only `k8s/` or `monitoring/` changes (re-applies the manifests, keeps the running
images) and manually (Actions → *cd* → *Run workflow*, tag `sha-<commit>`) to roll back to any published version.
It never runs on pull requests: PR code must not execute on the production machine.

## One-time setup (repository owner)

**1. Docker Hub secrets** — see [docker.md](docker.md#publishing-to-docker-hub):
`DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN` (secrets), `DOCKERHUB_NAMESPACE` (variable, optional).

**2. Required checks**: Settings → Rules → Rulesets → `protect-main` → *Require status checks to pass* →
add `lint-and-test` and `manifests (Dockerfiles + Kubernetes)`. A PR can then only be merged when the
tests pass. (Do not make the `docker-images` jobs required: they only run when a PR touches the images,
and a required check that never runs blocks the PR.)

**3. Self-hosted runner on the production Mac** (the one running Docker Desktop + Kubernetes,
[kubernetes.md](kubernetes.md) done first):

1. GitHub repo → Settings → Actions → Runners → **New self-hosted runner** → macOS, ARM64 (Apple Silicon).
2. In a terminal, run the *Download* and *Configure* commands GitHub shows, in a folder **outside** the
   repository (e.g. `~/actions-runner`). Accept the default name and labels (`self-hosted`, `macOS`, `ARM64`).
3. Start it: `./run.sh` (stops when you close the terminal), or as a background service that starts with
   the Mac: `./svc.sh install && ./svc.sh start`.
4. The runner appears as **Idle** in Settings → Actions → Runners.

The runner uses the Mac's `kubectl` and its `docker-desktop` context: Docker Desktop must be running for
a deployment to work. If the Mac is off, the `cd` job waits in the queue and runs when the runner is back.

**4. Fork pull requests**: Settings → Actions → General → *Fork pull request workflows* → require approval
for all outside collaborators (the repository is public and has a self-hosted runner).

## Check it works

1. Merge any change to `main` that touches the images (or run *docker-images* manually).
2. Actions tab: `docker-images` (3 green jobs) → `cd` starts automatically → green.
3. `kubectl -n ride-mlops rollout history deploy/api` shows the new revision with its commit.
4. Take screenshots of the Actions runs and of the deployed services for the README.

## Retraining workflow

[`retrain.yml`](../.github/workflows/retrain.yml) is the production version of the training pipeline: it gets
the data, retrains, and redeploys, with the same steps as the Airflow DAG ([pipeline.md](pipeline.md)).

```
Get the data (RAW_DATA_DIR on the production Mac)
  └─ Kubernetes Job (k8s/train-in-cluster.sh, API image + repository code):
       validate data (stops on bad data) → clean + split → train → register a new version in MLflow
       → promote to @production only if it beats the baselines AND the current production version
  └─ restart the API (it loads models:/ride-classifier@production) → smoke test → summary (promoted or not)
```

| Trigger | How |
|---|---|
| Manual | Actions → *retrain* → *Run workflow*. Tick *pause_api* on an 8 GB Mac (frees memory, a few minutes of downtime). |
| Weekly | Monday 04:00 UTC, only if the repository variable `RETRAIN_ON_SCHEDULE` is `true` (off by default: the production Mac must be on). |
| Drift | `repository_dispatch` event of type `retrain`, e.g. from an alert webhook ([monitoring.md](monitoring.md)): `curl -X POST -H "Authorization: Bearer <token>" https://api.github.com/repos/colinernld/ML-ops/dispatches -d '{"event_type":"retrain"}'` |

One-time setup: Settings → Secrets and variables → Actions → **Variables** → `RAW_DATA_DIR` = folder of the
production Mac with `GBG500.parquet` and `GBG500_labels.csv` (the data is never in git). It shares the
`cd-production` concurrency group with `cd.yml`: a retraining never runs during a deployment, and vice versa.
A model that is not better is a normal outcome: production stays on the current version and the reason is
written on the new version in MLflow (tag `promotion`).
