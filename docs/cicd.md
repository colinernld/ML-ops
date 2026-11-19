# CI/CD

Three GitHub Actions workflows in [`.github/workflows/`](../.github/workflows):

```
pull request ──► ci (lint + tests + manifests) ─┐
             └─► docker-images (build + smoke test)├─► review + approval ─► merge to main
                                                    ┘
main ──► ci
     └─► docker-images: build amd64+arm64, push to Docker Hub (latest, sha-<commit>)
              └─► cd (self-hosted runner on the production Mac):
                  kubectl apply -k k8s/ → images sha-<commit> → rollout → smoke test → rollback if it fails
```

| Workflow | Runs on | Does |
|---|---|---|
| `ci.yml` | GitHub (ubuntu) | `ruff check`, `pytest` (synthetic data, no dataset needed), hadolint on every Dockerfile, kustomize + kubeconform on `k8s/` |
| `docker-images.yml` | GitHub (ubuntu) | PR: build + start the API and web app containers and hit their health checks. `main`: multi-arch build and push to Docker Hub ([docker.md](docker.md)) |
| `cd.yml` | **self-hosted runner** (production Mac) | deploy the images of the merged commit to Kubernetes, wait for the rollout, smoke-test the API (`/ready` + a real prediction), web app and MLflow, **roll back automatically** if anything fails |

`cd.yml` also runs when only `k8s/` changes (re-applies the manifests, keeps the running images) and
manually (Actions → *cd* → *Run workflow*, tag `sha-<commit>`) to roll back to any published version.
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
