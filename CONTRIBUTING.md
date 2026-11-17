# Contributing

Group of 5, one part each. `main` is protected: every change goes through a pull request
reviewed by another member.

## Who does what

| Part | Branches | Main folders |
|---|---|---|
| Data + model | `feat/data-*`, `feat/model-*` | `src/trajclf/`, `docs/` |
| MLflow + Minio | `feat/mlflow-*` | `docker-compose.yml`, `.env.example` |
| API + web app | `feat/api-*`, `feat/webapp-*` | `src/api/`, `webapp/` |
| Docker / Kubernetes / CI-CD | `feat/docker-*`, `feat/k8s-*`, `feat/ci-*` | `docker/`, `k8s/`, `.github/workflows/` |
| Monitoring / retraining / docs | `feat/monitoring-*`, `feat/retrain-*`, `docs/*` | `monitoring/`, `README.md` |

## Workflow

```bash
git checkout main && git pull                 # always start from an up-to-date main
git checkout -b feat/api-predict-endpoint     # one branch per task
# ... code, small commits with clear messages ...
ruff check . && pytest                        # must pass before pushing
git push -u origin feat/api-predict-endpoint
```

Then open a pull request on GitHub (the template appears automatically), link the issue
(`Closes #12`) and ask one teammate for a review.

## Rules

- **Never push directly to `main`.** One approval is required to merge.
- **Small PRs** (one feature, ideally < 400 lines) are reviewed faster.
- **Review within 24 h** when you are asked: run the "How to test it" commands, comment, approve.
- **Squash and merge**, then delete the branch.
- If `main` moved while you were working: `git pull origin main` on your branch, fix conflicts, push.
- **Never commit** data (`data/`), trained models (`artifacts/`, `mlruns/`) or secrets (`.env`).
  Put example values in `.env.example`.

## Commit messages

Short, imperative, prefixed by the part: `api: add /predict endpoint`, `k8s: add api deployment`,
`model: tune window size`, `docs: add MLflow screenshots`.
