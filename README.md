# MLOps project: ride classification from IMU signals

Classify bike/scooter rides into 5 categories (**Safe, Safe overall, Bad weather, Tandem, Reckless**)
from the phone's accelerometer and gyroscope, with a full MLOps pipeline around the model.

> This README covers the **data + model** part. The other parts (MLflow/Minio, API + web app,
> Docker/K8s/CI-CD, monitoring) plug into the interfaces described in [Interfaces for the team](#interfaces-for-the-team).

| Doc | Content |
|---|---|
| [CLAUDE.md](CLAUDE.md) | team status, course requirements, interfaces, open items (also read by AI assistants) |
| [docs/data.md](docs/data.md) | dataset, exploration figures, data validation rules |
| [docs/model_card.md](docs/model_card.md) | model, training, evaluation, limitations |
| [docs/experiments.md](docs/experiments.md) | every variant tried and why the final one was kept |
| [docs/mlflow.md](docs/mlflow.md) | MLflow + MinIO stack: start it, log runs, register and load the model |

## Quick start

1. Download `Project.zip (data)` from Inside (Materials → Project) and unzip it into `data/raw/`
   (`GBG500.parquet`, `GBG500_labels.csv`, `SRV3490.parquet`).
2. Install and run:

```bash
pip install -e ".[dev,mlflow]"

python -m trajclf.data        # validate + clean + stratified split → data/processed/
python -m trajclf.train       # train CNN, evaluate on hold-out rides → artifacts/
python -m trajclf.evaluate    # 5-fold cross-validation → artifacts/cv_results.csv
python scripts/eda.py         # exploration figures → docs/figures/
pytest                        # 22 tests
```

Hyper-parameters live in [`params.yaml`](params.yaml); command-line flags override them
(`python -m trajclf.evaluate --epochs 40 --run-name more-epochs`).

With the MLflow stack running, training logs params, metrics, reports and registers the model;
cross-validation runs are logged in a separate experiment to compare variants
(details in [docs/mlflow.md](docs/mlflow.md)):

```bash
cp .env.example .env && docker compose up -d   # MLflow UI http://localhost:5050, MinIO http://localhost:9001
export MLFLOW_TRACKING_URI=http://localhost:5050
python -m trajclf.train --register-name ride-classifier    # experiment "ride-classification"
python -m trajclf.evaluate --run-name baseline             # experiment "ride-classification-cv"
python -m trajclf.promote                                  # newest version → @production if it beats baselines + production
```

## Pipeline

```
raw parquet + labels
  └─ validation.py   schema, 10 Hz sampling, missing values, physical ranges, labels → stops on failure
  └─ data.py         sort, dedup timestamps, interpolate, stratified split by ride
  └─ features.py     remove phone orientation, add |a| and |r|, standardise, 25.6 s windows
  └─ model.py        1D CNN (~10.6k parameters)
  └─ train.py        balanced sampling + augmentation, ride score = mean over windows, MLflow logging
  └─ evaluate.py     5-fold CV vs "always Safe" and random baselines
  └─ predict.py      RideClassifier: validated inference on one ride (used by the API)
```

## Results

The dataset is very imbalanced (83 % *Safe*, 6 *Reckless*), so always answering "Safe" already gets
83 % accuracy. The model is judged on **balanced accuracy** and **macro-F1** against two baselines.

5-fold cross-validation (500 rides, mean ± std):

| Metric | CNN | Always "Safe" | Random (class frequencies) |
|---|---|---|---|
| Balanced accuracy | **0.44 ± 0.17** | 0.20 | 0.27 ± 0.04 |
| Macro-F1 | **0.36 ± 0.12** | 0.18 | 0.26 ± 0.04 |
| Accuracy | 0.69 ± 0.08 | 0.83 | 0.75 |

The model beats both baselines on the metrics that matter for imbalanced data. Reckless recall
(0.2 ± 0.45, one Reckless ride per fold) is not reliable yet. Details and limits in the
[model card](docs/model_card.md); the 9 variants compared are in [experiments](docs/experiments.md).

## Interfaces for the team

**Model artifacts** (`artifacts/`, or the `reports/` folder of the MLflow run):
`model.pt` (weights) + `model_config.json` (classes, window, normaliser) + metrics/report files.
They are not in Git: run `python -m trajclf.train` or download them from MLflow.

**Python inference** (for the API):

```python
from trajclf.predict import RideClassifier
from trajclf.validation import DataValidationError

clf = RideClassifier.load("artifacts")   # or a folder downloaded from MLflow / Minio
try:
    clf.predict(df)   # one ride: columns ax, ay, az, rx, ry, rz (time_index optional), 10 Hz
    # {"label": "Safe", "probabilities": {"Safe": 0.81, ...}, "n_windows": 23}
except DataValidationError as err:   # bad input → return HTTP 422 with str(err)
    ...
```

**MLflow pyfunc** (registered as `ride-classifier`):

```python
import mlflow
model = mlflow.pyfunc.load_model("models:/ride-classifier@production")   # or a version: models:/ride-classifier/1
model.predict(df)   # add a `ride_id` column to score several rides at once
```

**Example inputs** for the API / web app: `examples/ride_safe.csv`, `examples/ride_tandem.csv`.

**Monitoring**: suggested drift metrics are listed at the end of the [model card](docs/model_card.md).

## Repository layout

```
params.yaml        hyper-parameters (single source of truth)
docker-compose.yml MLflow server + MinIO + Postgres   → docker compose up -d
.env.example       settings for docker-compose (copy to .env)
docker/mlflow/     MLflow server image
src/trajclf/
  config.py        constants, paths, params loader
  validation.py    data quality checks (dataset + single ride)
  data.py          validate, clean, split           → python -m trajclf.data
  features.py      preprocessing, windows, normaliser
  model.py         1D CNN
  train.py         training + evaluation + MLflow    → python -m trajclf.train
  evaluate.py      k-fold cross-validation           → python -m trajclf.evaluate
  predict.py       RideClassifier (inference entry point)
  mlflow_model.py  MLflow pyfunc wrapper
  promote.py       moves the @production alias in the registry  → python -m trajclf.promote
scripts/eda.py     exploration figures and stats
scripts/check_mlflow.py  checks the MLflow + MinIO stack end to end
tests/             unit, validation and end-to-end tests (pytest)
docs/              data, model card, experiments, MLflow, figures
examples/          sample rides as CSV
```
