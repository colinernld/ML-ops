# MLOps project: ride classification from IMU signals

Classify bike/scooter rides into 5 categories (**Safe, Safe overall, Bad weather, Tandem, Reckless**)
from the phone's accelerometer and gyroscope, with a full MLOps pipeline around the model.

> This README covers the **data + model** part. The other parts (MLflow/Minio, API + web app,
> Docker/K8s/CI-CD, monitoring) plug into the interfaces described in [Interfaces for the team](#interfaces-for-the-team).

## Data

Download `Project.zip (data)` from Inside (Materials → Project) and unzip it into `data/raw/`:

```
data/raw/GBG500.parquet        # 500 rides, IMU signal every 100 ms
data/raw/GBG500_labels.csv     # RIDE_ID, LABEL
data/raw/SRV3490.parquet       # (not used yet) 3490 rides with crash/topple events
```

| | |
|---|---|
| Signals | `ax, ay, az` (acceleration, m/s²), `rx, ry, rz` (rotation, °/s) at 10 Hz |
| Ride length | 1,807 – 4,767 samples (3 – 8 min) |
| Missing values | 10,201 rows (0.7 %) spread over 439 rides → linear interpolation |
| Classes | Safe 416 · Safe overall 38 · Bad weather 22 · Tandem 18 · **Reckless 6** |

The dataset is **very imbalanced**: always answering "Safe" gives 83 % accuracy. We therefore judge
the model on **balanced accuracy** and **macro-F1**, and always compare it to two baselines
(always "Safe", and random guessing with the class frequencies).

## Quick start

```bash
pip install -e ".[dev,mlflow]"

python -m trajclf.data        # clean + stratified 80/20 split → data/processed/
python -m trajclf.train       # train CNN, evaluate on test rides → artifacts/
python -m trajclf.evaluate    # 5-fold cross-validation → artifacts/cv_results.csv
pytest                        # unit tests
```

With an MLflow server running, the same training run logs params, metrics, reports and a
registered pyfunc model:

```bash
MLFLOW_TRACKING_URI=http://localhost:5000 python -m trajclf.train --register-name ride-classifier
```

## Approach

1. **Cleaning** (`data.py`): sort by `time_index`, drop duplicated timestamps, interpolate missing values.
2. **Split** at **ride level**, stratified by label, so windows of one ride never end up in both train and test.
3. **Preprocessing** (`features.py`): every phone is mounted differently, so raw axes let the model
   memorise rides (gravity sits on `ay` for one ride and is spread across axes for another).
   We subtract each ride's median per channel and add the norms of acceleration and rotation, then
   standardise with statistics computed on the training rides only. Result: 8 channels.
4. **Windows**: 256 samples (25.6 s) with 50 % overlap → ~9,400 training windows.
5. **Model** (`model.py`): 1D CNN (3 conv blocks → mean+max pooling → dropout → linear), ~10.6k parameters.
6. **Imbalance**: windows are sampled so that each class is seen equally often; noise and scaling
   augmentation; label smoothing.
7. **Ride prediction**: average of the window probabilities, then argmax.

## Results

5-fold cross-validation (500 rides, mean ± std):

| Metric | CNN | Always "Safe" | Random (class frequencies) |
|---|---|---|---|
| Balanced accuracy | **0.44 ± 0.13** | 0.20 | 0.27 ± 0.04 |
| Macro-F1 | **0.30 ± 0.14** | 0.18 | 0.26 ± 0.04 |
| Accuracy | 0.56 ± 0.20 | 0.83 | 0.75 |

The model beats both baselines on the metrics that matter for imbalanced data. Its plain accuracy is
lower because it now dares to predict the rare classes (it over-predicts "Safe overall").
With only 6 Reckless rides (1 per test fold), Reckless recall is not reliable yet (0.2 ± 0.45).

Hold-out test set (100 rides) confusion matrix:

| true \ pred | Safe | Safe overall | Bad weather | Tandem | Reckless |
|---|---|---|---|---|---|
| Safe | 59 | 18 | 1 | 5 | 0 |
| Safe overall | 1 | 7 | 0 | 0 | 0 |
| Bad weather | 1 | 2 | 1 | 0 | 0 |
| Tandem | 1 | 2 | 0 | 0 | 1 |
| Reckless | 0 | 1 | 0 | 0 | 0 |

**Next ideas**: pre-train on SRV3490 (3,490 rides, crash detection), tune window size, add
spectral features, treat Reckless as anomaly detection (the dataset notes describe it this way).

## Interfaces for the team

**Model artifacts** (`artifacts/`, or the `reports/` folder of the MLflow run):
`model.pt` (weights) + `model_config.json` (classes, window, normaliser) + metrics/report files.

**Python inference** (for the API):

```python
from trajclf.predict import RideClassifier

clf = RideClassifier.load("artifacts")          # or a folder downloaded from MLflow / Minio
clf.predict(df)  # df: one ride, columns ax, ay, az, rx, ry, rz (time_index optional)
# {"label": "Safe", "probabilities": {"Safe": 0.37, ...}, "n_windows": 15}
```

Missing values are interpolated, and rides shorter than 25.6 s are zero-padded.

**MLflow pyfunc** (registered as `ride-classifier`):

```python
import mlflow
model = mlflow.pyfunc.load_model("models:/ride-classifier/1")
model.predict(df)   # add a `ride_id` column to score several rides at once
```

**Example inputs** for the API / web app: `examples/ride_safe.csv`, `examples/ride_tandem.csv`.

## Repository layout

```
src/trajclf/
  config.py        constants (signal columns, classes, window size, paths)
  data.py          load, clean, split          → python -m trajclf.data
  features.py      preprocessing, windows, normaliser
  model.py         1D CNN
  train.py         training + evaluation + MLflow logging → python -m trajclf.train
  evaluate.py      k-fold cross-validation     → python -m trajclf.evaluate
  predict.py       RideClassifier (inference entry point)
  mlflow_model.py  MLflow pyfunc wrapper
tests/             unit tests (pytest)
examples/          sample rides as CSV
```
