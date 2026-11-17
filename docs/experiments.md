# Experiments

All variants are compared with the **same 5-fold stratified cross-validation** (seed 42, 500 rides,
~100 test rides per fold). Only one parameter changes at a time compared with the reference run.
Reproduce any line with `python -m trajclf.evaluate` and the matching flags (set
`MLFLOW_TRACKING_URI` to log it in the `ride-classification-cv` MLflow experiment).

Baselines on the same folds: always "Safe" → balanced accuracy 0.20, macro-F1 0.18, accuracy 0.83;
random with class frequencies → 0.27 / 0.26 / 0.75.

| Run | Change | Balanced acc. | Macro-F1 | Accuracy | Decision |
|---|---|---|---|---|---|
| v0 | raw IMU axes, width 32, 20 epochs (hold-out split only) | 0.20 | 0.18 | 0.83 | ✗ predicts "Safe" for every test ride (memorises rides) |
| reference | orientation-robust features, width 16, 12 epochs | 0.44 ± 0.13 | 0.30 ± 0.14 | 0.56 ± 0.20 | first model above baselines |
| win512 | window 51.2 s | 0.37 ± 0.11 | 0.25 ± 0.04 | 0.44 ± 0.16 | ✗ fewer windows to learn from |
| win128 | window 12.8 s | 0.40 ± 0.06 | 0.27 ± 0.06 | 0.57 ± 0.14 | ✗ |
| w32 | width 32 (4× more parameters) | 0.39 ± 0.12 | 0.29 ± 0.11 | 0.67 ± 0.26 | ✗ overfits |
| lr3e-3 | learning rate 3e-3 | 0.42 ± 0.10 | 0.34 ± 0.10 | 0.66 ± 0.14 | ~ |
| **ep25** | **25 epochs** | **0.44 ± 0.17** | **0.36 ± 0.12** | **0.69 ± 0.08** | **✓ kept** (`params.yaml`) |
| ep40 | 40 epochs | 0.34 ± 0.04 | 0.30 ± 0.05 | 0.76 | ✗ drifts back towards "Safe" |
| pre_ep25 | encoder pre-trained on SRV3490, 25 epochs | 0.38 ± 0.09 | 0.33 ± 0.08 | 0.73 | ✗ see below |

## Why 25 epochs

Same balanced accuracy as the reference, but macro-F1 +0.06 and accuracy +0.13 with a much lower
spread across folds (accuracy std 0.08 vs 0.20): the model stops over-predicting "Safe overall".
At 40 epochs it overfits the 333 Safe training rides and falls back to predicting "Safe".

## Pre-training on SRV3490 (negative result)

Idea: learn generic riding features on the 3,490 SRV rides (7× more data) before fine-tuning.
Pretext tasks: predict the city (8 classes) and whether the ride is a crash ride, on 41,730 sampled
windows. The pretext tasks were barely learned (city accuracy 0.52 vs 0.42 for the majority city,
crash-ride accuracy 0.82 vs 0.80), and fine-tuning from this encoder lowered balanced accuracy.
The pretext labels are ride-level and mostly unrelated to riding style, so they do not transfer.
Not kept (the pre-training code was left out of the repository); a self-supervised objective (e.g. contrastive or masked-signal prediction) would be the
next thing to try.

## Limits

- With 6 Reckless rides there is 1 per test fold: Reckless recall is 0 or 1 per fold (0.2 ± 0.45
  overall). This number cannot be trusted until more Reckless rides are collected.
- Standard deviations across folds are large (±0.1 – 0.17): differences below ~0.05 between
  variants are not significant.
