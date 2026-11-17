# Model card: `ride-classifier`

## Overview

| | |
|---|---|
| Task | classify a full bike/scooter ride into Safe, Safe overall, Bad weather, Tandem, Reckless |
| Input | one ride of phone IMU data at 10 Hz: `ax, ay, az` (m/s²), `rx, ry, rz` (°/s), 50 – 36,000 samples |
| Output | predicted class + probability for each of the 5 classes |
| Model | 1D CNN, 3 conv blocks (16 → 32 → 64 filters), mean + max pooling, dropout 0.5, ~10.6k parameters |
| Training data | GBG500, 400 rides (stratified 80 % split), ~9,400 windows of 25.6 s |
| Code | `src/trajclf/` · hyper-parameters in `params.yaml` · experiments in [experiments.md](experiments.md) |

## How a prediction is made

1. **Validation** (`validation.py`): reject rides with missing columns, wrong length, > 20 % missing
   values, infinite or physically impossible values.
2. **Cleaning**: sort by time, linear interpolation of missing values.
3. **Features** (`features.py`): remove the ride's median per axis (phone orientation), add |a| and
   |r|, standardise with the training-set mean/std → 8 channels.
4. **Windows** of 256 samples (25.6 s), 50 % overlap; rides shorter than that are zero-padded.
5. **CNN** gives class probabilities per window; the **ride probability is the mean over windows**.

## Training

- Balanced sampling (each class seen equally often), noise + scaling augmentation, label smoothing 0.1.
- AdamW, lr 1e-3, weight decay 1e-3, batch 64, 25 epochs, seed 42. Trains in ~1 min on a laptop CPU.
- The normaliser is fitted on training rides only; train and test never share windows of the same ride.

## Evaluation

Because 83 % of rides are *Safe*, the model is judged on **balanced accuracy** and **macro-F1**, and
always compared with two baselines computed on the same data.

**5-fold cross-validation** (all 500 rides, mean ± std):

| | CNN | Always "Safe" | Random (class frequencies) |
|---|---|---|---|
| Balanced accuracy | **0.44 ± 0.17** | 0.20 | 0.27 ± 0.04 |
| Macro-F1 | **0.36 ± 0.12** | 0.18 | 0.26 ± 0.04 |
| Accuracy | 0.69 ± 0.08 | 0.83 | 0.75 |
| Reckless recall | 0.2 ± 0.45 (1 ride per fold) | 0 | 0 |

**Hold-out test set** (100 rides, model shipped in `artifacts/`): accuracy 0.80, balanced
accuracy 0.33, macro-F1 0.34.

| true \ pred | Safe | Safe overall | Bad weather | Tandem | Reckless |
|---|---|---|---|---|---|
| Safe | 76 | 1 | 4 | 2 | 0 |
| Safe overall | 6 | 2 | 0 | 0 | 0 |
| Bad weather | 3 | 0 | 1 | 0 | 0 |
| Tandem | 2 | 1 | 0 | 1 | 0 |
| Reckless | 0 | 1 | 0 | 0 | 0 |

## Limitations

- **Very small minority classes**: 6 Reckless, 18 Tandem, 22 Bad weather rides. Results vary a lot
  between folds; Reckless detection is not reliable yet.
- **Plain accuracy is below the "always Safe" baseline**: the model trades a few Safe rides for
  catching rare classes. Choose the metric according to the cost of a missed Reckless ride vs a false alarm.
- **One city**: all rides are from Gothenburg. Behaviour on other cities, phones or mounting
  positions is unknown (SRV3490 could be used to test this).
- **Gyroscope saturation** at ±300 °/s in the data: aggressive manoeuvres are clipped.

## Intended use

Course project (MLOps, Albert School). Flagging rides for review, not taking automatic decisions
about riders.

## Monitoring suggestions (for the monitoring part)

- Input drift: per-ride mean of |a| and |r|, share of missing values, ride length.
- Output drift: share of each predicted class over time (≈ 83 % Safe expected).
- Validation errors returned by the API (rejected rides).
