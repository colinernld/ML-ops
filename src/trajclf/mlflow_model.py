"""MLflow pyfunc wrapper so the model can be registered and served from the Model Registry.

Input: a DataFrame with columns ax, ay, az, rx, ry, rz (one ride), optionally a `ride_id`
column to score several rides at once. Output: one row per ride with label + probabilities.
"""

from __future__ import annotations

import mlflow.pyfunc
import pandas as pd


class RidePyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from trajclf.predict import RideClassifier

        self.clf = RideClassifier.load(context.artifacts["model_dir"])

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        groups = model_input.groupby("ride_id", sort=False) if "ride_id" in model_input else [(None, model_input)]
        rows = []
        for ride_id, ride in groups:
            out = self.clf.predict(ride)
            rows.append({"ride_id": ride_id, "label": out["label"], **out["probabilities"]})
        return pd.DataFrame(rows)
