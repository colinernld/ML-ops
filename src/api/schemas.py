"""Request and response models (they also produce the Swagger documentation at /docs)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trajclf.validation import MAX_RIDE_SAMPLES

_N = 60  # the example must have at least 50 samples (5 s at 10 Hz) to pass validation
_EXAMPLE = {
    "ax": [0.49] * _N,
    "ay": [-9.03] * _N,
    "az": [-3.59] * _N,
    "rx": [0.87] * _N,
    "ry": [0.01] * _N,
    "rz": [0.07] * _N,
}

Channel = list[float | None]  # null = missing sample, filled by interpolation if < 20 % of the ride


class Ride(BaseModel):
    """One ride: the six IMU channels sampled at 10 Hz (one value per 100 ms), same length for all."""

    model_config = ConfigDict(json_schema_extra={"examples": [_EXAMPLE]})

    ax: Channel = Field(max_length=MAX_RIDE_SAMPLES, description="acceleration x (m/s²)")
    ay: Channel = Field(max_length=MAX_RIDE_SAMPLES, description="acceleration y (m/s²)")
    az: Channel = Field(max_length=MAX_RIDE_SAMPLES, description="acceleration z (m/s²)")
    rx: Channel = Field(max_length=MAX_RIDE_SAMPLES, description="rotation x (deg/s)")
    ry: Channel = Field(max_length=MAX_RIDE_SAMPLES, description="rotation y (deg/s)")
    rz: Channel = Field(max_length=MAX_RIDE_SAMPLES, description="rotation z (deg/s)")
    time_index: list[float] | None = Field(
        default=None, max_length=MAX_RIDE_SAMPLES, description="optional timestamps (ms), used to sort the samples"
    )

    @model_validator(mode="after")
    def same_length(self) -> Ride:
        lengths = {name: len(getattr(self, name)) for name in ("ax", "ay", "az", "rx", "ry", "rz")}
        if self.time_index is not None:
            lengths["time_index"] = len(self.time_index)
        if len(set(lengths.values())) > 1:
            raise ValueError(f"all channels must have the same length, got {lengths}")
        return self


class ModelInfoOut(BaseModel):
    source: str = Field(description="'mlflow' (model registry) or 'local' (MODEL_DIR)")
    uri: str
    name: str | None = None
    version: str | None = Field(default=None, description="registry version actually loaded")
    alias: str | None = None
    run_id: str | None = None
    loaded_at: str
    classes: list[str]
    window: int = Field(description="samples per window")
    stride: int


class Prediction(BaseModel):
    label: str = Field(description="predicted class")
    probabilities: dict[str, float] = Field(description="probability per class (average over the ride's windows)")
    confidence: float = Field(description="probability of the predicted class")
    n_samples: int
    n_windows: int = Field(description="number of 25.6 s windows scored")
    model_version: str | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "label": "Safe",
                    "probabilities": {
                        "Safe": 0.81,
                        "Safe overall": 0.1,
                        "Bad weather": 0.04,
                        "Tandem": 0.03,
                        "Reckless": 0.02,
                    },
                    "confidence": 0.81,
                    "n_samples": 2300,
                    "n_windows": 17,
                    "model_version": "3",
                }
            ]
        }
    )


class Health(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
    error: str | None = None


class Error(BaseModel):
    detail: str
