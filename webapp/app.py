"""Streamlit web app to test the model through the API.

    API_URL=http://localhost:8000 streamlit run webapp/app.py      # → http://localhost:8501

Thin client on purpose: no model and no PyTorch here. Every prediction goes through the API, so the web
app always shows what production answers (same model version, same validation rules).
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
EXAMPLES_DIR = Path(os.getenv("EXAMPLES_DIR", Path(__file__).resolve().parents[1] / "examples"))
TIMEOUT = 30

st.set_page_config(page_title="Ride classifier", page_icon="🚲", layout="wide")


@st.cache_data(ttl=10)
def api_status() -> tuple[dict | None, dict | None, str | None]:
    """(health, model info, error). Cached a few seconds to avoid calling the API on every rerun."""
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        model = requests.get(f"{API_URL}/model", timeout=5)
        return health, (model.json() if model.ok else None), None
    except requests.RequestException as exc:
        return None, None, str(exc)


def predict(name: str, content: bytes) -> tuple[dict | None, str | None]:
    try:
        r = requests.post(f"{API_URL}/predict/file", files={"file": (name, content, "text/csv")}, timeout=TIMEOUT)
    except requests.RequestException as exc:
        return None, f"API unreachable: {exc}"
    if r.ok:
        return r.json(), None
    try:
        detail = r.json().get("detail", r.text)
    except ValueError:
        detail = r.text
    return None, f"HTTP {r.status_code}: {detail}"


# ---------- sidebar: which API / model ----------
with st.sidebar:
    st.header("API")
    st.code(API_URL)
    health, model, error = api_status()
    if error:
        st.error(f"Unreachable: {error}")
    elif model:
        st.success(f"Model v{model['version']} ({model['source']})" if model.get("version") else "Local model")
        st.caption(f"Alias: {model.get('alias') or '–'} · loaded {model['loaded_at']}")
    else:
        st.warning("API up, no model loaded yet")
    st.markdown(f"[Swagger]({API_URL}/docs) · [Metrics]({API_URL}/metrics)")

# ---------- main: pick a ride ----------
st.title("🚲 Ride classifier")
st.write(
    "Classifies a bike/scooter ride from the phone's accelerometer and gyroscope (10 Hz) into "
    "**Safe, Safe overall, Bad weather, Tandem, Reckless**. Upload a CSV with columns "
    "`ax, ay, az, rx, ry, rz` (optional `time_index`), or try a sample ride."
)

examples = sorted(EXAMPLES_DIR.glob("*.csv")) if EXAMPLES_DIR.exists() else []
source = st.radio("Ride", ["Upload a CSV", "Sample ride"] if examples else ["Upload a CSV"], horizontal=True)

name, content = None, None
if source == "Sample ride":
    chosen = st.selectbox("Sample", examples, format_func=lambda p: p.name)
    name, content = chosen.name, chosen.read_bytes()
else:
    uploaded = st.file_uploader("CSV file", type="csv")
    if uploaded:
        name, content = uploaded.name, uploaded.getvalue()

if content is None:
    st.stop()

# Raw signal preview (display only: no computation in the UI, the API does all the processing)
try:
    preview = pd.read_csv(io.BytesIO(content))
    left, right = st.columns(2)
    acc = [c for c in ("ax", "ay", "az") if c in preview]
    rot = [c for c in ("rx", "ry", "rz") if c in preview]
    if acc:
        left.caption("Accelerometer (m/s²)")
        left.line_chart(preview[acc], height=220)
    if rot:
        right.caption("Gyroscope (deg/s)")
        right.line_chart(preview[rot], height=220)
    st.caption(f"{len(preview):,} samples ≈ {len(preview) / 600:.1f} min at 10 Hz")
except ValueError as exc:  # unreadable CSV (parser/encoding errors); the API gives the real error
    st.info(f"No preview: {exc}")

if st.button("Classify this ride", type="primary"):
    with st.spinner("Asking the API..."):
        result, err = predict(name, content)
    if err:
        st.error(err)
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction", result["label"])
        c2.metric("Confidence", f"{result['confidence']:.0%}")
        c3.metric("Windows scored", result["n_windows"])
        probs = pd.Series(result["probabilities"], name="probability").sort_values(ascending=False)
        st.bar_chart(probs, horizontal=True, height=240)
        st.caption(f"Model version: {result.get('model_version') or 'local'}")
        with st.expander("Raw API response"):
            st.json(result)
