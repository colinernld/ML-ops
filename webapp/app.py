"""Streamlit web app to test the model through the API.

    API_URL=http://localhost:8000 streamlit run webapp/app.py      # → http://localhost:8501

Thin client on purpose: no model and no PyTorch here. Every prediction goes through the API, so the web
app always shows what production answers (same model version, same validation rules). The only
computations done here are for display (signal magnitudes for the charts).
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
API_PUBLIC_URL = os.getenv("API_PUBLIC_URL", API_URL).rstrip("/")  # link shown to the browser (Swagger)
MLFLOW_UI_URL = os.getenv("MLFLOW_UI_URL", "")
EXAMPLES_DIR = Path(os.getenv("EXAMPLES_DIR", Path(__file__).resolve().parents[1] / "examples"))
TIMEOUT = 30
SAMPLING_HZ = 10

# Plain-language meaning of each class, shown with the verdict.
CLASSES = {
    "Safe": ("🟢", "Calm, normal riding.", "success"),
    "Safe overall": ("🟢", "Mostly calm riding, with a few unusual moments.", "success"),
    "Bad weather": ("🌧️", "Riding in rain or wind: more vibrations, but careful riding.", "info"),
    "Tandem": ("👥", "Two people on the bike or scooter: a heavier, steadier ride.", "info"),
    "Reckless": ("🔴", "Dangerous riding: sharp turns and sudden accelerations.", "error"),
}
# Median of each indicator over the Safe rides of the training data (docs/data.md).
TYPICAL_SAFE = {"turning": 15.5, "shaking": 0.93}

BLUE, GREY = "#2a78d6", "#c9c8c2"

st.set_page_config(page_title="Ride safety checker", page_icon="🚲", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1150px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- API calls
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
        return None, f"The API cannot be reached: {exc}"
    if r.ok:
        return r.json(), None
    try:
        detail = r.json().get("detail", r.text)
    except ValueError:
        detail = r.text
    return None, f"The API refused this ride (HTTP {r.status_code}): {detail}"


# ---------------------------------------------------------------- demo rides
@st.cache_data
def demo_rides() -> dict[str, dict]:
    """{file name: {"label": true class, ...}} for the sample rides shipped in examples/."""
    meta_file = EXAMPLES_DIR / "demo_rides.json"
    meta = json.loads(meta_file.read_text()).get("rides", {}) if meta_file.exists() else {}
    files = sorted(EXAMPLES_DIR.glob("*.csv")) if EXAMPLES_DIR.exists() else []
    rides = {f.name: meta.get(f.name, {}) for f in files}
    order = list(CLASSES)  # Safe first, Reckless last
    return dict(
        sorted(rides.items(), key=lambda kv: order.index(kv[1]["label"]) if kv[1].get("label") in order else 99)
    )


# ---------------------------------------------------------------- display helpers
def ride_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Orientation-free signals for the charts: how much the phone shakes and turns (1 s rolling mean)."""
    out = pd.DataFrame(index=df.index)
    out["minute"] = np.arange(len(df)) / (SAMPLING_HZ * 60)
    acc = np.sqrt((df[["ax", "ay", "az"]] ** 2).sum(axis=1))
    out["shaking"] = (acc - acc.rolling(SAMPLING_HZ * 5, min_periods=1, center=True).mean()).abs()
    out["turning"] = np.sqrt((df[["rx", "ry", "rz"]] ** 2).sum(axis=1))
    smooth = out[["shaking", "turning"]].rolling(SAMPLING_HZ, min_periods=1, center=True).mean()
    out[["shaking", "turning"]] = smooth
    return out


def step(number: int, title: str) -> None:
    st.markdown(
        f"<div style='color:{BLUE};font-weight:700;font-size:.8rem;letter-spacing:.05em'>STEP {number}</div>"
        f"<div style='font-size:1.6rem;font-weight:600;margin-bottom:.5rem'>{title}</div>",
        unsafe_allow_html=True,
    )


def line(data: pd.DataFrame, field: str, title: str, unit: str, typical: float | None) -> alt.Chart:
    base = alt.Chart(data).encode(x=alt.X("minute:Q", title="time (minutes)", axis=alt.Axis(grid=False)))
    chart = base.mark_line(color=BLUE, strokeWidth=1.5).encode(
        y=alt.Y(f"{field}:Q", title=unit),
        tooltip=[alt.Tooltip("minute:Q", format=".1f", title="minute"), alt.Tooltip(f"{field}:Q", format=".1f")],
    )
    if typical is not None:
        rule = alt.Chart(pd.DataFrame({"y": [typical]})).mark_rule(color="#52514e", strokeDash=[4, 4]).encode(y="y:Q")
        chart = chart + rule
    subtitle = "dashed line: typical safe ride" if typical is not None else ""
    return chart.properties(title=alt.TitleParams(title, subtitle=subtitle, anchor="start"), height=200)


def probability_chart(probs: dict[str, float], predicted: str) -> alt.Chart:
    data = pd.DataFrame({"class": list(CLASSES), "probability": [probs.get(c, 0.0) for c in CLASSES]})
    data["predicted"] = data["class"] == predicted
    data["text"] = data["probability"].map(lambda p: f"{p:.0%}")
    bars = (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=4, height=22)
        .encode(
            y=alt.Y("class:N", sort=list(CLASSES), title=None, axis=alt.Axis(labelFontSize=13)),
            x=alt.X("probability:Q", title=None, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            color=alt.condition("datum.predicted", alt.value(BLUE), alt.value(GREY)),
            tooltip=[alt.Tooltip("class:N"), alt.Tooltip("probability:Q", format=".1%")],
        )
    )
    text = bars.mark_text(align="left", dx=6, fontSize=13).encode(text="text:N", color=alt.value("#0b0b0b"))
    return (bars + text).properties(height=200)


# ---------------------------------------------------------------- sidebar: system status
with st.sidebar:
    st.subheader("System status")
    health, model, error = api_status()
    if error:
        st.error("🔴 API offline")
        st.caption(error)
    elif model:
        version = model.get("version")
        st.success(f"🟢 API online · model v{version}" if version else "🟢 API online · local model")
        if model.get("alias"):
            st.caption(f"Serving the `{model['alias']}` model from the MLflow registry")
    else:
        st.warning("🟠 API online, but no model is loaded yet")
    st.markdown(f"[API documentation (Swagger)]({API_PUBLIC_URL}/docs)")
    if MLFLOW_UI_URL:
        st.markdown(f"[Experiments and models (MLflow)]({MLFLOW_UI_URL})")
    st.divider()
    st.subheader("About the model")
    st.caption(
        "A 1D convolutional neural network trained on 400 rides recorded in Gothenburg. "
        "It reads the ride in 25-second windows and averages its opinion over the whole ride. "
        "With very few dangerous rides to learn from, it is better than chance but not perfect."
    )

# ---------------------------------------------------------------- header
st.title("🚲 Ride safety checker")
st.markdown(
    "A phone mounted on a bike or scooter records **how it shakes and turns, 10 times per second**. "
    "From that recording alone, the model recognises **how the person was riding**."
)
cols = st.columns(len(CLASSES))
for col, (name, (icon, text, _)) in zip(cols, CLASSES.items(), strict=True):
    col.markdown(
        f"**{icon} {name}**  \n<span style='font-size:.85rem;color:#52514e'>{text}</span>", unsafe_allow_html=True
    )

st.divider()

# ---------------------------------------------------------------- step 1: choose a ride
step(1, "Choose a ride")
rides = demo_rides()
mode = st.radio(
    "Ride source",
    ["Example rides", "Upload my own CSV"] if rides else ["Upload my own CSV"],
    horizontal=True,
    label_visibility="collapsed",
)

name, content, true_label = None, None, None
if mode == "Example rides":
    st.caption("Real rides kept aside during training: the model has never seen them.")
    labels = {
        f: (f"{CLASSES.get(m.get('label'), ('🚲',))[0]} {m['label']} ride" if m.get("label") else f)
        for f, m in rides.items()
    }
    chosen = st.radio("Example", list(rides), format_func=labels.get, horizontal=True, label_visibility="collapsed")
    name, content, true_label = chosen, (EXAMPLES_DIR / chosen).read_bytes(), rides[chosen].get("label")
else:
    uploaded = st.file_uploader(
        "CSV with one row per 100 ms and the columns ax, ay, az (acceleration) and rx, ry, rz (rotation)", type="csv"
    )
    if uploaded:
        name, content = uploaded.name, uploaded.getvalue()

if content is None:
    st.stop()

# A new ride clears the previous result.
if st.session_state.get("ride") != name:
    st.session_state.ride, st.session_state.result, st.session_state.error = name, None, None

# ---------------------------------------------------------------- step 2: what the phone recorded
step(2, "What the phone recorded")
try:
    ride = pd.read_csv(io.BytesIO(content))
    ind = ride_indicators(ride)
    turning, shaking = float(ind["turning"].mean()), float(ind["shaking"].mean())
    m1, m2, m3 = st.columns(3)
    m1.metric("Duration", f"{len(ride) / (SAMPLING_HZ * 60):.1f} min", help=f"{len(ride):,} measurements")
    m2.metric(
        "Average turning",
        f"{turning:.0f} °/s",
        f"{turning / TYPICAL_SAFE['turning'] - 1:+.0%} vs typical safe ride",
        delta_color="inverse",
        help="How fast the phone rotates: sharp turns and zig-zags push it up.",
    )
    m3.metric(
        "Average shaking",
        f"{shaking:.2f} m/s²",
        help="How much the phone vibrates and jolts: bumps, braking, accelerations.",
    )
    c1, c2 = st.columns(2)
    c1.altair_chart(
        line(
            ind,
            "turning",
            "Turning: how fast the phone rotates",
            "°/s",
            TYPICAL_SAFE["turning"],
        ),
        width="stretch",
    )
    c2.altair_chart(
        line(ind, "shaking", "Shaking: jolts and vibrations", "m/s²", None),
        width="stretch",
    )
    with st.expander("Raw sensor data (6 channels)"):
        st.line_chart(ride[[c for c in ("ax", "ay", "az", "rx", "ry", "rz") if c in ride]], height=220)
except (ValueError, KeyError) as exc:  # unreadable CSV or missing columns: the API gives the precise error
    st.info(f"No preview for this file ({exc}). You can still send it to the model.")

# ---------------------------------------------------------------- step 3: ask the model
step(3, "Ask the model")
if st.button("🔍  Analyse this ride", type="primary", width="stretch"):
    with st.spinner("The model is reading the ride..."):
        st.session_state.result, st.session_state.error = predict(name, content)

if st.session_state.get("error"):
    st.error(st.session_state.error)

result = st.session_state.get("result")
if result:
    label = result["label"]
    icon, text, kind = CLASSES.get(label, ("🚲", "", "info"))
    left, right = st.columns([1, 1.25], gap="large")
    with left, st.container(border=True):
        st.caption("THE MODEL SAYS")
        st.markdown(
            f"<div style='font-size:2.3rem;font-weight:700;line-height:1.2'>{icon} {label}</div>"
            f"<div style='font-size:1.05rem;color:#52514e;margin:.3rem 0 1rem'>{text}</div>",
            unsafe_allow_html=True,
        )
        st.progress(float(result["confidence"]), text=f"Confidence: {result['confidence']:.0%}")
        if true_label:
            if true_label == label:
                st.success(f"✅ Correct: this ride really was **{true_label}**.")
            else:
                st.warning(f"❌ Wrong: this ride really was **{true_label}**.")
    with right:
        st.caption("SCORE FOR EACH TYPE OF RIDE (the highest wins)")
        st.altair_chart(probability_chart(result["probabilities"], label), width="stretch")
    st.caption(
        f"The model read {result['n_windows']} windows of 25.6 s · "
        f"model version {result.get('model_version') or 'local'} · "
        "every prediction is counted in the monitoring metrics."
    )
    with st.expander("For developers: raw API response"):
        st.json(result)
