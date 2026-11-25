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

# Plain-language meaning of each class, with an accent colour for the verdict card.
CLASSES = {
    "Safe": ("🟢", "Calm, normal riding.", "#1f9d55"),
    "Safe overall": ("🟢", "Mostly calm, with a few unusual moments.", "#1f9d55"),
    "Bad weather": ("🌧️", "Rain or wind: more vibrations, careful riding.", "#2a78d6"),
    "Tandem": ("👥", "Two people on board: heavier, steadier ride.", "#6250d6"),
    "Reckless": ("🔴", "Dangerous riding: sharp turns, sudden accelerations.", "#d03b3b"),
}
# Median over the Safe rides of the training data (docs/data.md).
TYPICAL_SAFE_TURNING = 15.5

INK, MUTED, LINE, ACCENT = "#0f172a", "#64748b", "#e2e8f0", "#2a78d6"

st.set_page_config(page_title="Ride safety checker", page_icon="🚲", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    f"""
    <style>
      /* hide Streamlit's own chrome */
      #MainMenu, footer, header, [data-testid="stToolbar"], [data-testid="stDecoration"],
      [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {{display: none !important;}}
      .stApp {{background: #f6f8fb;}}
      .block-container {{padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1180px;}}
      html, body, [class*="css"], .stMarkdown, p, div {{font-family: "Inter", system-ui, -apple-system, "Segoe UI", sans-serif;}}

      .hero {{background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #2a78d6 100%); color: white;
              border-radius: 22px; padding: 2.1rem 2.4rem; margin-bottom: 1.6rem;
              box-shadow: 0 18px 40px -22px rgba(15,23,42,.6);}}
      .hero h1 {{color: white; font-size: 2.4rem; font-weight: 800; margin: 0 0 .4rem; letter-spacing: -.02em;}}
      .hero p {{color: #dbe7ff; font-size: 1.05rem; margin: 0; max-width: 760px; line-height: 1.55;}}
      .pill {{display: inline-flex; align-items: center; gap: .45rem; border-radius: 999px; padding: .32rem .8rem;
              font-size: .8rem; font-weight: 600; margin-bottom: 1rem;}}
      .pill.ok {{background: rgba(31,157,85,.18); color: #b8f5cf;}}
      .pill.ko {{background: rgba(208,59,59,.22); color: #ffd1d1;}}
      .pill.warn {{background: rgba(250,178,25,.22); color: #ffe7b0;}}
      .hero a {{color: #dbe7ff; font-size: .85rem; margin-right: 1.1rem;}}

      .classes {{display: grid; grid-template-columns: repeat(5, 1fr); gap: .7rem; margin-top: 1.5rem;}}
      .cls {{background: rgba(255,255,255,.09); border: 1px solid rgba(255,255,255,.14); border-radius: 14px;
             padding: .75rem .85rem;}}
      .cls b {{display: block; color: white; font-size: .95rem; margin-bottom: .2rem;}}
      .cls span {{color: #c7d6f5; font-size: .8rem; line-height: 1.35;}}

      .card {{background: white; border: 1px solid {LINE}; border-radius: 18px; padding: 1.4rem 1.6rem;
              box-shadow: 0 8px 24px -18px rgba(15,23,42,.35); margin-bottom: 1.2rem;}}
      .step {{color: {ACCENT}; font-weight: 700; font-size: .75rem; letter-spacing: .08em; text-transform: uppercase;}}
      .title {{color: {INK}; font-size: 1.45rem; font-weight: 700; margin: .1rem 0 .3rem;}}
      .sub {{color: {MUTED}; font-size: .92rem; margin-bottom: .2rem;}}

      .stats {{display: grid; grid-template-columns: repeat(3, 1fr); gap: .9rem; margin: .6rem 0 .4rem;}}
      .stat {{background: #f8fafc; border: 1px solid {LINE}; border-radius: 14px; padding: .9rem 1.1rem;}}
      .stat .k {{color: {MUTED}; font-size: .8rem; font-weight: 600; text-transform: uppercase; letter-spacing: .04em;}}
      .stat .v {{color: {INK}; font-size: 1.7rem; font-weight: 750; margin-top: .15rem;}}
      .stat .d {{font-size: .82rem; font-weight: 600; margin-top: .1rem;}}

      .verdict {{border-radius: 18px; padding: 1.5rem 1.6rem; background: white; border: 1px solid {LINE};
                 border-left-width: 8px; height: 100%;}}
      .verdict .lbl {{color: {MUTED}; font-size: .78rem; font-weight: 700; letter-spacing: .08em;}}
      .verdict .big {{font-size: 2.5rem; font-weight: 800; line-height: 1.15; margin: .35rem 0 .3rem;}}
      .verdict .txt {{color: #334155; font-size: 1.02rem; margin-bottom: 1.1rem;}}
      .bar {{background: #eef2f7; border-radius: 999px; height: 10px; overflow: hidden;}}
      .bar > div {{height: 100%; border-radius: 999px;}}
      .conf {{display: flex; justify-content: space-between; color: {MUTED}; font-size: .85rem; margin: .35rem 0 1rem;}}
      .truth {{border-radius: 12px; padding: .7rem .9rem; font-size: .92rem; font-weight: 600;}}
      .truth.ok {{background: #e9f8ef; color: #136c3a;}}
      .truth.ko {{background: #fff4e5; color: #8a4b00;}}

      /* Streamlit widgets */
      div.stButton > button {{background: {ACCENT}; color: white; border: 0; border-radius: 14px; height: 3.2rem;
                              font-size: 1.05rem; font-weight: 700; box-shadow: 0 10px 22px -12px rgba(42,120,214,.9);}}
      div.stButton > button:hover {{background: #1c62b8; color: white;}}
      [data-testid="stSegmentedControl"] button, div[data-baseweb="button-group"] button {{border-radius: 12px !important;}}
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
    order = list(CLASSES)
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
    out[["shaking", "turning"]] = out[["shaking", "turning"]].rolling(SAMPLING_HZ, min_periods=1, center=True).mean()
    return out


def styled(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure_view(strokeWidth=0)
        .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor="#eef2f7", domain=False, tickColor=LINE)
        .configure_title(color=INK, fontSize=14, anchor="start", subtitleColor=MUTED)
    )


def area(data: pd.DataFrame, field: str, title: str, subtitle: str, unit: str, typical: float | None) -> alt.Chart:
    base = alt.Chart(data).encode(x=alt.X("minute:Q", title="minutes", axis=alt.Axis(grid=False, tickCount=6)))
    chart = base.mark_area(
        line={"color": ACCENT, "strokeWidth": 1.4},
        color=alt.Gradient(
            gradient="linear",
            stops=[
                alt.GradientStop(color="rgba(42,120,214,0.30)", offset=0),
                alt.GradientStop(color="white", offset=1),
            ],
            x1=1,
            x2=1,
            y1=0,
            y2=1,
        ),
    ).encode(
        y=alt.Y(f"{field}:Q", title=unit),
        tooltip=[alt.Tooltip("minute:Q", format=".1f", title="minute"), alt.Tooltip(f"{field}:Q", format=".1f")],
    )
    if typical is not None:
        rule = alt.Chart(pd.DataFrame({"y": [typical]})).mark_rule(color=INK, strokeDash=[4, 4], opacity=0.6)
        chart = chart + rule.encode(y="y:Q")
    return styled(chart.properties(title=alt.TitleParams(title, subtitle=subtitle), height=210))


def probability_chart(probs: dict[str, float], predicted: str) -> alt.Chart:
    data = pd.DataFrame({"class": list(CLASSES), "probability": [probs.get(c, 0.0) for c in CLASSES]})
    data["predicted"] = data["class"] == predicted
    data["text"] = data["probability"].map(lambda p: f"{p:.0%}")
    bars = (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=6, height=24)
        .encode(
            y=alt.Y("class:N", sort=list(CLASSES), title=None, axis=alt.Axis(labelFontSize=13, labelColor=INK)),
            x=alt.X("probability:Q", title=None, scale=alt.Scale(domain=[0, 1]), axis=None),
            color=alt.condition("datum.predicted", alt.value(ACCENT), alt.value("#d9e2ec")),
            tooltip=[alt.Tooltip("class:N"), alt.Tooltip("probability:Q", format=".1%")],
        )
    )
    text = bars.mark_text(align="left", dx=8, fontSize=13, fontWeight=600).encode(text="text:N", color=alt.value(INK))
    return styled((bars + text).properties(height=230, title=alt.TitleParams("Score for each type of ride")))


def card_header(step: int, title: str, sub: str = "") -> None:
    st.markdown(
        f"<div class='step'>Step {step}</div><div class='title'>{title}</div>"
        + (f"<div class='sub'>{sub}</div>" if sub else ""),
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- hero
health, model, error = api_status()
if error:
    pill = "<span class='pill ko'>● API offline</span>"
elif model:
    version = f"model v{model['version']}" if model.get("version") else "local model"
    alias = f" · @{model['alias']}" if model.get("alias") else ""
    pill = f"<span class='pill ok'>● Live · {version}{alias}</span>"
else:
    pill = "<span class='pill warn'>● API online · no model loaded yet</span>"
links = f"<a href='{API_PUBLIC_URL}/docs' target='_blank'>API docs ↗</a>"
if MLFLOW_UI_URL:
    links += f"<a href='{MLFLOW_UI_URL}' target='_blank'>MLflow ↗</a>"
classes_html = "".join(f"<div class='cls'><b>{i} {n}</b><span>{t}</span></div>" for n, (i, t, _) in CLASSES.items())
st.markdown(
    f"""
    <div class="hero">
      {pill}
      <h1>🚲 Ride safety checker</h1>
      <p>A phone on a bike or scooter records how it <b>shakes</b> and <b>turns</b>, ten times per second.
         From that alone, our deep learning model tells <b>how the person was riding</b>.</p>
      <div style="margin-top:.9rem">{links}</div>
      <div class="classes">{classes_html}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- step 1: choose a ride
rides = demo_rides()
with st.container(border=False):
    card_header(1, "Choose a ride", "Real rides kept aside during training: the model has never seen them.")
    name, content, true_label = None, None, None
    options = list(rides) + ["upload"]
    labels = {f: f"{CLASSES.get(m.get('label'), ('🚲',))[0]} {m.get('label', f)}" for f, m in rides.items()}
    labels["upload"] = "📂 My own CSV"
    choice = st.segmented_control("Ride", options, format_func=labels.get, default=None, label_visibility="collapsed")
    if choice and choice != "upload":
        name, content, true_label = choice, (EXAMPLES_DIR / choice).read_bytes(), rides[choice].get("label")
    elif choice == "upload":
        uploaded = st.file_uploader(
            "CSV with one row per 100 ms and the columns ax, ay, az (acceleration) and rx, ry, rz (rotation)",
            type="csv",
        )
        if uploaded:
            name, content = uploaded.name, uploaded.getvalue()

if content is None:
    st.stop()

if st.session_state.get("ride") != name:  # a new ride clears the previous result
    st.session_state.ride, st.session_state.result, st.session_state.error = name, None, None

st.write("")

# ---------------------------------------------------------------- step 2: what the phone recorded
card_header(2, "What the phone recorded")
try:
    ride = pd.read_csv(io.BytesIO(content))
    ind = ride_indicators(ride)
    turning, shaking = float(ind["turning"].mean()), float(ind["shaking"].mean())
    delta = turning / TYPICAL_SAFE_TURNING - 1
    delta_color = "#d03b3b" if delta > 0.5 else ("#b7791f" if delta > 0.15 else "#1f9d55")
    st.markdown(
        f"""
        <div class="stats">
          <div class="stat"><div class="k">Duration</div><div class="v">{len(ride) / (SAMPLING_HZ * 60):.1f} min</div>
            <div class="d" style="color:{MUTED}">{len(ride):,} measurements</div></div>
          <div class="stat"><div class="k">Turning</div><div class="v">{turning:.0f} °/s</div>
            <div class="d" style="color:{delta_color}">{delta:+.0%} vs a typical safe ride</div></div>
          <div class="stat"><div class="k">Shaking</div><div class="v">{shaking:.2f} m/s²</div>
            <div class="d" style="color:{MUTED}">jolts, bumps and braking</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2, gap="medium")
    c1.altair_chart(
        area(
            ind,
            "turning",
            "Turning",
            "how fast the phone rotates · dashed: typical safe ride",
            "°/s",
            TYPICAL_SAFE_TURNING,
        ),
        width="stretch",
    )
    c2.altair_chart(area(ind, "shaking", "Shaking", "jolts and vibrations", "m/s²", None), width="stretch")
    with st.expander("Raw sensor data (6 channels)"):
        st.line_chart(ride[[c for c in ("ax", "ay", "az", "rx", "ry", "rz") if c in ride]], height=220)
except (ValueError, KeyError) as exc:  # unreadable CSV or missing columns: the API gives the precise error
    st.info(f"No preview for this file ({exc}). You can still send it to the model.")

st.write("")

# ---------------------------------------------------------------- step 3: ask the model
card_header(3, "Ask the model")
if st.button("🔍  Analyse this ride", type="primary", width="stretch"):
    with st.spinner("The model is reading the ride..."):
        st.session_state.result, st.session_state.error = predict(name, content)

if st.session_state.get("error"):
    st.error(st.session_state.error)

result = st.session_state.get("result")
if result:
    label = result["label"]
    icon, text, color = CLASSES.get(label, ("🚲", "", ACCENT))
    conf = float(result["confidence"])
    truth = ""
    if true_label:
        if true_label == label:
            truth = f"<div class='truth ok'>✅ Correct: this ride really was <b>{true_label}</b>.</div>"
        else:
            truth = f"<div class='truth ko'>❌ Not this time: this ride really was <b>{true_label}</b>.</div>"
    st.write("")
    left, right = st.columns([1, 1.2], gap="large")
    left.markdown(
        f"""
        <div class="verdict" style="border-left-color:{color}">
          <div class="lbl">THE MODEL SAYS</div>
          <div class="big" style="color:{color}">{icon} {label}</div>
          <div class="txt">{text}</div>
          <div class="bar"><div style="width:{conf:.0%};background:{color}"></div></div>
          <div class="conf"><span>Confidence</span><b style="color:{INK}">{conf:.0%}</b></div>
          {truth}
        </div>
        """,
        unsafe_allow_html=True,
    )
    with right:
        st.altair_chart(probability_chart(result["probabilities"], label), width="stretch")
    st.caption(
        f"Read as {result['n_windows']} windows of 25.6 s · model version {result.get('model_version') or 'local'} · "
        "every prediction is counted in the monitoring (Grafana)."
    )
    with st.expander("For developers: raw API response"):
        st.json(result)
