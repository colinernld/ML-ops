"""Exploratory figures + stats for docs/data.md.

Usage:
    python scripts/eda.py --raw-dir data/raw --out docs/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trajclf.config import CLASSES, SIGNAL_COLS
from trajclf.data import LABELS_FILE, RIDES_FILE, load_labels

BLUE, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.raw_dir / RIDES_FILE)
    labels = load_labels(args.raw_dir / LABELS_FILE)
    lengths = df.groupby("ride_id").size()
    df["label"] = df.ride_id.map(labels)

    # 1. class balance
    counts = labels.value_counts().reindex(CLASSES)
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    ax.barh(counts.index[::-1], counts.values[::-1], color=BLUE, height=0.6)
    for y, v in enumerate(counts.values[::-1]):
        ax.text(v + 4, y, f"{v}  ({v / counts.sum():.0%})", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, counts.max() * 1.25)
    style(ax, "Rides per class (GBG500, n = 500)")
    fig.tight_layout()
    fig.savefig(args.out / "class_balance.png", dpi=150)

    # 2. ride duration
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    ax.hist(lengths / 600, bins=30, color=BLUE, edgecolor="white", linewidth=1)
    ax.set_xlabel("ride duration (minutes)", color=MUTED, fontsize=9)
    style(ax, "Ride duration (10 Hz sampling)")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(args.out / "ride_duration.png", dpi=150)

    # 3. one example ride per class: acceleration and rotation magnitudes (first 2 minutes)
    fig, axes = plt.subplots(len(CLASSES), 2, figsize=(9, 8), sharex=True)
    rng = np.random.default_rng(0)
    for i, cls in enumerate(CLASSES):
        rid = rng.choice(labels[labels == cls].index)
        ride = df[df.ride_id == rid].sort_values("time_index").head(1200)
        t = (ride.time_index - ride.time_index.iloc[0]) / 1000
        acc = np.linalg.norm(ride[["ax", "ay", "az"]].to_numpy(), axis=1)
        rot = np.linalg.norm(ride[["rx", "ry", "rz"]].to_numpy(), axis=1)
        for j, (sig, unit) in enumerate([(acc, "|a| m/s²"), (rot, "|r| °/s")]):
            ax = axes[i, j]
            ax.plot(t, sig, color=BLUE, linewidth=0.8)
            style(ax, f"{cls} - {unit}")
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.grid(axis="x", visible=False)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)", color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out / "example_rides.png", dpi=150)

    # stats per class (used in docs/data.md)
    df["acc_norm"] = np.linalg.norm(df[["ax", "ay", "az"]].to_numpy(), axis=1)
    df["rot_norm"] = np.linalg.norm(df[["rx", "ry", "rz"]].to_numpy(), axis=1)
    per_ride = df.groupby("ride_id").agg(
        label=("label", "first"),
        minutes=("time_index", lambda s: (s.max() - s.min()) / 60000),
        acc_std=("acc_norm", "std"),
        rot_mean=("rot_norm", "mean"),
        rot_p99=("rot_norm", lambda s: s.quantile(0.99)),
    )
    stats = per_ride.groupby("label")[["minutes", "acc_std", "rot_mean", "rot_p99"]].median().reindex(CLASSES)
    stats.round(2).to_csv(args.out / "per_class_stats.csv")
    summary = {
        "n_rows": len(df),
        "n_rides": int(lengths.size),
        "missing_rows": int(df[SIGNAL_COLS].isna().any(axis=1).sum()),
        "rides_with_missing": int(df[df[SIGNAL_COLS].isna().any(axis=1)].ride_id.nunique()),
        "length_min": int(lengths.min()),
        "length_median": int(lengths.median()),
        "length_max": int(lengths.max()),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(stats.round(2).to_string())


if __name__ == "__main__":
    main()
