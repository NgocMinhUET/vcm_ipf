"""Visualization for multi-method comparison results.

Generates paper-ready figures:
    1. Temporal QP stability comparison (all methods on one plot)
    2. QP distribution comparison (box plots)
    3. Radar chart of method metrics
    4. Per-frame bar chart comparison
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from phase1.utils.log import get_logger

logger = get_logger("analysis.comparison_viz")


def plot_temporal_comparison(
    run_dir: Path,
    output_path: Path,
) -> None:
    """Plot temporal QP mean for all methods on the same axes.

    This is the KEY figure for proving temporal stability (KPI-2).
    """
    import csv

    run_dir = Path(run_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), dpi=150, sharex=True)

    colors = {
        "M0": "#888888", "M1": "#e74c3c", "M4": "#2ecc71",
        "M5": "#3498db", "M6": "#9b59b6", "M7": "#f39c12", "M8": "#1abc9c",
    }
    labels = {
        "M0": "M0: Uniform QP",
        "M1": "M1: Binary ROI",
        "M4": "M4: IPF (Ours)",
        "M5": "M5: Gaussian",
        "M6": "M6: Exponential",
        "M7": "M7: Distance Transform",
        "M8": "M8: Blurred ROI",
    }

    for mid_dir in sorted(run_dir.iterdir()):
        if not mid_dir.is_dir():
            continue
        mid = mid_dir.name
        csv_path = mid_dir / "frame_stats.csv"
        if not csv_path.exists():
            continue

        frames, means, stds = [], [], []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                frames.append(int(row["frame_idx"]))
                means.append(float(row["qp_mean"]))
                stds.append(float(row["qp_std"]))

        if not frames:
            continue

        color = colors.get(mid, "#333333")
        label = labels.get(mid, mid)
        lw = 2.5 if mid == "M4" else 1.5
        ls = "-" if mid == "M4" else "--"

        axes[0].plot(frames, means, color=color, linewidth=lw, linestyle=ls, label=label)
        axes[1].plot(frames, stds, color=color, linewidth=lw, linestyle=ls, label=label)

    axes[0].set_ylabel("Mean QP per Frame", fontsize=12)
    axes[0].set_title("Temporal QP Stability Comparison", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=9, loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("QP Std per Frame", fontsize=12)
    axes[1].set_xlabel("Frame Index", fontsize=12)
    axes[1].set_title("QP Spatial Variance per Frame", fontsize=14)
    axes[1].legend(fontsize=9, loc="best")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(output_path), bbox_inches="tight")
    plt.close(fig)
    logger.info("Temporal comparison plot saved: %s", output_path.name)


def plot_metric_bars(
    run_dir: Path,
    output_path: Path,
) -> None:
    """Bar chart comparing key metrics across methods."""
    run_dir = Path(run_dir)
    summary_path = run_dir / "comparison_summary.json"
    if not summary_path.exists():
        return

    data = json.loads(summary_path.read_text())
    methods_data = data.get("methods", {})

    if not methods_data:
        return

    mids = sorted(methods_data.keys())
    labels_map = {
        "M0": "Uniform", "M1": "Binary\nROI", "M4": "IPF\n(Ours)",
        "M5": "Gaussian", "M6": "Exp\nDecay", "M7": "Dist\nTransform", "M8": "Blurred\nROI",
    }
    colors_map = {
        "M0": "#888888", "M1": "#e74c3c", "M4": "#2ecc71",
        "M5": "#3498db", "M6": "#9b59b6", "M7": "#f39c12", "M8": "#1abc9c",
    }

    labels = [labels_map.get(m, m) for m in mids]
    colors = [colors_map.get(m, "#333") for m in mids]

    # Extract metrics
    jitter = [methods_data[m]["temporal_jitter_index"] for m in mids]
    ctu_std = [methods_data[m]["per_ctu_temporal_std_mean"] for m in mids]
    smoothness = [methods_data[m].get("spatial_smoothness_mean", 0) for m in mids]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)

    axes[0].bar(labels, jitter, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_title("Temporal Jitter Index\n(lower = better)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Mean |ΔQP|/frame/CTU")

    axes[1].bar(labels, ctu_std, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].set_title("Per-CTU Temporal σ\n(lower = more stable)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Mean σ_t per CTU")

    axes[2].bar(labels, smoothness, color=colors, edgecolor="black", linewidth=0.5)
    axes[2].set_title("Spatial Smoothness\n(lower = smoother)", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Mean |∇QP|")

    for ax in axes:
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), bbox_inches="tight")
    plt.close(fig)
    logger.info("Metric bars saved: %s", output_path.name)
