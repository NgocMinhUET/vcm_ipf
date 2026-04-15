"""Paper-ready figure generation for Phase 2 results.

Produces publication-quality figures for the SCIE manuscript:
    1. RD curves (Rate vs PSNR) per sequence
    2. RD curves (Rate vs mAP) per sequence
    3. BD-Rate bar chart across methods
    4. BD-Rate per-sequence heatmap
    5. Visual quality comparison at matched bitrate
    6. Temporal QP stability comparison (integrated with Phase 1)

All figures follow standard academic formatting:
    - Matplotlib with serif fonts
    - Consistent color scheme and markers
    - LaTeX-compatible labels
    - 300 DPI for print quality

Usage:
    python -m phase2.analysis.paper_figures \\
        --experiment-dir ~/Minh/ipf/phase2_outputs/pilot_v1 \\
        --output-dir ~/Minh/ipf/paper_figures
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("phase2.analysis.paper_figures")

METHOD_COLORS = {
    "M0": "#888888",
    "M1": "#e74c3c",
    "M4": "#2ecc71",
    "M5": "#3498db",
    "M6": "#9b59b6",
    "M7": "#f39c12",
    "M8": "#1abc9c",
}

METHOD_MARKERS = {
    "M0": "s",
    "M1": "D",
    "M4": "o",
    "M5": "^",
    "M6": "v",
    "M7": "<",
    "M8": ">",
}

METHOD_LABELS = {
    "M0": "M0 (Uniform QP)",
    "M1": "M1 (Binary ROI)",
    "M4": "M4 (IPF v2 - Ours)",
    "M5": "M5 (Gaussian Soft-map)",
    "M6": "M6 (Exponential Soft-map)",
    "M7": "M7 (Distance Transform)",
    "M8": "M8 (Blurred ROI)",
}


def _setup_matplotlib():
    """Configure matplotlib for publication-quality figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    return plt


class PaperFigureGenerator:
    """Generates paper-ready figures from Phase 2 experiment results."""

    def __init__(self, experiment_dir: str, output_dir: Optional[str] = None):
        self.experiment_dir = Path(experiment_dir).expanduser()
        self.output_dir = Path(output_dir).expanduser() if output_dir else (
            self.experiment_dir / "figures"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self) -> List[str]:
        """Generate all paper figures. Returns list of saved file paths."""
        summary = self._load_summary()
        analysis = self._load_analysis()

        saved = []
        saved.extend(self._plot_rd_curves_psnr(summary))
        saved.extend(self._plot_rd_curves_task(summary))

        if analysis:
            saved.extend(self._plot_bd_rate_bars(analysis))
            saved.extend(self._plot_bd_rate_heatmap(analysis))

        print(f"Generated {len(saved)} figures in {self.output_dir}")
        return saved

    def _load_summary(self) -> dict:
        """Load experiment summary."""
        path = self.experiment_dir / "experiment_summary.json"
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_analysis(self) -> dict:
        """Load analysis report."""
        path = self.experiment_dir / "analysis" / "analysis.json"
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _plot_rd_curves_psnr(self, summary: dict) -> List[str]:
        """Plot Rate-Distortion curves (bitrate vs PSNR) per sequence."""
        plt = _setup_matplotlib()
        saved = []

        rd_data = self._extract_rd_data(summary, quality_key="psnr_y_enc")
        sequences = sorted(set(s for s, _, _ in rd_data))

        for seq in sequences:
            fig, ax = plt.subplots(figsize=(8, 5.5))

            for method in sorted(set(m for _, m, _ in rd_data)):
                points = [(rate, q) for (s, m, _), (rate, q) in
                          zip(rd_data, [(p[2], p[3]) for p in rd_data])
                          if False]

            method_points: Dict[str, List[Tuple[float, float]]] = {}
            for s, m, qp, rate, quality in rd_data:
                if s != seq:
                    continue
                if m not in method_points:
                    method_points[m] = []
                method_points[m].append((rate, quality))

            for method in sorted(method_points.keys()):
                pts = sorted(method_points[method], key=lambda x: x[0])
                rates = [p[0] for p in pts]
                qualities = [p[1] for p in pts]

                ax.plot(
                    rates, qualities,
                    color=METHOD_COLORS.get(method, "#333"),
                    marker=METHOD_MARKERS.get(method, "o"),
                    markersize=7,
                    linewidth=1.8,
                    label=METHOD_LABELS.get(method, method),
                )

            ax.set_xlabel("Bitrate (kbps)")
            ax.set_ylabel("PSNR-Y (dB)")
            ax.set_title(f"Rate-Distortion: {seq}")
            ax.legend(loc="lower right", framealpha=0.9)
            ax.set_xscale("log")

            fpath = self.output_dir / f"rd_psnr_{seq}.png"
            fig.savefig(str(fpath))
            plt.close(fig)
            saved.append(str(fpath))

        if len(sequences) > 1:
            fig, axes = plt.subplots(
                1, len(sequences),
                figsize=(6 * len(sequences), 5),
                sharey=True,
            )
            if len(sequences) == 1:
                axes = [axes]

            for ax, seq in zip(axes, sequences):
                method_points = {}
                for s, m, qp, rate, quality in rd_data:
                    if s != seq:
                        continue
                    if m not in method_points:
                        method_points[m] = []
                    method_points[m].append((rate, quality))

                for method in sorted(method_points.keys()):
                    pts = sorted(method_points[method], key=lambda x: x[0])
                    rates = [p[0] for p in pts]
                    qualities = [p[1] for p in pts]
                    ax.plot(
                        rates, qualities,
                        color=METHOD_COLORS.get(method, "#333"),
                        marker=METHOD_MARKERS.get(method, "o"),
                        markersize=6,
                        linewidth=1.5,
                        label=METHOD_LABELS.get(method, method),
                    )
                ax.set_xlabel("Bitrate (kbps)")
                ax.set_title(seq)
                ax.set_xscale("log")

            axes[0].set_ylabel("PSNR-Y (dB)")
            axes[-1].legend(loc="lower right", fontsize=8, framealpha=0.9)

            fpath = self.output_dir / "rd_psnr_all_sequences.png"
            fig.savefig(str(fpath))
            plt.close(fig)
            saved.append(str(fpath))

        return saved

    def _plot_rd_curves_task(self, summary: dict) -> List[str]:
        """Plot Rate-Task curves (bitrate vs mAP) per sequence."""
        plt = _setup_matplotlib()
        saved = []

        rd_data = self._extract_rd_data(summary, quality_key="mAP50")
        if not any(q > 0 for _, _, _, _, q in rd_data):
            return saved

        sequences = sorted(set(s for s, _, _, _, _ in rd_data))

        for seq in sequences:
            fig, ax = plt.subplots(figsize=(8, 5.5))

            method_points: Dict[str, List[Tuple[float, float]]] = {}
            for s, m, qp, rate, quality in rd_data:
                if s != seq or quality <= 0:
                    continue
                if m not in method_points:
                    method_points[m] = []
                method_points[m].append((rate, quality))

            for method in sorted(method_points.keys()):
                pts = sorted(method_points[method], key=lambda x: x[0])
                rates = [p[0] for p in pts]
                qualities = [p[1] for p in pts]
                ax.plot(
                    rates, qualities,
                    color=METHOD_COLORS.get(method, "#333"),
                    marker=METHOD_MARKERS.get(method, "o"),
                    markersize=7,
                    linewidth=1.8,
                    label=METHOD_LABELS.get(method, method),
                )

            ax.set_xlabel("Bitrate (kbps)")
            ax.set_ylabel("mAP@50")
            ax.set_title(f"Rate-Task (Detection): {seq}")
            ax.legend(loc="lower right", framealpha=0.9)
            ax.set_xscale("log")

            fpath = self.output_dir / f"rd_task_{seq}.png"
            fig.savefig(str(fpath))
            plt.close(fig)
            saved.append(str(fpath))

        return saved

    def _plot_bd_rate_bars(self, analysis: dict) -> List[str]:
        """Plot BD-Rate comparison bar chart."""
        plt = _setup_matplotlib()
        saved = []

        agg = analysis.get("aggregated", {})
        if not agg:
            return saved

        methods = sorted(agg.keys())
        bd_psnr = [agg[m]["bd_rate_psnr_mean"] for m in methods]
        bd_roi = [agg[m]["bd_rate_roi_mean"] for m in methods]
        bd_task = [agg[m]["bd_rate_task_mean"] for m in methods]
        bd_psnr_std = [agg[m]["bd_rate_psnr_std"] for m in methods]

        x = np.arange(len(methods))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 6))

        bars1 = ax.bar(x - width, bd_psnr, width,
                       yerr=bd_psnr_std, capsize=4,
                       label="BD-Rate (PSNR)", color="#3498db", alpha=0.85)
        bars2 = ax.bar(x, bd_roi, width,
                       label="BD-Rate (ROI)", color="#2ecc71", alpha=0.85)
        bars3 = ax.bar(x + width, bd_task, width,
                       label="BD-Rate (Task)", color="#e74c3c", alpha=0.85)

        ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
        ax.set_xlabel("Method")
        ax.set_ylabel("BD-Rate (%)")
        ax.set_title("BD-Rate Comparison (vs M0 anchor)")
        ax.set_xticks(x)

        labels = [METHOD_LABELS.get(m, m) for m in methods]
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.legend()

        for i, m in enumerate(methods):
            if m == "M4":
                ax.get_xticklabels()[i].set_fontweight("bold")

        fpath = self.output_dir / "bd_rate_comparison.png"
        fig.savefig(str(fpath))
        plt.close(fig)
        saved.append(str(fpath))

        return saved

    def _plot_bd_rate_heatmap(self, analysis: dict) -> List[str]:
        """Plot BD-Rate per-sequence heatmap."""
        plt = _setup_matplotlib()
        saved = []

        agg = analysis.get("aggregated", {})
        if not agg:
            return saved

        methods = sorted(agg.keys())
        all_sequences = set()
        for m in methods:
            for bd in agg[m].get("per_sequence", []):
                all_sequences.add(bd["sequence"])
        sequences = sorted(all_sequences)

        if not sequences or len(sequences) < 2:
            return saved

        matrix = np.zeros((len(methods), len(sequences)))
        for i, m in enumerate(methods):
            for bd in agg[m].get("per_sequence", []):
                if bd["sequence"] in sequences:
                    j = sequences.index(bd["sequence"])
                    matrix[i, j] = bd["bd_rate_psnr"]

        fig, ax = plt.subplots(figsize=(max(8, len(sequences) * 2), len(methods) * 0.8 + 2))

        im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(len(sequences)))
        ax.set_xticklabels(sequences, rotation=45, ha="right")
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([METHOD_LABELS.get(m, m) for m in methods])

        for i in range(len(methods)):
            for j in range(len(sequences)):
                ax.text(j, i, f"{matrix[i,j]:+.1f}%",
                        ha="center", va="center", fontsize=9,
                        color="white" if abs(matrix[i,j]) > 10 else "black")

        fig.colorbar(im, ax=ax, label="BD-Rate (%)")
        ax.set_title("BD-Rate per Sequence (vs M0)")

        fpath = self.output_dir / "bd_rate_heatmap.png"
        fig.savefig(str(fpath))
        plt.close(fig)
        saved.append(str(fpath))

        return saved

    def _extract_rd_data(
        self,
        summary: dict,
        quality_key: str = "psnr_y_enc",
    ) -> List[Tuple[str, str, int, float, float]]:
        """Extract (sequence, method, qp, bitrate, quality) tuples from summary."""
        data = []
        for entry in summary.get("results", []):
            if not entry.get("success", False):
                continue
            rate = entry.get("bitrate_kbps", 0)
            if rate <= 0:
                continue
            quality = entry.get(quality_key, 0)

            data.append((
                entry["sequence"],
                entry["method"],
                entry["qp_base"],
                rate,
                quality,
            ))
        return data


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate paper-ready figures")
    parser.add_argument("--experiment-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    gen = PaperFigureGenerator(args.experiment_dir, args.output_dir)
    gen.generate_all()


if __name__ == "__main__":
    main()
