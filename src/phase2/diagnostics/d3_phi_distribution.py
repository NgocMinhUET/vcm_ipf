"""Diagnostic D3 — φ_oracle saliency distribution analysis.

Why this script exists
----------------------
PROJECT_AUDIT.md asks: does the IPF concept (CTU-level saliency-driven QP
allocation) even have *headroom* on our data? If φ_oracle is nearly
uniform across the frame, there is nothing to redistribute and the entire
framework is moot regardless of how clever the CNN is.

This script measures, per sequence:

* **Spatial entropy** of the per-frame normalised φ_oracle, in nats.
  A uniform 9×15 grid has H_max = log(135) ≈ 4.91. Lower entropy ⇒ stronger
  spatial concentration ⇒ more headroom for IPF.
* **Top-10 / bottom-10 percentile spread**: ratio of mean φ in the top
  decile vs the bottom decile. Larger spread ⇒ stronger gradient.
* **Top-K mass concentration**: fraction of total Σφ contained in the
  top-K CTUs (K = 10, 20). High concentration ⇒ "ROI exists".
* **Temporal stability**: per-CTU std across frames (do hot regions stay
  hot or drift?).
* **Cross-sequence correlation**: φ on the spatial grid for sequence A vs
  sequence B (Spearman). High correlation ⇒ φ may be more about generic
  scene structure than per-sequence task content.

Usage
-----
::

    PYTHONPATH=src python -m phase2.diagnostics.d3_phi_distribution \\
        --saliency-dir ~/Minh/ipf/phase3_outputs/saliency \\
        --sequences MOT17-02-DPM MOT17-04-DPM MOT17-09-DPM \\
        --output ~/Minh/ipf/phase3_outputs/diagnostics/d3_phi.json

Output ``d3_phi.json`` and a sibling ``d3_phi.md`` are written.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("phase2.diagnostics.d3_phi_distribution")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _list_phi_files(seq_dir: Path) -> List[Path]:
    cand = sorted(seq_dir.glob("phi_oracle_*.npy"))
    if cand:
        return cand
    return sorted(seq_dir.glob("*.npy"))


def _load_per_frame(seq_dir: Path, max_frames: int = 0) -> np.ndarray:
    """Stack all frames into a (T, H, W) array."""
    files = _list_phi_files(seq_dir)
    if max_frames > 0:
        files = files[:max_frames]
    if not files:
        raise FileNotFoundError(f"No .npy files in {seq_dir}")
    grids = [np.load(f).astype(np.float64) for f in files]
    shape = grids[0].shape
    grids = [g for g in grids if g.shape == shape]
    return np.stack(grids, axis=0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _normalise(grid: np.ndarray) -> np.ndarray:
    """Per-frame normalisation to a probability distribution."""
    g = np.clip(grid.astype(np.float64), 0.0, None)
    s = float(g.sum())
    if s <= 0:
        return np.full_like(g, 1.0 / g.size)
    return g / s


def _entropy(p: np.ndarray) -> float:
    """Shannon entropy in nats. ``p`` is a probability distribution."""
    flat = p.ravel()
    flat = flat[flat > 0]
    if flat.size == 0:
        return 0.0
    return float(-(flat * np.log(flat)).sum())


def _percentile_spread(grid: np.ndarray) -> float:
    flat = grid.ravel()
    if flat.size == 0:
        return 0.0
    top = float(np.mean(flat[flat >= np.quantile(flat, 0.90)]))
    bot = float(np.mean(flat[flat <= np.quantile(flat, 0.10)]))
    return top / max(bot, 1e-9)


def _top_k_mass(grid: np.ndarray, k: int) -> float:
    flat = np.sort(grid.ravel())[::-1]
    s = float(flat.sum())
    if s <= 0 or flat.size == 0:
        return 0.0
    return float(flat[:min(k, flat.size)].sum() / s)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size != y.size or x.size < 4:
        return float("nan")
    if float(np.std(x)) == 0 or float(np.std(y)) == 0:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


# ---------------------------------------------------------------------------
# Per-sequence analysis
# ---------------------------------------------------------------------------

@dataclass
class SequencePhiStats:
    sequence: str
    n_frames: int
    grid_shape: List[int]
    H_max_nats: float
    mean_entropy_nats: float
    entropy_ratio: float          # mean_entropy / H_max  (1 = uniform, 0 = delta)
    mean_percentile_spread: float
    mean_top10_mass: float
    mean_top20_mass: float
    temporal_std_mean: float
    headroom_score: float = 0.0   # composite, see compute_headroom()


def _compute_one(seq_name: str, grids: np.ndarray) -> SequencePhiStats:
    T, H, W = grids.shape
    H_max = float(np.log(H * W))
    entropies: List[float] = []
    spreads: List[float] = []
    t10: List[float] = []
    t20: List[float] = []
    for t in range(T):
        p = _normalise(grids[t])
        entropies.append(_entropy(p))
        spreads.append(_percentile_spread(grids[t]))
        t10.append(_top_k_mass(grids[t], 10))
        t20.append(_top_k_mass(grids[t], 20))
    mean_h = float(np.mean(entropies))
    temporal_std_mean = float(np.std(grids, axis=0).mean())
    stats = SequencePhiStats(
        sequence=seq_name,
        n_frames=T,
        grid_shape=[H, W],
        H_max_nats=H_max,
        mean_entropy_nats=mean_h,
        entropy_ratio=float(mean_h / max(H_max, 1e-9)),
        mean_percentile_spread=float(np.mean(spreads)),
        mean_top10_mass=float(np.mean(t10)),
        mean_top20_mass=float(np.mean(t20)),
        temporal_std_mean=temporal_std_mean,
    )
    stats.headroom_score = compute_headroom(stats)
    return stats


def compute_headroom(stats: SequencePhiStats) -> float:
    """Composite "IPF headroom" score in [0, 1].

    Higher = stronger spatial signal (more room for IPF to help).

    Heuristic: weighted average of
        (1 - entropy_ratio)        (concentration vs uniform)
        spread/(spread + 5)        (saturating around spread = 5)
        top10_mass                 (already in [0,1])
    """
    e = max(0.0, 1.0 - stats.entropy_ratio)
    s = stats.mean_percentile_spread / (stats.mean_percentile_spread + 5.0)
    t = stats.mean_top10_mass
    return float(0.4 * e + 0.3 * s + 0.3 * t)


def _cross_sequence_correlation(per_seq_means: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Spearman correlation of time-averaged φ across sequences (pairwise)."""
    out: Dict[str, float] = {}
    names = list(per_seq_means.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = per_seq_means[names[i]].ravel()
            b = per_seq_means[names[j]].ravel()
            if a.shape != b.shape:
                continue
            out[f"{names[i]}__vs__{names[j]}"] = _spearman(a, b)
    return out


def render_markdown(stats: List[SequencePhiStats],
                    cross_corr: Dict[str, float]) -> str:
    lines: List[str] = []
    lines.append("# D3 — φ_oracle distribution (IPF headroom)")
    lines.append("")
    lines.append(
        "Lower entropy ratio + larger spread + more top-10 mass ⇒ IPF has "
        "more headroom on this sequence."
    )
    lines.append("")
    lines.append(
        "| Sequence | n | grid | H/H_max | spread | top10 | top20 | temp σ̄ | headroom |"
    )
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for s in stats:
        lines.append(
            f"| {s.sequence} | {s.n_frames} | {s.grid_shape[0]}×{s.grid_shape[1]} | "
            f"{s.entropy_ratio:.3f} | {s.mean_percentile_spread:.1f} | "
            f"{s.mean_top10_mass:.3f} | {s.mean_top20_mass:.3f} | "
            f"{s.temporal_std_mean:.3f} | {s.headroom_score:.3f} |"
        )
    lines.append("")
    lines.append(
        "**Interpretation**: headroom > 0.5 ⇒ strong spatial signal. "
        "0.3–0.5 ⇒ moderate. < 0.3 ⇒ φ nearly uniform; IPF concept has "
        "little to exploit."
    )
    if cross_corr:
        lines.append("")
        lines.append("## Cross-sequence Spearman ρ on time-averaged φ")
        lines.append("")
        lines.append("| Pair | ρ |")
        lines.append("|---|---:|")
        for k, v in cross_corr.items():
            lines.append(f"| {k.replace('__vs__', ' vs ')} | {v:.3f} |")
        lines.append("")
        lines.append(
            "High |ρ| (> 0.7) suggests φ encodes generic frame-position bias "
            "rather than per-sequence task content. Low |ρ| (< 0.3) is desired."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="D3 — φ_oracle distribution")
    parser.add_argument("--saliency-dir", required=True, type=Path,
                        help="Parent dir containing one subdir per sequence")
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Cap frames per sequence (0 = all)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except AttributeError:
        pass

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    per_seq_stats: List[SequencePhiStats] = []
    per_seq_means: Dict[str, np.ndarray] = {}
    for seq in args.sequences:
        seq_dir = args.saliency_dir / seq
        if not seq_dir.exists():
            logger.warning("missing %s — skip", seq_dir)
            continue
        grids = _load_per_frame(seq_dir, args.max_frames)
        s = _compute_one(seq, grids)
        per_seq_stats.append(s)
        per_seq_means[seq] = grids.mean(axis=0)
        logger.info(
            "%s: H/H_max=%.3f spread=%.2f top10=%.3f headroom=%.3f",
            seq, s.entropy_ratio, s.mean_percentile_spread,
            s.mean_top10_mass, s.headroom_score,
        )

    cross_corr = _cross_sequence_correlation(per_seq_means)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "per_sequence": [asdict(s) for s in per_seq_stats],
        "cross_sequence_spearman": cross_corr,
    }, indent=2), encoding="utf-8")
    md = args.output.with_suffix(".md")
    md.write_text(render_markdown(per_seq_stats, cross_corr), encoding="utf-8")
    logger.info("Wrote %s and %s", args.output, md)


if __name__ == "__main__":
    main()
