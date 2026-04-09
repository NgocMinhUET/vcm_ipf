"""Multi-method comparison pipeline.

Runs detection/tracking ONCE, then applies all QP methods (M0, M1, M4-IPF,
M5-M8) to the same object states. Ensures strict fairness:
    1. Identical detections (shared tracker)
    2. Same QP mapping function (map_field_to_qp) for all methods
    3. Same bounded dynamics controller settings
    4. Only the importance map computation differs per method

This design guarantees that any measured difference in temporal stability
or spatial smoothness is attributable to the importance map formulation,
not to pipeline differences.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from phase1.core.config import IPFConfig, save_config
from phase1.core.schemas import ObjectState, RunMetadata
from phase1.io.video_reader import VideoReader
from phase1.tracking.detector_tracker import DetectorTracker
from phase1.field.importance_field import compute_superposition_field
from phase1.control.normalizer import TemporalNormalizer
from phase1.control.qp_mapper import map_field_to_qp
from phase1.control.bounded_dynamics import BoundedQPController
from phase1.baselines.qp_methods import (
    BaseQPMethod, create_method, METHOD_REGISTRY,
)
from phase1.analysis.metrics import (
    FrameQPStats, TemporalMetrics,
    compute_frame_stats, compute_temporal_metrics, compute_spatial_smoothness,
)
from phase1.export.qp_exporter import export_qp_vtm
from phase1.utils.log import setup_logging, get_logger
from phase1.utils.timer import timer

logger = get_logger("analysis.comparison")

DEFAULT_WARMUP_SKIP = 10


class ComparisonPipeline:
    """Run all methods on the same video and produce comparison data.

    Fairness protocol:
        - All methods share the same detector/tracker output
        - All methods (except M0) go through the same pipeline:
          importance_map → map_field_to_qp → BoundedQPController
        - IPF additionally uses TemporalNormalizer before QP mapping
          (this is part of its method, not a pipeline advantage)
        - Metrics are computed both with and without warmup exclusion

    Usage:
        cfg = load_config("configs/default.yaml")
        comp = ComparisonPipeline(cfg, methods=["M0","M1","M4","M5","M6","M7","M8"])
        comp.run()
    """

    def __init__(
        self,
        cfg: IPFConfig,
        methods: Optional[list[str]] = None,
        warmup_skip: int = DEFAULT_WARMUP_SKIP,
    ):
        self.cfg = cfg
        self.method_ids = methods or ["M0", "M1", "M4", "M5", "M6", "M7", "M8"]
        self.run_dir = Path(cfg.output_dir).expanduser() / cfg.run_id
        self.warmup_skip = warmup_skip

    def run(self) -> dict:
        """Execute comparison pipeline.

        Returns:
            Dictionary with temporal metrics for each method.
        """
        t_start = time.time()

        self.run_dir.mkdir(parents=True, exist_ok=True)
        for mid in self.method_ids:
            (self.run_dir / mid / "qp_vtm").mkdir(parents=True, exist_ok=True)

        log_file = self.run_dir / "comparison.log"
        setup_logging(self.cfg.log_level, log_file=log_file)
        save_config(self.cfg, self.run_dir / "config_snapshot.yaml")

        logger.info("=" * 60)
        logger.info("Multi-Method Comparison — Run: %s", self.cfg.run_id)
        logger.info("Methods: %s", ", ".join(self.method_ids))
        logger.info("Warmup skip: %d frames (for steady-state metrics)", self.warmup_skip)
        logger.info("=" * 60)

        tracker = DetectorTracker(self.cfg.detector, self.cfg.tracker, self.cfg.mass)
        reader = VideoReader(self.cfg.video_path, max_frames=self.cfg.max_frames)
        w, h = reader.frame_size

        logger.info("Video: %dx%d, %d frames", w, h, reader.total_frames)

        # Initialize baseline importance-map generators
        bl_cfg = self.cfg.baselines
        baseline_methods: dict[str, BaseQPMethod] = {}
        for mid in self.method_ids:
            if mid in ("M4", "M0"):
                continue
            if mid not in METHOD_REGISTRY:
                continue
            kwargs: dict = {}
            if mid == "M5":
                kwargs["sigma_factor"] = bl_cfg.m5_sigma_factor
            elif mid == "M6":
                kwargs["alpha"] = bl_cfg.m6_alpha
            elif mid == "M7":
                kwargs["cutoff_factor"] = bl_cfg.m7_cutoff_factor
            elif mid == "M8":
                kwargs["blur_factor"] = bl_cfg.m8_blur_factor
            baseline_methods[mid] = create_method(
                mid, self.cfg.qp_mapping, self.cfg.ctu, self.cfg.bounded_dynamics,
                **kwargs,
            )

        # IPF-specific components (M4)
        ipf_normalizer = TemporalNormalizer(self.cfg.normalization) if "M4" in self.method_ids else None

        # Per-method bounded dynamics (all non-M0 methods share same config)
        bd_controllers: dict[str, BoundedQPController] = {}
        for mid in self.method_ids:
            if mid != "M0":
                bd_controllers[mid] = BoundedQPController(self.cfg.bounded_dynamics)

        # Storage
        all_frame_stats: dict[str, list[FrameQPStats]] = {mid: [] for mid in self.method_ids}
        all_qp_maps: dict[str, list[np.ndarray]] = {mid: [] for mid in self.method_ids}
        all_smoothness: dict[str, list[float]] = {mid: [] for mid in self.method_ids}

        effective_total = self.cfg.max_frames or reader.total_frames
        pbar = tqdm(total=effective_total, desc="Comparing", unit="frame")

        try:
            for frame_idx, frame in reader:
                objects = tracker.process_frame(frame, frame_idx)

                for mid in self.method_ids:
                    if mid == "M0":
                        # Uniform QP anchor: no importance, constant QP
                        from phase1.field.importance_field import build_ctu_grid
                        _, _, nr, nc = build_ctu_grid(h, w, self.cfg.ctu.ctu_size)
                        imp_map = np.zeros((nr, nc), dtype=np.float64)
                        final_qp = np.full((nr, nc), self.cfg.qp_mapping.qp_base, dtype=np.int32)

                    elif mid == "M4":
                        # IPF: raw field → EMA normalization → QP mapping → bounded dynamics
                        raw_field, nr, nc = compute_superposition_field(
                            objects, h, w, self.cfg.field, self.cfg.ctu
                        )
                        imp_map = ipf_normalizer.normalize(raw_field)
                        raw_qp = map_field_to_qp(imp_map, self.cfg.qp_mapping)
                        final_qp = bd_controllers[mid].apply(raw_qp)

                    elif mid in baseline_methods:
                        # Baselines: importance map → SAME QP mapping → SAME bounded dynamics
                        imp_map = baseline_methods[mid].compute_importance_map(objects, h, w)
                        raw_qp = map_field_to_qp(imp_map, self.cfg.qp_mapping)
                        final_qp = bd_controllers[mid].apply(raw_qp)

                    else:
                        continue

                    stats = compute_frame_stats(
                        final_qp, imp_map, frame_idx, len(objects), self.cfg.qp_mapping.mu
                    )
                    all_frame_stats[mid].append(stats)
                    all_qp_maps[mid].append(final_qp.copy())
                    all_smoothness[mid].append(compute_spatial_smoothness(final_qp))

                    export_qp_vtm(
                        final_qp,
                        self.run_dir / mid / "qp_vtm" / f"qp_{frame_idx:06d}.txt",
                        frame_idx,
                    )

                pbar.update(1)
                pbar.set_postfix(objs=len(objects))
        finally:
            reader.release()
            pbar.close()

        # Compute temporal metrics: both full-sequence and steady-state
        temporal_full: dict[str, TemporalMetrics] = {}
        temporal_steady: dict[str, TemporalMetrics] = {}
        for mid in self.method_ids:
            if all_qp_maps[mid]:
                temporal_full[mid] = compute_temporal_metrics(all_qp_maps[mid], mid, skip_first_n=0)
                temporal_steady[mid] = compute_temporal_metrics(
                    all_qp_maps[mid], mid, skip_first_n=self.warmup_skip
                )

        # Write per-method frame stats CSV
        for mid in self.method_ids:
            if all_frame_stats[mid]:
                csv_path = self.run_dir / mid / "frame_stats.csv"
                rows = [asdict(s) for s in all_frame_stats[mid]]
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)

        # Write comparison summary
        summary = {
            "run_id": self.cfg.run_id,
            "video": self.cfg.video_path,
            "n_frames": effective_total,
            "warmup_skip": self.warmup_skip,
            "methods_full": {},
            "methods_steady": {},
        }
        for mid in self.method_ids:
            sm_vals = all_smoothness.get(mid, [])
            sm_mean = float(np.mean(sm_vals)) if sm_vals else 0.0
            sm_steady = float(np.mean(sm_vals[self.warmup_skip:])) if len(sm_vals) > self.warmup_skip else sm_mean

            if mid in temporal_full:
                summary["methods_full"][mid] = {
                    **asdict(temporal_full[mid]),
                    "spatial_smoothness_mean": sm_mean,
                }
            if mid in temporal_steady:
                summary["methods_steady"][mid] = {
                    **asdict(temporal_steady[mid]),
                    "spatial_smoothness_mean": sm_steady,
                }

        summary_path = self.run_dir / "comparison_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        # Write both tables
        self._write_comparison_table(
            temporal_full, all_smoothness, "FULL SEQUENCE", "comparison_table_full.txt", 0
        )
        self._write_comparison_table(
            temporal_steady, all_smoothness, f"STEADY STATE (skip first {self.warmup_skip})",
            "comparison_table_steady.txt", self.warmup_skip,
        )

        t_total = time.time() - t_start
        logger.info("=" * 60)
        logger.info("Comparison complete in %.1f s", t_total)
        logger.info("Output: %s", self.run_dir)
        logger.info("=" * 60)

        return summary

    def _write_comparison_table(
        self,
        temporal: dict[str, TemporalMetrics],
        smoothness: dict[str, list[float]],
        title: str,
        filename: str,
        skip_n: int,
    ) -> None:
        """Write a human-readable comparison table."""
        lines = []
        lines.append("=" * 100)
        lines.append(f"COMPARISON TABLE — {title}")
        lines.append("=" * 100)
        lines.append(
            f"{'Method':<25} {'QP Mean':>8} {'QP Std':>8} "
            f"{'Δt Mean':>8} {'Δt Max':>8} "
            f"{'CTU σ_t':>8} {'Jitter':>8} {'Smooth':>8}"
        )
        lines.append("-" * 100)

        for mid in self.method_ids:
            if mid not in temporal:
                continue
            tm = temporal[mid]
            sm_vals = smoothness.get(mid, [0])
            sm_slice = sm_vals[skip_n:] if len(sm_vals) > skip_n else sm_vals
            sm = float(np.mean(sm_slice))

            name = mid
            if mid == "M4":
                name = "M4 (IPF-Ours)"

            lines.append(
                f"{name:<25} {tm.qp_mean_of_means:>8.2f} {tm.qp_std_of_means:>8.3f} "
                f"{tm.mean_frame_to_frame_delta:>8.3f} {tm.max_frame_to_frame_delta:>8.3f} "
                f"{tm.per_ctu_temporal_std_mean:>8.3f} {tm.temporal_jitter_index:>8.3f} {sm:>8.3f}"
            )

        lines.append("-" * 100)
        lines.append("Legend:")
        lines.append("  QP Mean    = average QP across all frames")
        lines.append("  QP Std     = std of per-frame mean QP (lower = more stable)")
        lines.append("  Δt Mean    = mean frame-to-frame QP mean change (lower = smoother)")
        lines.append("  Δt Max     = max single-frame QP jump")
        lines.append("  CTU σ_t    = mean per-CTU temporal std (lower = less flicker)")
        lines.append("  Jitter     = mean per-CTU frame-to-frame |ΔQP| (lower = better)")
        lines.append("  Smooth     = spatial gradient (lower = smoother QP transitions)")
        lines.append("=" * 100)

        table_path = self.run_dir / filename
        table_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Table saved: %s", filename)

        for line in lines:
            logger.info(line)
