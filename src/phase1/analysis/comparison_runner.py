"""Multi-method comparison pipeline.

Runs detection/tracking ONCE, then applies all QP methods (M0, M1, M4-IPF,
M5-M8) to the same object states. This ensures:
    1. Fair comparison (identical detections)
    2. Efficiency (detection only runs once)
    3. Reproducibility (same random seed, same objects)
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


class ComparisonPipeline:
    """Run all methods on the same video and produce comparison data.

    Usage:
        cfg = load_config("configs/default.yaml")
        comp = ComparisonPipeline(cfg, methods=["M0", "M1", "M4", "M5", "M6", "M7", "M8"])
        comp.run()
    """

    def __init__(
        self,
        cfg: IPFConfig,
        methods: Optional[list[str]] = None,
    ):
        self.cfg = cfg
        self.method_ids = methods or ["M0", "M1", "M4", "M5", "M6", "M7", "M8"]
        self.run_dir = Path(cfg.output_dir).expanduser() / cfg.run_id

    def run(self) -> dict:
        """Execute comparison pipeline.

        Returns:
            Dictionary with temporal metrics for each method.
        """
        t_start = time.time()

        # Setup directories
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for mid in self.method_ids:
            (self.run_dir / mid / "qp_vtm").mkdir(parents=True, exist_ok=True)
            (self.run_dir / mid / "qp_maps").mkdir(parents=True, exist_ok=True)

        log_file = self.run_dir / "comparison.log"
        setup_logging(self.cfg.log_level, log_file=log_file)
        save_config(self.cfg, self.run_dir / "config_snapshot.yaml")

        logger.info("=" * 60)
        logger.info("Multi-Method Comparison — Run: %s", self.cfg.run_id)
        logger.info("Methods: %s", ", ".join(self.method_ids))
        logger.info("=" * 60)

        # Initialize detector/tracker (shared across all methods)
        tracker = DetectorTracker(self.cfg.detector, self.cfg.tracker, self.cfg.mass)
        reader = VideoReader(self.cfg.video_path, max_frames=self.cfg.max_frames)
        w, h = reader.frame_size

        logger.info("Video: %dx%d, %d frames", w, h, reader.total_frames)

        # Initialize methods
        baseline_methods: dict[str, BaseQPMethod] = {}
        for mid in self.method_ids:
            if mid == "M4":
                continue  # IPF handled separately
            if mid in METHOD_REGISTRY:
                baseline_methods[mid] = create_method(
                    mid, self.cfg.qp_mapping, self.cfg.ctu, self.cfg.bounded_dynamics
                )

        # IPF components (M4)
        ipf_normalizer = TemporalNormalizer(self.cfg.normalization) if "M4" in self.method_ids else None
        ipf_controller = BoundedQPController(self.cfg.bounded_dynamics) if "M4" in self.method_ids else None

        # Per-method bounded dynamics controllers (for fair temporal comparison)
        bd_controllers: dict[str, BoundedQPController] = {}
        for mid in self.method_ids:
            if mid != "M0":  # M0 doesn't need temporal control
                bd_controllers[mid] = BoundedQPController(self.cfg.bounded_dynamics)

        # Storage
        all_frame_stats: dict[str, list[FrameQPStats]] = {mid: [] for mid in self.method_ids}
        all_qp_maps: dict[str, list[np.ndarray]] = {mid: [] for mid in self.method_ids}
        all_smoothness: dict[str, list[float]] = {mid: [] for mid in self.method_ids}

        # Process frames
        effective_total = self.cfg.max_frames or reader.total_frames
        pbar = tqdm(total=effective_total, desc="Comparing", unit="frame")

        try:
            for frame_idx, frame in reader:
                # Step 1: Detect + track (SHARED)
                objects = tracker.process_frame(frame, frame_idx)

                # Step 2: Apply each method
                for mid in self.method_ids:
                    if mid == "M4":
                        # IPF method (our proposed)
                        raw_field, nr, nc = compute_superposition_field(
                            objects, h, w, self.cfg.field, self.cfg.ctu
                        )
                        norm_field = ipf_normalizer.normalize(raw_field)
                        raw_qp = map_field_to_qp(norm_field, self.cfg.qp_mapping)
                        final_qp = ipf_controller.apply(raw_qp)
                        imp_map = norm_field
                    elif mid in baseline_methods:
                        imp_map, raw_qp_int = baseline_methods[mid].compute_qp_map(objects, h, w)
                        raw_qp = raw_qp_int.astype(np.float64)
                        # Apply bounded dynamics for fair temporal comparison
                        if mid in bd_controllers:
                            final_qp = bd_controllers[mid].apply(raw_qp)
                        else:
                            final_qp = raw_qp_int
                    else:
                        continue

                    # Collect stats
                    stats = compute_frame_stats(
                        final_qp, imp_map, frame_idx, len(objects), self.cfg.qp_mapping.mu
                    )
                    all_frame_stats[mid].append(stats)
                    all_qp_maps[mid].append(final_qp.copy())
                    all_smoothness[mid].append(compute_spatial_smoothness(final_qp))

                    # Export QP maps
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

        # Compute temporal metrics for each method
        temporal_results: dict[str, TemporalMetrics] = {}
        for mid in self.method_ids:
            if all_qp_maps[mid]:
                tm = compute_temporal_metrics(all_qp_maps[mid], mid)
                temporal_results[mid] = tm

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
            "methods": {},
        }
        for mid, tm in temporal_results.items():
            smoothness_vals = all_smoothness.get(mid, [])
            summary["methods"][mid] = {
                **asdict(tm),
                "spatial_smoothness_mean": float(np.mean(smoothness_vals)) if smoothness_vals else 0.0,
            }

        summary_path = self.run_dir / "comparison_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        # Write comparison table (readable)
        self._write_comparison_table(temporal_results, all_smoothness)

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
    ) -> None:
        """Write a human-readable comparison table."""
        lines = []
        lines.append("=" * 100)
        lines.append("COMPARISON TABLE — QP Map Methods")
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
            sm = float(np.mean(smoothness.get(mid, [0])))

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

        table_path = self.run_dir / "comparison_table.txt"
        table_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Comparison table saved: %s", table_path.name)

        # Also print to log
        for line in lines:
            logger.info(line)
