"""Multi-method comparison pipeline with ablation support.

Runs detection/tracking ONCE, then applies all QP methods (M0, M1, M4-IPF,
M5-M8) and optionally ablation variants (A1, A3, A4, A6, M2, M3) to the
same object states. Ensures strict fairness:
    1. Identical detections (shared tracker)
    2. Same QP mapping function (map_field_to_qp) for all methods
    3. Same bounded dynamics controller settings
    4. Only the importance map computation differs per method

Ablation variants isolate each IPF component:
    A1: No EMA normalization (per-frame percentile normalize)
    A3: Single strongest object (no multi-object superposition)
    A4: Max superposition instead of sum
    A6: Gaussian kernel replacement (same mass/EMA/bounded)
    M2: Binary ROI + EMA temporal smoothing (tracking-only, no IPF field)
    M3: IPF spatial field only (no EMA, no bounded dynamics)
"""

from __future__ import annotations

import csv
import json
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from phase1.core.config import IPFConfig, save_config
from phase1.core.schemas import ObjectState, RunMetadata
from phase1.io.video_reader import VideoReader
from phase1.tracking.detector_tracker import DetectorTracker
from phase1.field.importance_field import (
    build_ctu_grid,
    compute_superposition_field,
    compute_gaussian_superposition_field,
    compute_importance_mass,
)
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

ABLATION_VARIANTS = {
    "A1": {
        "name": "IPF no-EMA",
        "description": "IPF field + per-frame normalize (no temporal EMA smoothing)",
        "use_ema": False,
        "use_bounded": True,
        "objects": "all",
        "superposition": "sum",
        "kernel": "ipf",
    },
    "A3": {
        "name": "IPF single-obj",
        "description": "IPF with only the single strongest object (no superposition)",
        "use_ema": True,
        "use_bounded": True,
        "objects": "single",
        "superposition": "sum",
        "kernel": "ipf",
    },
    "A4": {
        "name": "IPF max-super",
        "description": "IPF with max superposition instead of sum",
        "use_ema": True,
        "use_bounded": True,
        "objects": "all",
        "superposition": "max",
        "kernel": "ipf",
    },
    "A6": {
        "name": "IPF gauss-kern",
        "description": "Gaussian kernel with same mass/EMA/bounded as IPF",
        "use_ema": True,
        "use_bounded": True,
        "objects": "all",
        "superposition": "sum",
        "kernel": "gaussian",
    },
    "M2": {
        "name": "Tracking-only",
        "description": "Binary ROI + EMA normalization + bounded dynamics (no IPF field)",
        "use_ema": True,
        "use_bounded": True,
        "objects": "all",
        "superposition": "sum",
        "kernel": "binary",
    },
    "M3": {
        "name": "IPF spatial-only",
        "description": "IPF field with per-frame normalize, no bounded dynamics",
        "use_ema": False,
        "use_bounded": False,
        "objects": "all",
        "superposition": "sum",
        "kernel": "ipf",
    },
}


def _select_objects(objects: list[ObjectState], mode: str) -> list[ObjectState]:
    """Select objects based on ablation mode."""
    if mode == "single" and objects:
        best = max(objects, key=lambda o: compute_importance_mass(o))
        return [best]
    return objects


def _compute_binary_field(
    objects: list[ObjectState],
    frame_h: int,
    frame_w: int,
    ctu_size: int,
) -> tuple[np.ndarray, int, int]:
    """Compute binary ROI field at CTU resolution (for M2 ablation)."""
    grid_x, grid_y, nr, nc = build_ctu_grid(frame_h, frame_w, ctu_size)
    field = np.zeros((nr, nc), dtype=np.float64)
    ctu = ctu_size

    for obj in objects:
        x1 = obj.x_center - obj.width / 2
        y1 = obj.y_center - obj.height / 2
        x2 = obj.x_center + obj.width / 2
        y2 = obj.y_center + obj.height / 2

        c_start = max(0, int(x1 // ctu))
        c_end = min(nc - 1, int(x2 // ctu))
        r_start = max(0, int(y1 // ctu))
        r_end = min(nr - 1, int(y2 // ctu))

        field[r_start:r_end + 1, c_start:c_end + 1] = 1.0

    return field, nr, nc


def _normalize_per_frame(field_map: np.ndarray, p_low: float = 5.0, p_high: float = 95.0) -> np.ndarray:
    """Per-frame percentile normalization (no temporal smoothing).

    Same percentile bounds as EMA normalizer but applied independently
    per frame, isolating the EMA's temporal contribution.
    """
    p_lo = float(np.percentile(field_map, p_low))
    p_hi = float(np.percentile(field_map, p_high))
    denom = p_hi - p_lo + 1e-8
    return np.clip((field_map - p_lo) / denom, 0.0, 1.0)


class ComparisonPipeline:
    """Run all methods on the same video and produce comparison data.

    Fairness protocol:
        - All methods share the same detector/tracker output
        - All methods (except M0) go through the same pipeline:
          importance_map -> map_field_to_qp -> BoundedQPController
        - IPF additionally uses TemporalNormalizer before QP mapping
          (this is part of its method, not a pipeline advantage)
        - Ablation variants modify exactly ONE component of the IPF pipeline
        - Metrics are computed both with and without warmup exclusion
    """

    def __init__(
        self,
        cfg: IPFConfig,
        methods: Optional[list[str]] = None,
        warmup_skip: int = DEFAULT_WARMUP_SKIP,
        enable_ablation: bool = False,
    ):
        self.cfg = cfg
        self.method_ids = methods or ["M0", "M1", "M4", "M5", "M6", "M7", "M8"]
        self.run_dir = Path(cfg.output_dir).expanduser() / cfg.run_id
        self.warmup_skip = warmup_skip

        if enable_ablation or cfg.ablation.enabled:
            for vid in cfg.ablation.variants:
                if vid not in self.method_ids and vid in ABLATION_VARIANTS:
                    self.method_ids.append(vid)

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
        ablation_ids = [m for m in self.method_ids if m in ABLATION_VARIANTS]
        if ablation_ids:
            logger.info("Ablation variants: %s", ", ".join(ablation_ids))
            for aid in ablation_ids:
                logger.info("  %s: %s", aid, ABLATION_VARIANTS[aid]["description"])
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
            if mid in ("M4", "M0") or mid in ABLATION_VARIANTS:
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

        # IPF-specific components (M4) + ablation normalizers
        ipf_normalizer = TemporalNormalizer(self.cfg.normalization) if "M4" in self.method_ids else None

        ablation_normalizers: dict[str, TemporalNormalizer] = {}
        for mid in self.method_ids:
            if mid in ABLATION_VARIANTS and ABLATION_VARIANTS[mid]["use_ema"]:
                ablation_normalizers[mid] = TemporalNormalizer(self.cfg.normalization)

        # Per-method bounded dynamics (all non-M0 methods share same config)
        bd_controllers: dict[str, BoundedQPController] = {}
        for mid in self.method_ids:
            if mid == "M0":
                continue
            if mid in ABLATION_VARIANTS and not ABLATION_VARIANTS[mid]["use_bounded"]:
                continue
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
                    imp_map, final_qp = self._process_method(
                        mid, objects, h, w,
                        baseline_methods, ipf_normalizer,
                        ablation_normalizers, bd_controllers,
                    )

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

    def _process_method(
        self,
        mid: str,
        objects: list[ObjectState],
        h: int,
        w: int,
        baseline_methods: dict[str, BaseQPMethod],
        ipf_normalizer: Optional[TemporalNormalizer],
        ablation_normalizers: dict[str, TemporalNormalizer],
        bd_controllers: dict[str, BoundedQPController],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Process a single method for one frame. Returns (importance_map, final_qp)."""

        if mid == "M0":
            _, _, nr, nc = build_ctu_grid(h, w, self.cfg.ctu.ctu_size)
            imp_map = np.zeros((nr, nc), dtype=np.float64)
            final_qp = np.full((nr, nc), self.cfg.qp_mapping.qp_base, dtype=np.int32)

        elif mid == "M4":
            raw_field, nr, nc = compute_superposition_field(
                objects, h, w, self.cfg.field, self.cfg.ctu
            )
            imp_map = ipf_normalizer.normalize(raw_field)
            raw_qp = map_field_to_qp(imp_map, self.cfg.qp_mapping)
            final_qp = bd_controllers[mid].apply(raw_qp)

        elif mid in baseline_methods:
            imp_map = baseline_methods[mid].compute_importance_map(objects, h, w)
            raw_qp = map_field_to_qp(imp_map, self.cfg.qp_mapping)
            final_qp = bd_controllers[mid].apply(raw_qp)

        elif mid in ABLATION_VARIANTS:
            imp_map, final_qp = self._process_ablation(
                mid, objects, h, w, ablation_normalizers, bd_controllers,
            )

        else:
            _, _, nr, nc = build_ctu_grid(h, w, self.cfg.ctu.ctu_size)
            imp_map = np.zeros((nr, nc), dtype=np.float64)
            final_qp = np.full((nr, nc), self.cfg.qp_mapping.qp_base, dtype=np.int32)

        return imp_map, final_qp

    def _process_ablation(
        self,
        mid: str,
        objects: list[ObjectState],
        h: int,
        w: int,
        ablation_normalizers: dict[str, TemporalNormalizer],
        bd_controllers: dict[str, BoundedQPController],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Process an ablation variant for one frame."""
        abl = ABLATION_VARIANTS[mid]

        # 1. Object selection
        objs = _select_objects(objects, abl["objects"])

        # 2. Compute raw field based on kernel type
        kernel = abl["kernel"]
        if kernel == "binary":
            raw_field, nr, nc = _compute_binary_field(objs, h, w, self.cfg.ctu.ctu_size)
        elif kernel == "gaussian":
            raw_field, nr, nc = compute_gaussian_superposition_field(
                objs, h, w, self.cfg.field, self.cfg.ctu
            )
        else:
            field_cfg = self.cfg.field
            if abl["superposition"] != field_cfg.superposition:
                field_cfg = field_cfg.model_copy()
                field_cfg.superposition = abl["superposition"]
            raw_field, nr, nc = compute_superposition_field(
                objs, h, w, field_cfg, self.cfg.ctu
            )

        # 3. Normalization: EMA or per-frame
        if abl["use_ema"] and mid in ablation_normalizers:
            imp_map = ablation_normalizers[mid].normalize(raw_field)
        else:
            imp_map = _normalize_per_frame(
                raw_field,
                self.cfg.normalization.percentile_low,
                self.cfg.normalization.percentile_high,
            )

        # 4. QP mapping (same for all)
        raw_qp = map_field_to_qp(imp_map, self.cfg.qp_mapping)

        # 5. Bounded dynamics or direct round+clip
        if abl["use_bounded"] and mid in bd_controllers:
            final_qp = bd_controllers[mid].apply(raw_qp)
        else:
            final_qp = np.clip(
                np.round(raw_qp),
                self.cfg.bounded_dynamics.qp_min,
                self.cfg.bounded_dynamics.qp_max,
            ).astype(np.int32)

        return imp_map, final_qp

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
        lines.append("=" * 110)
        lines.append(f"COMPARISON TABLE — {title}")
        lines.append("=" * 110)
        lines.append(
            f"{'Method':<25} {'QP Mean':>8} {'QP Std':>8} "
            f"{'Dt Mean':>8} {'Dt Max':>8} "
            f"{'CTU s_t':>8} {'Jitter':>8} {'Smooth':>8}"
        )
        lines.append("-" * 110)

        baseline_ids = ["M0", "M1", "M4", "M5", "M6", "M7", "M8"]
        ablation_ids = sorted([m for m in self.method_ids if m in ABLATION_VARIANTS])

        has_ablation = bool(ablation_ids)

        for mid in baseline_ids:
            if mid not in temporal:
                continue
            self._append_method_row(lines, mid, temporal[mid], smoothness, skip_n)

        if has_ablation:
            lines.append("-" * 110)
            lines.append("ABLATION VARIANTS (each modifies ONE component of M4):")
            lines.append("-" * 110)
            for mid in ablation_ids:
                if mid not in temporal:
                    continue
                self._append_method_row(lines, mid, temporal[mid], smoothness, skip_n)

        lines.append("-" * 110)
        lines.append("Legend:")
        lines.append("  QP Mean    = average QP across all frames")
        lines.append("  QP Std     = std of per-frame mean QP (lower = more stable)")
        lines.append("  Dt Mean    = mean frame-to-frame QP mean change (lower = smoother)")
        lines.append("  Dt Max     = max single-frame QP jump")
        lines.append("  CTU s_t    = mean per-CTU temporal std (lower = less flicker)")
        lines.append("  Jitter     = mean per-CTU frame-to-frame |dQP| (lower = better)")
        lines.append("  Smooth     = spatial gradient (lower = smoother QP transitions)")
        if has_ablation:
            lines.append("")
            lines.append("Ablation key:")
            for aid in ablation_ids:
                abl = ABLATION_VARIANTS[aid]
                lines.append(f"  {aid:<6} = {abl['description']}")
        lines.append("=" * 110)

        table_path = self.run_dir / filename
        table_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Table saved: %s", filename)

        for line in lines:
            logger.info(line)

    def _append_method_row(
        self,
        lines: list[str],
        mid: str,
        tm: TemporalMetrics,
        smoothness: dict[str, list[float]],
        skip_n: int,
    ) -> None:
        """Append a single method row to the comparison table."""
        sm_vals = smoothness.get(mid, [0])
        sm_slice = sm_vals[skip_n:] if len(sm_vals) > skip_n else sm_vals
        sm = float(np.mean(sm_slice))

        name = mid
        if mid == "M4":
            name = "M4 (IPF-Ours)"
        elif mid in ABLATION_VARIANTS:
            name = f"{mid} ({ABLATION_VARIANTS[mid]['name']})"

        lines.append(
            f"{name:<25} {tm.qp_mean_of_means:>8.2f} {tm.qp_std_of_means:>8.3f} "
            f"{tm.mean_frame_to_frame_delta:>8.3f} {tm.max_frame_to_frame_delta:>8.3f} "
            f"{tm.per_ctu_temporal_std_mean:>8.3f} {tm.temporal_jitter_index:>8.3f} {sm:>8.3f}"
        )
