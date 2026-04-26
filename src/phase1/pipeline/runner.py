"""End-to-end Phase 1 pipeline runner.

Orchestrates the full precompute flow:
    Video → Detect/Track → Object States → IPF Field → Normalize
    → QP Map → Bounded Dynamics → Export + Visualize

This module is the single entry point for processing a video.
All configuration is read from IPFConfig; all outputs go to run_dir.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from phase1.core.config import IPFConfig, save_config
from phase1.core.schemas import ObjectState, FrameResult, RunMetadata
from phase1.io.video_reader import VideoReader
from phase1.tracking.detector_tracker import DetectorTracker
from phase1.field.importance_field import compute_superposition_field
from phase1.control.normalizer import TemporalNormalizer
from phase1.control.qp_mapper import map_field_to_qp, map_field_to_delta_qp
from phase1.control.bounded_dynamics import BoundedQPController
from phase1.export.state_writer import StateWriter
from phase1.export.qp_exporter import (
    export_qp_vtm,
    export_delta_qp_vtm,
    export_qp_csv,
)
from phase1.export.summary_writer import (
    write_run_metadata,
    write_frame_summaries,
    write_aggregate_summary,
)
from phase1.viz.visualizer import (
    save_field_overlay,
    save_qp_overlay,
    save_comparison_panel,
    save_temporal_qp_plot,
)
from phase1.utils.log import setup_logging, get_logger
from phase1.utils.timer import timer

logger = get_logger("pipeline.runner")


class Phase1Pipeline:
    """Full Phase 1 pipeline: precompute QP maps from video.

    Usage:
        cfg = load_config("configs/default.yaml")
        pipeline = Phase1Pipeline(cfg)
        pipeline.run()
    """

    def __init__(self, cfg: IPFConfig):
        self.cfg = cfg
        # Resolve output_dir: expand ~ and make absolute if relative
        output_dir = Path(cfg.output_dir).expanduser()
        if not output_dir.is_absolute():
            # Place relative paths next to the video file, or in home dir
            video_parent = Path(cfg.video_path).expanduser().resolve().parent \
                if cfg.video_path else Path.home()
            output_dir = (video_parent / output_dir).resolve()
        self.run_dir = output_dir / cfg.run_id

        # Sub-directories
        self.dir_states = self.run_dir / "object_states"
        self.dir_fields = self.run_dir / "fields"
        self.dir_qp = self.run_dir / "qp_maps"
        self.dir_qp_vtm = self.run_dir / "qp_vtm"              # legacy absolute QP
        self.dir_qp_delta_vtm = self.run_dir / "qp_vtm_delta"  # Phase 3 delta QP
        self.dir_viz = self.run_dir / "viz"
        self.dir_panels = self.run_dir / "viz" / "panels"
        self.dir_logs = self.run_dir / "logs"

        # Pipeline components (initialized in setup)
        self._tracker: Optional[DetectorTracker] = None
        self._normalizer: Optional[TemporalNormalizer] = None
        self._qp_controller: Optional[BoundedQPController] = None

    def _create_dirs(self) -> None:
        """Create all output directories."""
        for d in [
            self.dir_states, self.dir_fields, self.dir_qp,
            self.dir_qp_vtm, self.dir_qp_delta_vtm,
            self.dir_viz, self.dir_panels, self.dir_logs,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def _setup_components(self) -> None:
        """Initialize all pipeline components."""
        self._tracker = DetectorTracker(
            self.cfg.detector,
            self.cfg.tracker,
            self.cfg.mass,
        )
        self._normalizer = TemporalNormalizer(self.cfg.normalization)
        self._qp_controller = BoundedQPController(self.cfg.bounded_dynamics)

    def run(self) -> RunMetadata:
        """Execute the full pipeline.

        Returns:
            RunMetadata with aggregate statistics.
        """
        t_start = time.time()

        # Setup
        self._create_dirs()
        log_file = self.dir_logs / "pipeline.log"
        setup_logging(self.cfg.log_level, log_file=log_file)
        logger.info("=" * 60)
        logger.info("IPF Phase 1 Pipeline — Run: %s", self.cfg.run_id)
        logger.info("=" * 60)

        # Save config snapshot for reproducibility
        save_config(self.cfg, self.run_dir / "config_snapshot.yaml")

        # Initialize components
        self._setup_components()

        # Open video
        reader = VideoReader(self.cfg.video_path, max_frames=self.cfg.max_frames)
        w, h = reader.frame_size

        metadata = RunMetadata(
            run_id=self.cfg.run_id,
            video_path=self.cfg.video_path,
            config_path="",
            total_frames=reader.total_frames,
            frame_width=w,
            frame_height=h,
            fps=reader.fps,
            ctu_size=self.cfg.ctu.ctu_size,
        )

        # Compute CTU grid dimensions
        ctu_size = self.cfg.ctu.ctu_size
        n_ctu_rows = int(np.ceil(h / ctu_size))
        n_ctu_cols = int(np.ceil(w / ctu_size))
        metadata.ctu_rows = n_ctu_rows
        metadata.ctu_cols = n_ctu_cols

        logger.info("Video: %dx%d @ %.1f fps, %d frames", w, h, reader.fps, reader.total_frames)
        logger.info("CTU grid: %d x %d (ctu_size=%d)", n_ctu_rows, n_ctu_cols, ctu_size)

        # Process frames
        frame_results: list[FrameResult] = []
        state_writer = StateWriter(self.dir_states / "object_states.jsonl")

        effective_total = self.cfg.max_frames or reader.total_frames
        pbar = tqdm(total=effective_total, desc="Processing", unit="frame")

        try:
            for frame_idx, frame in reader:
                with timer(f"frame_{frame_idx}", log=False) as t:
                    result = self._process_single_frame(
                        frame, frame_idx, h, w, state_writer
                    )
                result.processing_time_ms = t["elapsed_ms"]
                frame_results.append(result)
                pbar.update(1)
                pbar.set_postfix(
                    objs=result.n_objects,
                    qp_mean=f"{np.mean(result.final_qp_map):.1f}" if result.final_qp_map is not None else "N/A",
                    ms=f"{t['elapsed_ms']:.0f}",
                )
        finally:
            state_writer.close()
            reader.release()
            pbar.close()

        # Post-processing
        t_total = time.time() - t_start
        metadata.processed_frames = len(frame_results)
        metadata.total_time_s = round(t_total, 2)
        metadata.avg_frame_time_ms = round(
            np.mean([r.processing_time_ms for r in frame_results]), 2
        ) if frame_results else 0.0
        metadata.avg_objects_per_frame = round(
            np.mean([r.n_objects for r in frame_results]), 2
        ) if frame_results else 0.0
        metadata.status = "completed"

        # Save summaries
        write_run_metadata(metadata, self.run_dir)
        write_frame_summaries(frame_results, self.run_dir)
        write_aggregate_summary(frame_results, metadata, self.run_dir)

        # Generate temporal stability plot
        save_temporal_qp_plot(frame_results, self.run_dir / "viz" / "temporal_qp.png")

        logger.info("=" * 60)
        logger.info("Pipeline complete: %d frames in %.1f s (%.1f ms/frame)",
                     len(frame_results), t_total,
                     metadata.avg_frame_time_ms)
        logger.info("Output: %s", self.run_dir)
        logger.info("=" * 60)

        return metadata

    def _process_single_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        frame_h: int,
        frame_w: int,
        state_writer: StateWriter,
    ) -> FrameResult:
        """Process one frame through the full IPF pipeline.

        Steps:
            1. Detect + track objects
            2. Compute importance field (superposition)
            3. Temporal normalization
            4. Map to raw QP
            5. Apply bounded dynamics
            6. Export + visualize
        """
        result = FrameResult(frame_idx=frame_idx)

        # --- Step 1: Detection + Tracking ---
        objects = self._tracker.process_frame(frame, frame_idx)
        result.objects = objects
        result.n_objects = len(objects)

        # Save object states
        if self.cfg.output.save_object_states:
            state_writer.write_frame_objects(objects)

        # --- Step 2: Importance Field ---
        raw_field, n_rows, n_cols = compute_superposition_field(
            objects, frame_h, frame_w, self.cfg.field, self.cfg.ctu,
        )
        result.raw_field = raw_field

        # --- Step 3: Temporal Normalization ---
        norm_field = self._normalizer.normalize(raw_field)
        result.normalized_field = norm_field

        # --- Step 4: Raw QP Mapping ---
        # Legacy absolute-QP path (kept for pilot v1 compatibility).
        raw_qp = map_field_to_qp(norm_field, self.cfg.qp_mapping)
        result.raw_qp_map = raw_qp

        # Phase 3 delta-QP path (Q_base-agnostic). Identical asymmetric
        # power-law mapping, just without the Q_base offset.
        raw_delta_qp = map_field_to_delta_qp(norm_field, self.cfg.qp_mapping)

        # --- Step 5: Bounded QP Dynamics ---
        # Apply temporal low-pass + slew-rate limiting to the absolute map.
        # For the delta map we apply the SAME smoothing by subtracting qp_base
        # so the two outputs remain consistent (they differ only by a constant).
        final_qp = self._qp_controller.apply(raw_qp)
        final_delta_qp = final_qp - float(self.cfg.qp_mapping.qp_base)
        result.final_qp_map = final_qp

        # --- Step 6: Export QP Maps ---
        if self.cfg.output.save_qp_vtm:
            export_qp_vtm(
                final_qp,
                self.dir_qp_vtm / f"qp_{frame_idx:06d}.txt",
                frame_idx,
            )
        if self.cfg.output.save_qp_delta_vtm:
            export_delta_qp_vtm(
                final_delta_qp,
                self.dir_qp_delta_vtm / f"qp_{frame_idx:06d}.txt",
                frame_idx,
                delta_min=self.cfg.qp_mapping.delta_clip_min,
                delta_max=self.cfg.qp_mapping.delta_clip_max,
            )

        if self.cfg.output.save_qp_csv:
            export_qp_csv(final_qp, self.dir_qp / f"qp_{frame_idx:06d}.csv")

        # --- Step 7: Visualization ---
        should_viz = (frame_idx % self.cfg.viz.save_every_n_frames == 0)
        if should_viz:
            if self.cfg.viz.save_field_maps:
                save_field_overlay(
                    frame, norm_field, objects,
                    self.dir_viz / f"field_{frame_idx:06d}.jpg",
                    self.cfg.ctu.ctu_size, self.cfg.viz,
                )

            if self.cfg.viz.save_qp_overlays:
                save_qp_overlay(
                    frame, final_qp, objects,
                    self.dir_viz / f"qp_{frame_idx:06d}.jpg",
                    self.cfg.ctu.ctu_size,
                    (self.cfg.bounded_dynamics.qp_min, self.cfg.bounded_dynamics.qp_max),
                    self.cfg.viz,
                )

            save_comparison_panel(
                frame, norm_field, final_qp, objects,
                self.dir_panels / f"panel_{frame_idx:06d}.png",
                frame_idx, self.cfg.ctu.ctu_size, self.cfg.viz,
            )

        return result
