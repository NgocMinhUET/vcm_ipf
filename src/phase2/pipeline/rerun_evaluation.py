"""Re-run PSNR and Task-accuracy evaluation on existing encoded outputs.

Use case: the encoding ran successfully (the bitstreams and recon.yuv files
on disk are valid VVC outputs) but the post-encode evaluation logic had bugs
(e.g. PSNR-ROI = PSNR-full, mAP = 0).  Rather than waste 30+ hours re-encoding,
this script reads the existing files in `<output_dir>/<run_id>/recon.yuv` and
recomputes only the evaluation metrics with the fixed code.

Usage:
    python -m phase2.pipeline.rerun_evaluation \
        --config configs/pilot.yaml \
        [--methods M0 M1 M4 M5 M6] \
        [--sequences MOT17-04-DPM]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from phase2.core.config import load_phase2_config, Phase2Config
from phase2.evaluation.psnr import compute_sequence_psnr
from phase2.evaluation.task_accuracy import TaskEvaluator
from phase2.encoding.yuv_utils import save_frames_from_yuv
from phase2.pipeline.encode_pipeline import EncodingPipeline


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def rerun(
    cfg: Phase2Config,
    only_methods: Optional[List[str]] = None,
    only_sequences: Optional[List[str]] = None,
) -> None:
    """Re-evaluate every existing run; do NOT re-encode."""
    pipeline = EncodingPipeline(cfg)
    logger = logging.getLogger("phase2.rerun")

    summary_path = pipeline.output_dir / "experiment_summary.json"
    if not summary_path.exists():
        logger.error("No experiment_summary.json at %s — nothing to re-evaluate.", summary_path)
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    logger.info("Re-evaluating %d runs from %s", len(summary["results"]), summary_path)

    n_done = 0
    n_skipped = 0
    n_failed = 0

    for entry in summary["results"]:
        seq_name = entry["sequence"]
        method = entry["method"]
        qp_base = entry["qp_base"]
        run_id = entry["run_id"]

        if only_methods and method not in only_methods:
            n_skipped += 1
            continue
        if only_sequences and seq_name not in only_sequences:
            n_skipped += 1
            continue

        seq_cfg = next((s for s in cfg.sequences if s.name == seq_name), None)
        if seq_cfg is None:
            logger.warning("[%s] sequence not in config — skip", run_id)
            n_skipped += 1
            continue

        run_dir = pipeline.output_dir / run_id
        recon_path = run_dir / "recon.yuv"
        if not recon_path.exists():
            logger.warning("[%s] recon.yuv missing — skip", run_id)
            n_skipped += 1
            continue

        logger.info("[%s] re-evaluating", run_id)

        # Reference boxes (cached per sequence; one YOLO pass per sequence)
        ref_boxes = pipeline._get_reference_boxes(seq_cfg)

        # ---- Re-compute PSNR (full + ROI + BG) ----
        try:
            psnr_result = compute_sequence_psnr(
                original_yuv=seq_cfg.yuv_path,
                reconstructed_yuv=str(recon_path),
                width=seq_cfg.width,
                height=seq_cfg.height,
                n_frames=seq_cfg.n_frames,
                boxes_per_frame=ref_boxes,
                expansion=cfg.evaluation.roi_expansion,
            )
            entry["psnr_y_full"] = psnr_result.psnr_y_full_mean
            entry["psnr_y_roi"] = psnr_result.psnr_y_roi_mean
            entry["psnr_y_bg"] = psnr_result.psnr_y_bg_mean
        except Exception as exc:
            logger.error("[%s] PSNR re-compute failed: %s", run_id, exc)
            n_failed += 1
            continue

        # ---- Re-compute task accuracy (mAP) ----
        if cfg.evaluation.compute_task_accuracy and pipeline.task_evaluator:
            dec_frames_dir = run_dir / "decoded_frames"
            if not dec_frames_dir.exists() or not any(dec_frames_dir.iterdir()):
                logger.info("[%s]   extracting decoded frames...", run_id)
                save_frames_from_yuv(
                    str(recon_path), str(dec_frames_dir),
                    seq_cfg.width, seq_cfg.height,
                    seq_cfg.n_frames,
                    seq_cfg.original_width, seq_cfg.original_height,
                )

            try:
                task_result = pipeline.task_evaluator.compute_task_metrics(
                    decoded_frames_dir=str(dec_frames_dir),
                    reference_frames_dir=seq_cfg.frames_dir,
                    n_frames=seq_cfg.n_frames,
                )
                entry["mAP50"] = task_result.mAP50
                entry["mAP75"] = task_result.mAP75
                entry["mAP50_95"] = task_result.mAP50_95
                entry["mean_detections_per_frame"] = task_result.mean_detections_per_frame
            except Exception as exc:
                logger.error("[%s] task re-compute failed: %s", run_id, exc)

        n_done += 1
        logger.info(
            "[%s]   PSNR full=%.2f roi=%.2f bg=%.2f | mAP50=%.3f",
            run_id,
            entry.get("psnr_y_full", 0),
            entry.get("psnr_y_roi", 0),
            entry.get("psnr_y_bg", 0),
            entry.get("mAP50", 0),
        )

        # Persist incrementally so a crash mid-loop doesn't lose progress.
        backup_path = summary_path.with_suffix(".reeval.json")
        backup_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Final write
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        "Re-evaluation complete: %d done, %d skipped, %d failed.  "
        "Updated %s",
        n_done, n_skipped, n_failed, summary_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Phase2 YAML config")
    parser.add_argument("--methods", nargs="*", default=None,
                        help="Optional method filter (e.g. --methods M0 M4)")
    parser.add_argument("--sequences", nargs="*", default=None,
                        help="Optional sequence filter")
    args = parser.parse_args()

    cfg = load_phase2_config(args.config)
    _setup_logging(cfg.log_level)
    rerun(cfg, args.methods, args.sequences)


if __name__ == "__main__":
    main()
