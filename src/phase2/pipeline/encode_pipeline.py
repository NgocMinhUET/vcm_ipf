"""Phase 2 encoding pipeline orchestrator.

Manages the full encode-decode-evaluate workflow:
    1. Locate Phase 1 QP maps for each method
    2. Encode YUV with VTM using external QP maps
    3. Decode bitstream (compliance verification)
    4. Compute PSNR (full + ROI)
    5. Extract decoded frames and run task evaluation
    6. Aggregate results per (sequence, method, QP) triple

This is the central module that connects all Phase 2 components.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from phase2.core.config import Phase2Config, SequenceConfig
from phase2.encoding.vtm_encoder import VTMEncoder, EncodeResult
from phase2.encoding.vtm_decoder import VTMDecoder, DecodeResult
from phase2.encoding.yuv_utils import save_frames_from_yuv
from phase2.evaluation.psnr import compute_sequence_psnr, SequencePSNR
from phase2.evaluation.task_accuracy import TaskEvaluator, TaskMetrics, save_detections

logger = logging.getLogger("phase2.pipeline")


@dataclass
class RunResult:
    """Complete result for one (sequence, method, QP) encoding run."""
    sequence: str
    method: str
    qp_base: int
    run_id: str
    encode: Optional[dict] = None
    decode: Optional[dict] = None
    psnr: Optional[dict] = None
    task: Optional[dict] = None
    total_time_s: float = 0.0


class EncodingPipeline:
    """Orchestrates the full Phase 2 encode-decode-evaluate pipeline."""

    def __init__(self, cfg: Phase2Config):
        self.cfg = cfg
        self.output_dir = Path(cfg.output_dir).expanduser() / cfg.experiment_id

        self.encoder = VTMEncoder(
            encoder_path=cfg.vtm.encoder_path,
            encoder_cfg=cfg.vtm.encoder_cfg,
            internal_bit_depth=cfg.vtm.internal_bit_depth,
            threads=cfg.vtm.threads,
        )
        self.decoder = VTMDecoder(decoder_path=cfg.vtm.decoder_path)
        self.task_evaluator = None
        if cfg.evaluation.compute_task_accuracy or cfg.evaluation.compute_psnr:
            # YOLO is needed for both task-accuracy (mAP) and ROI PSNR (boxes).
            self.task_evaluator = TaskEvaluator(
                model_name=cfg.evaluation.detector_model,
                confidence=cfg.evaluation.detector_confidence,
                device=cfg.evaluation.detector_device,
            )

        # Per-sequence cache of reference detections (run YOLO on original
        # frames exactly once per sequence — used for BOTH ROI PSNR boxes and
        # mAP pseudo-GT).  Persisted to disk so re-runs skip detection.
        self._ref_boxes_cache: Dict[str, List[List[tuple]]] = {}

    def run_all(self) -> List[RunResult]:
        """Run the full experiment matrix.

        Iterates over all (sequence, method, QP) combinations defined
        in the configuration and produces results for each.
        """
        all_results = []
        total_runs = (
            len(self.cfg.sequences)
            * len(self.cfg.encoding.methods)
            * len(self.cfg.encoding.qp_points)
        )

        logger.info("=" * 70)
        logger.info("Phase 2 Encoding Pipeline — %s", self.cfg.experiment_id)
        logger.info("Sequences: %d, Methods: %d, QP points: %d, Total: %d runs",
                     len(self.cfg.sequences), len(self.cfg.encoding.methods),
                     len(self.cfg.encoding.qp_points), total_runs)
        logger.info("=" * 70)

        run_idx = 0
        for seq_cfg in self.cfg.sequences:
            for method in self.cfg.encoding.methods:
                for qp in self.cfg.encoding.qp_points:
                    run_idx += 1
                    logger.info(
                        "[%d/%d] %s / %s / QP=%d",
                        run_idx, total_runs, seq_cfg.name, method, qp
                    )

                    result = self.run_single(seq_cfg, method, qp)
                    all_results.append(result)

                    self._save_run_result(result)

        self._save_experiment_summary(all_results)

        logger.info("=" * 70)
        logger.info("All %d runs complete. Results: %s", total_runs, self.output_dir)
        logger.info("=" * 70)

        return all_results

    def run_single(
        self,
        seq_cfg: SequenceConfig,
        method: str,
        qp_base: int,
    ) -> RunResult:
        """Run a single (sequence, method, QP) encoding experiment."""
        t_start = time.time()

        run_id = f"{seq_cfg.name}_{method}_QP{qp_base}"
        run_dir = self.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        result = RunResult(
            sequence=seq_cfg.name,
            method=method,
            qp_base=qp_base,
            run_id=run_id,
        )

        # --- Step 1: Locate QP maps ---
        qp_map_dir = self._find_qp_maps(seq_cfg.name, method)

        # --- Step 2: Encode ---
        bs_path = run_dir / "bitstream.bin"
        recon_path = run_dir / "recon.yuv"
        enc_log = run_dir / "encoder.log"

        enc_result = self.encoder.encode(
            input_yuv=seq_cfg.yuv_path,
            output_bitstream=str(bs_path),
            output_recon=str(recon_path),
            width=seq_cfg.width,
            height=seq_cfg.height,
            qp=qp_base,
            n_frames=seq_cfg.n_frames,
            fps=seq_cfg.fps,
            external_qp_dir=qp_map_dir,
            log_path=str(enc_log),
            timeout_s=self.cfg.vtm.timeout_s,
        )
        result.encode = asdict(enc_result)

        if not enc_result.success:
            logger.error("Encoding failed for %s: %s", run_id, enc_result.error_msg)
            result.total_time_s = time.time() - t_start
            return result

        # --- Step 3: Decode (compliance verification) ---
        dec_recon = run_dir / "decoded.yuv"
        dec_log = run_dir / "decoder.log"

        dec_result = self.decoder.decode(
            bitstream=str(bs_path),
            output_recon=str(dec_recon),
            log_path=str(dec_log),
        )
        result.decode = asdict(dec_result)

        if not dec_result.success:
            logger.warning("Decoding failed for %s: %s", run_id, dec_result.error_msg)

        # --- Reference boxes (run YOLO once per sequence, then cache) ---
        # Used for BOTH ROI PSNR (Step 4) and mAP pseudo-GT (Step 5).  This
        # is the single source of truth for "where the objects are".
        ref_boxes = None
        if enc_result.success and seq_cfg.frames_dir:
            ref_boxes = self._get_reference_boxes(seq_cfg)

        # --- Step 4: PSNR (full + ROI + BG) ---
        if self.cfg.evaluation.compute_psnr and enc_result.success:
            logger.info("  Computing PSNR (with %d reference boxes-per-frame)...",
                        len(ref_boxes) if ref_boxes else 0)
            # PSNR is computed on the FULL CTU-padded frame (e.g. 1920x1152)
            # because that's what VTM reconstructs.  Boxes were detected on
            # the un-padded original (1920x1080) and remain valid since
            # padding is added below the visible region.
            psnr_result = compute_sequence_psnr(
                original_yuv=seq_cfg.yuv_path,
                reconstructed_yuv=str(recon_path),
                width=seq_cfg.width,
                height=seq_cfg.height,
                n_frames=seq_cfg.n_frames,
                boxes_per_frame=ref_boxes,
                expansion=self.cfg.evaluation.roi_expansion,
            )
            result.psnr = {
                "n_frames": psnr_result.n_frames,
                "psnr_y_full_mean": psnr_result.psnr_y_full_mean,
                "psnr_y_roi_mean": psnr_result.psnr_y_roi_mean,
                "psnr_y_bg_mean": psnr_result.psnr_y_bg_mean,
                "psnr_y_full_std": psnr_result.psnr_y_full_std,
            }

        # --- Step 5: Task accuracy ---
        if self.cfg.evaluation.compute_task_accuracy and self.task_evaluator and enc_result.success:
            logger.info("  Extracting decoded frames...")
            dec_frames_dir = run_dir / "decoded_frames"
            save_frames_from_yuv(
                str(recon_path), str(dec_frames_dir),
                seq_cfg.width, seq_cfg.height,
                seq_cfg.n_frames,
                seq_cfg.original_width, seq_cfg.original_height,
            )

            logger.info("  Running task evaluation (YOLOv8)...")
            ref_frames_dir = seq_cfg.frames_dir if seq_cfg.frames_dir else None
            task_result = self.task_evaluator.compute_task_metrics(
                decoded_frames_dir=str(dec_frames_dir),
                reference_frames_dir=ref_frames_dir,
                n_frames=seq_cfg.n_frames,
            )
            result.task = asdict(task_result)

        result.total_time_s = time.time() - t_start

        # Prefer PSNR from independent YUV comparison (Step 4); fall back to
        # encoder-log value (may be 0 for some VTM versions).
        psnr_display = (
            result.psnr.get("psnr_y_full_mean", 0.0)
            if result.psnr
            else enc_result.psnr_y
        )
        logger.info(
            "  Done: bitrate=%.1f kbps, PSNR_Y=%.2f dB, time=%.1fs",
            enc_result.bitrate_kbps,
            psnr_display,
            result.total_time_s,
        )

        return result

    def _get_reference_boxes(
        self,
        seq_cfg: SequenceConfig,
    ) -> Optional[List[List[tuple]]]:
        """Return per-frame YOLO boxes for the original (uncompressed) frames.

        Boxes are detected exactly once per sequence and cached:
            1. In-memory (`self._ref_boxes_cache`) for the current process
            2. On-disk at `<output_dir>/_reference_boxes/<seq_name>.json`
               so that re-runs and re-evaluation skip detection.

        Returns:
            List of length n_frames; each element is a list of
            (x1, y1, x2, y2) tuples.  Returns None on any failure.
        """
        seq_name = seq_cfg.name
        if seq_name in self._ref_boxes_cache:
            return self._ref_boxes_cache[seq_name]

        cache_dir = self.output_dir / "_reference_boxes"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{seq_name}.json"

        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                boxes = [[tuple(b) for b in frame_boxes] for frame_boxes in data]
                self._ref_boxes_cache[seq_name] = boxes
                logger.info(
                    "  Loaded %d frames of reference boxes from cache: %s",
                    len(boxes), cache_path,
                )
                return boxes
            except Exception as exc:
                logger.warning("  Failed to load box cache (%s); re-detecting", exc)

        if not self.task_evaluator:
            return None
        if not seq_cfg.frames_dir:
            return None

        logger.info(
            "  Detecting reference boxes on original frames: %s "
            "(one-time, will be cached)", seq_cfg.frames_dir,
        )
        try:
            dets = self.task_evaluator.detect_on_frames(
                frames_dir=seq_cfg.frames_dir,
                n_frames=seq_cfg.n_frames,
            )
        except Exception as exc:
            logger.error("  Reference detection failed: %s", exc)
            return None

        boxes = [list(d.boxes) for d in dets]
        self._ref_boxes_cache[seq_name] = boxes

        try:
            cache_path.write_text(
                json.dumps([[list(b) for b in fb] for fb in boxes], indent=2),
                encoding="utf-8",
            )
            logger.info("  Cached %d frames of boxes to %s", len(boxes), cache_path)
        except Exception as exc:
            logger.warning("  Failed to persist box cache: %s", exc)

        total_dets = sum(len(fb) for fb in boxes)
        logger.info(
            "  Reference detection complete: %d frames, %d total detections "
            "(avg %.1f per frame)",
            len(boxes), total_dets, total_dets / max(len(boxes), 1),
        )
        return boxes

    def _find_qp_maps(self, seq_name: str, method: str) -> Optional[str]:
        """Locate Phase 1 QP maps for a given sequence and method.

        For M0 (uniform QP), no external QP maps are needed.
        For other methods, find the qp_vtm/ directory from Phase 1 outputs.
        """
        if method == "M0":
            return None

        phase1_dir = Path(self.cfg.encoding.phase1_output_dir).expanduser()
        prefix = self.cfg.encoding.phase1_run_prefix

        run_dir = phase1_dir / f"{prefix}{seq_name}" / method / "qp_vtm"
        if run_dir.exists():
            return str(run_dir)

        run_dir_alt = phase1_dir / f"{prefix}{seq_name}" / method / "qp_vtm"
        if run_dir_alt.exists():
            return str(run_dir_alt)

        logger.warning(
            "QP maps not found for %s/%s at %s", seq_name, method, run_dir
        )
        return None

    def _save_run_result(self, result: RunResult) -> None:
        """Save individual run result as JSON."""
        run_dir = self.output_dir / result.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "result.json"
        path.write_text(
            json.dumps(asdict(result), indent=2, default=str),
            encoding="utf-8",
        )

    def _save_experiment_summary(self, results: List[RunResult]) -> None:
        """Save experiment-level summary with all run results."""
        summary = {
            "experiment_id": self.cfg.experiment_id,
            "n_runs": len(results),
            "n_successful": sum(1 for r in results if r.encode and r.encode.get("success")),
            "sequences": [s.name for s in self.cfg.sequences],
            "methods": self.cfg.encoding.methods,
            "qp_points": self.cfg.encoding.qp_points,
            "results": [],
        }

        for r in results:
            entry = {
                "run_id": r.run_id,
                "sequence": r.sequence,
                "method": r.method,
                "qp_base": r.qp_base,
                "total_time_s": r.total_time_s,
            }
            if r.encode:
                entry["bitrate_kbps"] = r.encode.get("bitrate_kbps", 0)
                entry["psnr_y_enc"] = r.encode.get("psnr_y", 0)
                entry["success"] = r.encode.get("success", False)
            if r.psnr:
                entry["psnr_y_full"] = r.psnr.get("psnr_y_full_mean", 0)
                entry["psnr_y_roi"] = r.psnr.get("psnr_y_roi_mean", 0)
            if r.task:
                entry["mAP50"] = r.task.get("mAP50", 0)
                entry["mAP50_95"] = r.task.get("mAP50_95", 0)
            summary["results"].append(entry)

        path = self.output_dir / "experiment_summary.json"
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        self._write_summary_table(summary)

    def _write_summary_table(self, summary: dict) -> None:
        """Write human-readable summary table."""
        lines = []
        lines.append("=" * 100)
        lines.append(f"EXPERIMENT SUMMARY — {summary['experiment_id']}")
        lines.append(f"Runs: {summary['n_successful']}/{summary['n_runs']} successful")
        lines.append("=" * 100)
        lines.append(
            f"{'Run ID':<35} {'Rate(kbps)':>10} {'PSNR-Y':>8} "
            f"{'PSNR-ROI':>9} {'mAP50':>7} {'Time(s)':>8}"
        )
        lines.append("-" * 100)

        for r in summary["results"]:
            # Prefer independently-computed PSNR (from YUV comparison) over
            # encoder-log PSNR which may be 0 on some VTM versions.
            psnr_display = r.get("psnr_y_full") or r.get("psnr_y_enc", 0)
            lines.append(
                f"{r['run_id']:<35} "
                f"{r.get('bitrate_kbps', 0):>10.1f} "
                f"{psnr_display:>8.2f} "
                f"{r.get('psnr_y_roi', 0):>9.2f} "
                f"{r.get('mAP50', 0):>7.3f} "
                f"{r.get('total_time_s', 0):>8.1f}"
            )
        lines.append("=" * 100)

        path = self.output_dir / "experiment_table.txt"
        path.write_text("\n".join(lines), encoding="utf-8")
