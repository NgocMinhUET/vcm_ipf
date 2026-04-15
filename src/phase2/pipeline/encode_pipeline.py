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
from typing import Dict, List, Optional

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
        if cfg.evaluation.compute_task_accuracy:
            self.task_evaluator = TaskEvaluator(
                model_name=cfg.evaluation.detector_model,
                confidence=cfg.evaluation.detector_confidence,
                device=cfg.evaluation.detector_device,
            )

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

        # --- Step 4: PSNR ---
        if self.cfg.evaluation.compute_psnr and enc_result.success:
            logger.info("  Computing PSNR...")
            psnr_result = compute_sequence_psnr(
                original_yuv=seq_cfg.yuv_path,
                reconstructed_yuv=str(recon_path),
                width=seq_cfg.width,
                height=seq_cfg.height,
                n_frames=seq_cfg.n_frames,
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
        if self.task_evaluator and enc_result.success:
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
        logger.info(
            "  Done: bitrate=%.1f kbps, PSNR_Y=%.2f dB, time=%.1fs",
            enc_result.bitrate_kbps,
            enc_result.psnr_y,
            result.total_time_s,
        )

        return result

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
            lines.append(
                f"{r['run_id']:<35} "
                f"{r.get('bitrate_kbps', 0):>10.1f} "
                f"{r.get('psnr_y_enc', 0):>8.2f} "
                f"{r.get('psnr_y_roi', 0):>9.2f} "
                f"{r.get('mAP50', 0):>7.3f} "
                f"{r.get('total_time_s', 0):>8.1f}"
            )
        lines.append("=" * 100)

        path = self.output_dir / "experiment_table.txt"
        path.write_text("\n".join(lines), encoding="utf-8")
