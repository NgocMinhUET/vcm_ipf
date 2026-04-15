"""VTM encoder wrapper for Phase 2 experiments.

Provides a Python interface to the patched VTM encoder, handling:
    - Command construction with proper parameters
    - External QP map directory injection
    - Encoder log parsing for bitrate and encoding time
    - Error detection and reporting
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("phase2.encoding.vtm_encoder")


@dataclass
class EncodeResult:
    """Result of a VTM encoding run."""
    success: bool
    bitstream_path: str
    recon_path: str
    log_path: str
    bitrate_kbps: float
    total_bits: int
    n_frames: int
    encoding_time_s: float
    psnr_y: float
    psnr_u: float
    psnr_v: float
    error_msg: str = ""


class VTMEncoder:
    """Wrapper for the patched VTM EncoderApp."""

    def __init__(
        self,
        encoder_path: str,
        encoder_cfg: str,
        internal_bit_depth: int = 8,
        threads: int = 1,
    ):
        self.encoder_path = Path(encoder_path).expanduser()
        self.encoder_cfg = Path(encoder_cfg).expanduser()
        self.internal_bit_depth = internal_bit_depth
        self.threads = threads

        if not self.encoder_path.exists():
            raise FileNotFoundError(f"VTM encoder not found: {self.encoder_path}")
        if not self.encoder_cfg.exists():
            raise FileNotFoundError(f"VTM config not found: {self.encoder_cfg}")

    def encode(
        self,
        input_yuv: str,
        output_bitstream: str,
        output_recon: str,
        width: int,
        height: int,
        qp: int,
        n_frames: int,
        fps: int = 30,
        external_qp_dir: Optional[str] = None,
        log_path: Optional[str] = None,
    ) -> EncodeResult:
        """Encode a YUV sequence with VTM.

        Args:
            input_yuv: Path to raw YUV 4:2:0 input.
            output_bitstream: Path for output VVC bitstream (.bin).
            output_recon: Path for reconstructed YUV output.
            width: Frame width (CTU-aligned).
            height: Frame height (CTU-aligned).
            qp: Base QP value (overridden per-CTU if external_qp_dir is set).
            n_frames: Number of frames to encode.
            fps: Frames per second.
            external_qp_dir: Directory with per-CTU QP maps (None = uniform QP).
            log_path: Path to save encoder log.

        Returns:
            EncodeResult with metrics parsed from encoder log.
        """
        input_path = Path(input_yuv).expanduser()
        bs_path = Path(output_bitstream).expanduser()
        recon_path = Path(output_recon).expanduser()

        bs_path.parent.mkdir(parents=True, exist_ok=True)
        recon_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(self.encoder_path),
            "-c", str(self.encoder_cfg),
            # Treat unknown cfg file options as warnings, not errors.
            # Needed because some packaged .cfg files contain options that were
            # removed between VTM versions (e.g. NumWppThreads, NumWppExtraLines).
            "-w",
            "-i", str(input_path),
            "-b", str(bs_path),
            "-o", str(recon_path),
            "-wdt", str(width),
            "-hgt", str(height),
            "-q", str(qp),
            "-f", str(n_frames),
            "-fr", str(fps),
            f"--InternalBitDepth={self.internal_bit_depth}",
        ]

        if external_qp_dir:
            qp_dir = Path(external_qp_dir).expanduser()
            cmd.append(f"--ExternalQPMapDir={qp_dir}")

        logger.info("VTM encode: QP=%d, %dx%d, %d frames", qp, width, height, n_frames)
        logger.debug("Command: %s", " ".join(cmd))

        t_start = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,
            )
            encoding_time = time.time() - t_start

            full_log = result.stdout + "\n" + result.stderr

            if log_path:
                lp = Path(log_path).expanduser()
                lp.parent.mkdir(parents=True, exist_ok=True)
                lp.write_text(full_log, encoding="utf-8")

            if result.returncode != 0:
                logger.error("VTM encoder failed (exit code %d)", result.returncode)
                return EncodeResult(
                    success=False,
                    bitstream_path=str(bs_path),
                    recon_path=str(recon_path),
                    log_path=log_path or "",
                    bitrate_kbps=0,
                    total_bits=0,
                    n_frames=n_frames,
                    encoding_time_s=encoding_time,
                    psnr_y=0, psnr_u=0, psnr_v=0,
                    error_msg=result.stderr[-500:] if result.stderr else "Unknown error",
                )

            metrics = self._parse_encoder_log(full_log, n_frames, fps)
            metrics["encoding_time_s"] = encoding_time

            return EncodeResult(
                success=True,
                bitstream_path=str(bs_path),
                recon_path=str(recon_path),
                log_path=log_path or "",
                n_frames=n_frames,
                **metrics,
            )

        except subprocess.TimeoutExpired:
            return EncodeResult(
                success=False,
                bitstream_path=str(bs_path),
                recon_path=str(recon_path),
                log_path=log_path or "",
                bitrate_kbps=0, total_bits=0, n_frames=n_frames,
                encoding_time_s=7200,
                psnr_y=0, psnr_u=0, psnr_v=0,
                error_msg="Encoding timed out after 7200s",
            )

    def _parse_encoder_log(self, log: str, n_frames: int, fps: int) -> dict:
        """Parse VTM encoder summary log for bitrate and PSNR.

        VTM summary line format:
            SUMMARY --------------------------------------------------------
            Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR
                  100    a    1234.5678   35.1234   40.5678   42.1234   36.5678
        """
        bitrate = 0.0
        total_bits = 0
        psnr_y = 0.0
        psnr_u = 0.0
        psnr_v = 0.0

        summary_pattern = re.compile(
            r"(\d+)\s+[a-z]\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
        )

        lines = log.split("\n")
        in_summary = False
        for line in lines:
            if "SUMMARY" in line:
                in_summary = True
                continue
            if in_summary:
                match = summary_pattern.search(line)
                if match:
                    bitrate = float(match.group(2))
                    psnr_y = float(match.group(3))
                    psnr_u = float(match.group(4))
                    psnr_v = float(match.group(5))
                    total_bits = int(bitrate * 1000 * n_frames / fps)
                    break

        # Fallback: try to get bitrate from bitstream file size
        if bitrate == 0.0:
            bits_pattern = re.compile(r"Bytes written to file:\s*(\d+)")
            for line in lines:
                match = bits_pattern.search(line)
                if match:
                    total_bytes = int(match.group(1))
                    total_bits = total_bytes * 8
                    bitrate = total_bits / (n_frames / fps) / 1000
                    break

        return {
            "bitrate_kbps": bitrate,
            "total_bits": total_bits,
            "psnr_y": psnr_y,
            "psnr_u": psnr_u,
            "psnr_v": psnr_v,
        }
