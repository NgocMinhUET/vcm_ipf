"""VTM encoder wrapper for Phase 2 experiments.

Provides a Python interface to the patched VTM encoder, handling:
    - Command construction with proper parameters
    - External QP map directory injection
    - Encoder log parsing for bitrate and encoding time
    - Error detection and reporting

Bitrate extraction strategy (in priority order):
    1. Parse VTM SUMMARY block (reliable when present).
    2. Parse "Bytes written to file:" line from log.
    3. Compute from bitstream file size on disk (always available).
This triple-fallback guarantees a non-zero bitrate for any VTM version.
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
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

        # Build a cleaned copy of the cfg once (remove options removed in VTM-23.4).
        self._clean_cfg_path = self._make_clean_cfg(self.encoder_cfg)

    # Options that existed in older VTM versions but were removed in VTM-23.x.
    _DEPRECATED_OPTIONS = {
        "NumWppThreads",
        "NumWppExtraLines",
        "WppBitEqual",
        "EntropyCodingSyncEnabled",
    }

    @staticmethod
    def _make_clean_cfg(src: Path) -> Path:
        """Return path to a temporary cfg with deprecated options stripped out."""
        text = src.read_text(encoding="utf-8", errors="replace")
        clean_lines = []
        for line in text.splitlines():
            stripped = line.lstrip()
            skip = any(
                stripped.startswith(opt + " ") or stripped.startswith(opt + "\t") or stripped == opt
                for opt in VTMEncoder._DEPRECATED_OPTIONS
            )
            if not skip:
                clean_lines.append(line)
        cleaned = "\n".join(clean_lines) + "\n"

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix="_vtm_clean.cfg", delete=False, encoding="utf-8"
        )
        tmp.write(cleaned)
        tmp.close()
        logger.debug("Clean VTM cfg written to %s", tmp.name)
        return Path(tmp.name)

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
        timeout_s: Optional[int] = None,
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
            # Use the pre-cleaned cfg (deprecated options already stripped).
            "-c", str(self._clean_cfg_path),
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

        # Adaptive timeout: 90 s/frame budget (min 3600 s).
        # QP=22 at 1920x1152 takes ~36 s/frame on a modern CPU, so 90 s gives
        # a 2.5× margin.  Caller can override via timeout_s.
        effective_timeout = timeout_s if timeout_s is not None else max(3600, n_frames * 90)

        logger.info(
            "VTM encode: QP=%d, %dx%d, %d frames, timeout=%ds",
            qp, width, height, n_frames, effective_timeout,
        )
        logger.debug("Command: %s", " ".join(cmd))

        t_start = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
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

            # --- Bitrate fallback: use bitstream file size if log parsing failed ---
            # This is always reliable regardless of VTM version / log format changes.
            if metrics["bitrate_kbps"] == 0.0 and bs_path.exists():
                total_bytes = bs_path.stat().st_size
                if total_bytes > 0:
                    total_bits_file = total_bytes * 8
                    bitrate_file = total_bits_file / (n_frames / fps) / 1000  # kbps
                    logger.info(
                        "Log parsing returned bitrate=0; computed from file size: %.1f kbps",
                        bitrate_file,
                    )
                    metrics["bitrate_kbps"] = bitrate_file
                    metrics["total_bits"] = total_bits_file

            return EncodeResult(
                success=True,
                bitstream_path=str(bs_path),
                recon_path=str(recon_path),
                log_path=log_path or "",
                n_frames=n_frames,
                **metrics,
            )

        except subprocess.TimeoutExpired:
            elapsed = time.time() - t_start
            logger.error("VTM encode timed out after %.0fs (limit=%ds)", elapsed, effective_timeout)
            return EncodeResult(
                success=False,
                bitstream_path=str(bs_path),
                recon_path=str(recon_path),
                log_path=log_path or "",
                bitrate_kbps=0, total_bits=0, n_frames=n_frames,
                encoding_time_s=elapsed,
                psnr_y=0, psnr_u=0, psnr_v=0,
                error_msg=f"Encoding timed out after {effective_timeout}s",
            )

    def _parse_encoder_log(self, log: str, n_frames: int, fps: int) -> dict:
        """Parse VTM encoder summary log for bitrate and PSNR.

        VTM-23.4 prints a SUMMARY block at the end of encoding:

            SUMMARY --------------------------------------------------------
            Total Frames |   Bitrate     Y-PSNR    U-PSNR    V-PSNR    YUV-PSNR
                     200 a    1234.5678   35.1234   40.5678   42.1234   36.5678

        The frame-type column ([a-zA-Z]) may be lowercase or uppercase depending
        on the VTM version and configuration.  We use two regex tiers (strict
        then loose) so we never silently return zeros from a format change.
        """
        bitrate = 0.0
        total_bits = 0
        psnr_y = 0.0
        psnr_u = 0.0
        psnr_v = 0.0

        # --- Tier 1: SUMMARY block parser ---
        # Matches lines like:   200 a   1234.5678   35.1234   40.5678   42.1234   36.5678
        # The frame-type column is a single letter (lower OR upper).
        summary_data_re = re.compile(
            r"(\d+)\s+[a-zA-Z]\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
        )

        lines = log.split("\n")
        in_summary = False
        for line in lines:
            if "SUMMARY" in line:
                in_summary = True
                continue
            if in_summary and line.strip():
                m = summary_data_re.search(line)
                if m:
                    bitrate = float(m.group(2))
                    psnr_y = float(m.group(3))
                    psnr_u = float(m.group(4))
                    psnr_v = float(m.group(5))
                    total_bits = int(bitrate * 1000 * n_frames / fps)
                    logger.debug("Parsed SUMMARY: bitrate=%.3f kbps  PSNR-Y=%.4f dB", bitrate, psnr_y)
                    break
                # Stop after the first non-empty line that doesn't match
                # (header row "Total Frames | …" is non-numeric, skip it)
                if not re.search(r"Total\s+Frames|Bitrate|---", line):
                    in_summary = False  # abandoned; reset and keep scanning

        # --- Tier 2: "Bytes written to file" line ---
        if bitrate == 0.0:
            bytes_re = re.compile(r"[Bb]ytes\s+written\s+to\s+file[:\s]+(\d+)")
            for line in lines:
                m = bytes_re.search(line)
                if m:
                    total_bytes = int(m.group(1))
                    total_bits = total_bytes * 8
                    bitrate = total_bits / (n_frames / fps) / 1000
                    logger.info("Bitrate from 'Bytes written' line: %.1f kbps", bitrate)
                    break

        # --- Tier 3: loose scan for any PSNR-Y line as last resort ---
        if psnr_y == 0.0:
            # VTM sometimes prints per-frame PSNR; take the last occurrence
            psnr_re = re.compile(r"PSNR\s+Y\s*[:=]\s*([\d.]+)", re.IGNORECASE)
            for line in reversed(lines):
                m = psnr_re.search(line)
                if m:
                    psnr_y = float(m.group(1))
                    logger.info("PSNR-Y from loose scan: %.4f dB", psnr_y)
                    break

        return {
            "bitrate_kbps": bitrate,
            "total_bits": total_bits,
            "psnr_y": psnr_y,
            "psnr_u": psnr_u,
            "psnr_v": psnr_v,
        }
