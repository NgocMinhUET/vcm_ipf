"""VTM decoder wrapper for Phase 2 experiments.

Decodes VVC bitstreams to reconstructed YUV using the standard
(unmodified) VTM decoder, verifying bitstream compliance.
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("phase2.encoding.vtm_decoder")


@dataclass
class DecodeResult:
    """Result of a VTM decoding run."""
    success: bool
    recon_path: str
    log_path: str
    decoding_time_s: float
    n_frames_decoded: int
    error_msg: str = ""


class VTMDecoder:
    """Wrapper for the standard VTM DecoderApp."""

    def __init__(self, decoder_path: str):
        self.decoder_path = Path(decoder_path).expanduser()
        if not self.decoder_path.exists():
            raise FileNotFoundError(f"VTM decoder not found: {self.decoder_path}")

    def decode(
        self,
        bitstream: str,
        output_recon: str,
        log_path: str = "",
    ) -> DecodeResult:
        """Decode a VVC bitstream to reconstructed YUV.

        Args:
            bitstream: Path to VVC bitstream (.bin).
            output_recon: Path for decoded YUV output.
            log_path: Optional path to save decoder log.

        Returns:
            DecodeResult with success status and timing.
        """
        bs_path = Path(bitstream).expanduser()
        recon_path = Path(output_recon).expanduser()
        recon_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(self.decoder_path),
            "-b", str(bs_path),
            "-o", str(recon_path),
        ]

        logger.info("VTM decode: %s", bs_path.name)

        t_start = time.time()
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600,
            )
            dec_time = time.time() - t_start
            full_log = result.stdout + "\n" + result.stderr

            if log_path:
                lp = Path(log_path).expanduser()
                lp.parent.mkdir(parents=True, exist_ok=True)
                lp.write_text(full_log, encoding="utf-8")

            n_decoded = 0
            for line in full_log.split("\n"):
                if "POC" in line:
                    n_decoded += 1

            if result.returncode != 0:
                return DecodeResult(
                    success=False,
                    recon_path=str(recon_path),
                    log_path=log_path,
                    decoding_time_s=dec_time,
                    n_frames_decoded=n_decoded,
                    error_msg=result.stderr[-500:] if result.stderr else "Unknown",
                )

            return DecodeResult(
                success=True,
                recon_path=str(recon_path),
                log_path=log_path,
                decoding_time_s=dec_time,
                n_frames_decoded=n_decoded,
            )

        except subprocess.TimeoutExpired:
            return DecodeResult(
                success=False,
                recon_path=str(recon_path),
                log_path=log_path,
                decoding_time_s=3600,
                n_frames_decoded=0,
                error_msg="Decoding timed out",
            )
