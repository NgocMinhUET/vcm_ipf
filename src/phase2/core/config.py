"""Phase 2 configuration management.

Defines the encoding experiment configuration including VTM paths,
encoding parameters, QP grid, method list, and evaluation settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict

import yaml
from pydantic import BaseModel, Field


class VTMConfig(BaseModel):
    """VTM encoder/decoder paths and settings."""
    encoder_path: str = Field(
        # Patched static binary built with ExternalQPMapDir support.
        # Located at bin/EncoderAppStatic (static build, NOT the cmake umake output).
        "~/Minh/ipf/vtm/VVCSoftware_VTM/bin/EncoderAppStatic",
        description="Path to patched VTM EncoderApp (must support --ExternalQPMapDir)",
    )
    decoder_path: str = Field(
        "~/Minh/ipf/vtm/VVCSoftware_VTM/bin/DecoderAppStatic",
        description="Path to VTM DecoderApp",
    )
    encoder_cfg: str = Field(
        "~/Minh/ipf/vtm/VVCSoftware_VTM/cfg/encoder_lowdelay_vtm.cfg",
        description="VTM encoder configuration profile",
    )
    internal_bit_depth: int = Field(8, description="Internal bit depth (8 or 10)")
    threads: int = Field(1, description="VTM parallel threads (1 = deterministic)")
    timeout_s: Optional[int] = Field(
        None,
        description=(
            "Per-run encoding timeout in seconds.  None = auto (max(3600, n_frames×90))."
        ),
    )


class SequenceConfig(BaseModel):
    """Configuration for a single test sequence."""
    name: str = Field(..., description="Sequence identifier (e.g. MOT17-04-DPM)")
    yuv_path: str = Field(..., description="Path to raw YUV 4:2:0 file")
    width: int = Field(1920, description="Frame width (CTU-aligned)")
    height: int = Field(1088, description="Frame height (CTU-aligned after padding)")
    original_width: int = Field(1920, description="Original width before padding")
    original_height: int = Field(1080, description="Original height before padding")
    fps: int = Field(30)
    n_frames: int = Field(200, description="Number of frames to encode")
    frames_dir: str = Field("", description="Original frames directory for task eval")


class EncodingConfig(BaseModel):
    """Encoding experiment parameters."""
    qp_points: List[int] = Field(
        default_factory=lambda: [22, 27, 32, 37],
        description="QP base values for RD curve (need >= 4 for BD-Rate)",
    )
    methods: List[str] = Field(
        default_factory=lambda: ["M0", "M1", "M4", "M5", "M6"],
        description="Methods to encode (must have QP maps from Phase 1)",
    )
    phase1_output_dir: str = Field(
        "~/Minh/ipf/phase1_outputs",
        description="Phase 1 output directory containing QP maps",
    )
    phase1_run_prefix: str = Field(
        "multi_seq_",
        description="Phase 1 run ID prefix for finding QP maps",
    )


class EvaluationConfig(BaseModel):
    """Evaluation settings."""
    compute_psnr: bool = Field(True, description="Compute PSNR (full-frame + ROI)")
    compute_ssim: bool = Field(False, description="Compute SSIM (slower)")
    compute_task_accuracy: bool = Field(True, description="Run YOLOv8 on decoded frames")
    detector_model: str = Field("yolov8n.pt", description="YOLOv8 model for task eval")
    detector_confidence: float = Field(
        0.25,
        description="Operating-point confidence (used for legacy P×R + visible n_dets stats)",
    )
    detector_conf_low: float = Field(
        0.001,
        description="Low confidence threshold for COCO PR-curve sweep "
                    "(must be ≤ detector_confidence; standard COCO uses 0.001)",
    )
    detector_classes: List[int] = Field(
        default_factory=lambda: [0],
        description="COCO class IDs to score (default: [0] = person, suits MOT)",
    )
    detector_device: str = Field("cuda:0")
    roi_expansion: float = Field(
        0.1, description="Fractional expansion of GT boxes for ROI PSNR computation"
    )


class Phase2Config(BaseModel):
    """Root configuration for Phase 2 experiments."""
    experiment_id: str = Field("pilot_v1", description="Experiment identifier")
    output_dir: str = Field("~/Minh/ipf/phase2_outputs")

    vtm: VTMConfig = Field(default_factory=VTMConfig)
    sequences: List[SequenceConfig] = Field(default_factory=list)
    encoding: EncodingConfig = Field(default_factory=EncodingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    log_level: str = Field("INFO")


def load_phase2_config(path: str) -> Phase2Config:
    """Load Phase2Config from YAML."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Phase2Config(**raw)


def save_phase2_config(cfg: Phase2Config, path: str) -> None:
    """Save Phase2Config to YAML."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(cfg.model_dump(), f, default_flow_style=False, sort_keys=False)
