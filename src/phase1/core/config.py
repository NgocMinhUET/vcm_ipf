"""Configuration management for the IPF pipeline.

All parameters are gathered into a single Pydantic model loaded from YAML.
This ensures type validation, default values, and easy serialization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

class DetectorConfig(BaseModel):
    model_name: str = Field("yolov8n.pt", description="Ultralytics model identifier")
    confidence_threshold: float = Field(0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0)
    device: str = Field("cuda:0", description="'cuda:0', 'cpu', or 'auto'")
    img_size: int = Field(640, description="Inference resolution")
    classes: Optional[List[int]] = Field(
        None, description="COCO class IDs to keep; None = all"
    )


class TrackerConfig(BaseModel):
    tracker_type: str = Field("bytetrack", description="bytetrack | botsort")
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    track_buffer: int = 30
    match_thresh: float = 0.8


class FieldConfig(BaseModel):
    beta: float = Field(2.0, description="Distance decay exponent")
    eps_d: float = Field(1e-3, description="Epsilon for distance denominator")
    eps_k: float = Field(
        1.0,
        description="Kernel regularization parameter. Controls the core plateau "
        "radius of the Cauchy/Lorentzian kernel phi = m/(d^beta + eps_k). "
        "eps_k=1.0 gives HWHM at d_norm=1 (one object-width). "
        "WARNING: eps_k << 1 creates delta-function spikes that destroy "
        "field quality after normalization.",
    )
    alpha_w: float = Field(0.75, description="Width scaling for normalized distance")
    alpha_h: float = Field(0.75, description="Height scaling for normalized distance")
    superposition: str = Field(
        "sum",
        description="sum | max | lp. "
        "'sum' and 'max' are legacy shortcuts. "
        "'lp' uses the L_p-norm aggregator controlled by `p_norm` — the "
        "Phase 3 academic generalization "
        "(see 11_PHASE3_RESEARCH_PROTOCOL.md §2 Level 4).",
    )
    p_norm: float = Field(
        float("inf"),
        description="L_p-norm exponent when superposition='lp'. "
        "p=1 → sum, p=2 → Euclidean blend, p=4 → near-max, p=inf → hard max. "
        "Unused when superposition ∈ {sum, max}.",
    )


class MassConfig(BaseModel):
    use_class_priority: bool = True
    use_track_age: bool = True
    age_warmup_frames: int = Field(5, ge=1)
    class_priorities: Dict[str, float] = Field(
        default_factory=lambda: {
            "person": 1.0,
            "car": 1.0,
            "truck": 0.9,
            "bus": 0.9,
            "motorcycle": 0.8,
            "bicycle": 0.8,
        }
    )


class NormalizationConfig(BaseModel):
    rho: float = Field(0.95, ge=0.0, le=1.0, description="EMA smoothing factor")
    percentile_low: float = Field(5.0, description="Lower percentile for a_t")
    percentile_high: float = Field(95.0, description="Upper percentile for b_t")
    eps_n: float = Field(1e-8, description="Epsilon for normalization denominator")


class QPMappingConfig(BaseModel):
    qp_base: int = Field(
        32,
        ge=0,
        le=63,
        description="Calibration base QP. Used only by the LEGACY absolute-QP "
        "exporter. Phase 3 delta-QP pipeline is Q_base-agnostic.",
    )
    delta_roi: float = Field(10.0, ge=0.0, description="Max QP decrease for ROI")
    delta_bg: float = Field(6.0, ge=0.0, description="Max QP increase for background")
    gamma_roi: float = Field(1.0, gt=0.0, description="ROI mapping curvature")
    gamma_bg: float = Field(1.0, gt=0.0, description="Background mapping curvature")
    mu: float = Field(0.3, ge=0.0, le=1.0, description="Foreground/background threshold")
    # Phase 3 export clamps (see 11_PHASE3_RESEARCH_PROTOCOL.md §1.4).
    delta_clip_min: int = Field(
        -8, le=0, description="Lower clamp on exported dQP (Phase 3)"
    )
    delta_clip_max: int = Field(
        4, ge=0, description="Upper clamp on exported dQP (Phase 3)"
    )


class BoundedDynamicsConfig(BaseModel):
    eta: float = Field(0.7, ge=0.0, le=1.0, description="Temporal low-pass factor")
    delta_slew: float = Field(3.0, ge=0.0, description="Max QP change per frame")
    qp_min: int = Field(10, ge=0, le=63)
    qp_max: int = Field(51, ge=0, le=63)


class BaselineConfig(BaseModel):
    """Tunable parameters for baseline QP methods.

    Defaults are chosen to produce comparable ROI coverage (~25-35% of CTUs)
    to IPF, ensuring fair comparison at similar operating points (~QP 32).
    """
    m5_sigma_factor: float = Field(
        0.75, gt=0.0,
        description="Gaussian sigma = factor * object_size (controls spread)")
    m6_alpha: float = Field(
        0.8, gt=0.0,
        description="Exponential decay rate (lower = wider spread)")
    m7_cutoff_factor: float = Field(
        0.5, gt=0.0,
        description="Distance cutoff in avg-bbox-diagonal units "
        "(0.5 ≈ 1 CTU transition zone for typical pedestrians)")
    m8_blur_factor: float = Field(
        0.4, gt=0.0,
        description="Blur sigma in avg-bbox-diagonal units "
        "(0.4 ≈ ~1 CTU Gaussian spread for typical pedestrians)")


class AblationConfig(BaseModel):
    """Configuration for ablation study.

    Each ablation variant removes or replaces ONE component of the full
    IPF pipeline (M4) to isolate its contribution.

    Variant definitions:
        A1: No EMA normalization (per-frame percentile normalize instead)
        A3: Single strongest object only (no multi-object superposition)
        A4: Max superposition instead of sum
        A6: Gaussian kernel replacement (same mass/EMA/bounded, different kernel)
        M2: Binary ROI + EMA temporal smoothing (tracking-only, no IPF field)
        M3: IPF spatial field only (no EMA, no bounded dynamics)
    """
    enabled: bool = Field(False, description="Include ablation variants in comparison")
    variants: List[str] = Field(
        default_factory=lambda: ["A1", "A3", "A4", "A6", "M2", "M3"],
        description="Ablation variant IDs to include when enabled",
    )


class CTUConfig(BaseModel):
    ctu_size: int = Field(128, description="CTU size in pixels (64 or 128)")


class VizConfig(BaseModel):
    save_field_maps: bool = True
    save_qp_overlays: bool = True
    save_object_overlays: bool = True
    colormap: str = "jet"
    overlay_alpha: float = 0.4
    dpi: int = 100
    save_every_n_frames: int = Field(1, ge=1)


class OutputConfig(BaseModel):
    save_object_states: bool = True
    save_field_npy: bool = False
    save_qp_csv: bool = True
    save_qp_vtm: bool = Field(
        True,
        description="Write legacy absolute-QP maps to qp_vtm/. Kept for "
        "backward compatibility with pilot v1.",
    )
    save_qp_delta_vtm: bool = Field(
        True,
        description="Write Phase 3 delta-QP maps to qp_vtm_delta/. These are "
        "Q_base-agnostic and are composed with the run-time Q_base by the "
        "Phase 2 encoder wrapper.",
    )
    save_summary_json: bool = True
    save_frame_summaries: bool = True


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

class IPFConfig(BaseModel):
    """Root configuration for the IPF Phase 1 pipeline."""

    run_id: str = Field("run_001", description="Unique run identifier")
    video_path: str = Field("", description="Path to input video")
    output_dir: str = Field("outputs/", description="Root output directory")

    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    field: FieldConfig = Field(default_factory=FieldConfig)
    mass: MassConfig = Field(default_factory=MassConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    qp_mapping: QPMappingConfig = Field(default_factory=QPMappingConfig)
    bounded_dynamics: BoundedDynamicsConfig = Field(default_factory=BoundedDynamicsConfig)
    baselines: BaselineConfig = Field(default_factory=BaselineConfig)
    ablation: AblationConfig = Field(default_factory=AblationConfig)
    ctu: CTUConfig = Field(default_factory=CTUConfig)
    viz: VizConfig = Field(default_factory=VizConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    max_frames: Optional[int] = Field(
        None, description="Process only first N frames (None = all)"
    )
    log_level: str = Field("INFO", description="Logging level")


def load_config(path: Union[str, Path]) -> IPFConfig:
    """Load IPFConfig from a YAML file, falling back to defaults for missing keys."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return IPFConfig(**raw)


def save_config(cfg: IPFConfig, path: Union[str, Path]) -> None:
    """Serialize config to YAML for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg.model_dump(), f, default_flow_style=False, sort_keys=False)
