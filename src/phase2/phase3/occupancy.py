"""Phase 3 Stage C — **Occupancy-Guided IPF** (OG-IPF).

Why this module exists
----------------------
The classical IPF (Phase 1) treats every CTU as a sample of a smooth
mass-distance potential field
``φ_{j,c} = m_j / (d_{j,c}^β + ε_k)`` and aggregates across objects.
This works well for "where in the frame are objects on average?", but
it is **biased against small or partially-occluded objects** because:

* the mass term ``m_j ∝ √(w_j h_j)`` shrinks with object area;
* the kernel is normalised by object width/height, so small bboxes
  produce small kernels that decay quickly outside the box;
* the resulting Φ inside a CTU that overlaps a small distant pedestrian
  can be lower than Φ in a CTU near (but outside) a large near-camera
  pedestrian — exactly the opposite of what task-driven coding wants.

For machine-vision coding the right object-CTU treatment is:

    "If a CTU overlaps any machine-relevant object, it must receive
     minimum quality protection regardless of mass; CTUs near (but
     outside) objects deserve mild protection; far CTUs deserve none."

This module decouples **object overlap** (a binary-ish geometric fact)
from **contextual influence** (the smooth potential field), and
combines them through a per-CTU utility that drives RD-log A+ in
``og_ipf.py``.

Reference (paper §3 narrative)
------------------------------
We compute, per CTU ``c`` and per object ``j``:

    A^ctu_{j,c} = |R_c ∩ B_j| / |R_c|         ← what fraction of the CTU
                                                is inside the object
    A^obj_{j,c} = |R_c ∩ B_j| / |B_j|         ← what fraction of the
                                                object is in this CTU
                                                (>> A^ctu for SMALL objs)
    G_{j,c}     = clip(λ_ctu·A^ctu + λ_obj·A^obj, 0, 1)

``A^obj`` is the term that protects small objects: a 32×32 person
landing entirely inside one CTU has ``A^obj = 1``, while ``A^ctu`` is
only ``32·32 / 128·128 = 0.0625``. The convex combination
``λ_ctu·A^ctu + λ_obj·A^obj`` therefore gives that CTU a high
occupancy gate. Large objects spanning many CTUs get high ``A^ctu``
and are also protected. Both modes lead to ``G ≈ 1`` *inside* the box.

Then per-object utility:

    P_{j,t} = π_j · τ_{j,t} · c_{j,t}   · (w_j h_j)^γ        ; γ ≥ 0
    D_{j,c} = 1 / (d_{j,c}^β + ε_k)                    ; smooth field
    C_{j,c} = (1 - G_{j,c}) · D_{j,c}                  ; outside-only
    U_{j,c} = P_{j,t} · (α_in · G_{j,c} + α_ctx · C_{j,c})

and aggregation across objects:

    U_c     = max_j U_{j,c}        ; or  ( Σ U^p )^{1/p} for dense scenes
    G_c     = max_j G_{j,c}        ; used for min-object-protection

Defaults are documented in ``OGConfig`` and chosen so the formula
reduces to the legacy IPF kernel in the limit ``λ_obj = α_in = 0,
γ = 0.5``. This keeps the implementation honest: every change versus
classical IPF is opt-in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OGConfig:
    """Hyper-parameters of the Occupancy-Guided IPF.

    Defaults are calibrated against the spec in PROJECT_STATE §7.20 and
    follow the user's research request to keep ``γ = 0`` (no implicit
    size term) and to use ``max`` aggregation for dense MOT17 scenes.

    All fields are immutable so the same config can be reused across
    frames without aliasing surprises.
    """

    # Occupancy gate
    lambda_ctu: float = 0.4
    lambda_obj: float = 0.6
    # Object priority
    gamma:      float = 0.0          # area exponent on (w·h)^γ; 0 = ignore size
    # Distance kernel (matches Phase 1 importance_field defaults)
    alpha_w:    float = 1.5
    alpha_h:    float = 1.5
    eps_d:      float = 1e-3
    beta:       float = 2.0
    eps_k:      float = 1.0
    # Utility blend
    alpha_in:   float = 1.0          # weight on occupancy gate G
    alpha_ctx:  float = 0.3          # weight on (1-G)·D context
    # Aggregation across objects
    aggregator: str   = "max"        # "max" or "lp"
    p_norm:     float = 4.0          # used only when aggregator == "lp"
    # Numerical safety
    eps_u:      float = 1e-9         # floor for U_c (avoids zero division upstream)


@dataclass
class ObjectBox:
    """Minimal per-frame, per-object record for OG-IPF.

    Field names mirror the Phase 1 ``ObjectState`` so the two are
    interchangeable; only the geometry + priority terms are needed
    here.
    """

    x: float                # bbox top-left x (pixels)
    y: float                # bbox top-left y (pixels)
    w: float                # bbox width (pixels)
    h: float                # bbox height (pixels)
    confidence: float = 1.0
    track_age:  float = 1.0
    class_priority: float = 1.0

    @property
    def x_center(self) -> float:
        return self.x + 0.5 * self.w

    @property
    def y_center(self) -> float:
        return self.y + 0.5 * self.h

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float,
                  confidence: float = 1.0, **kw) -> "ObjectBox":
        return cls(x=float(x1), y=float(y1),
                   w=float(max(0.0, x2 - x1)),
                   h=float(max(0.0, y2 - y1)),
                   confidence=float(confidence), **kw)

    @classmethod
    def from_dict(cls, d: dict) -> "ObjectBox":
        # Tolerant loader for both Phase 1 ObjectState and YOLO xyxy dumps.
        if "x" in d and "y" in d and "w" in d and "h" in d:
            return cls(x=float(d["x"]), y=float(d["y"]),
                       w=float(d["w"]), h=float(d["h"]),
                       confidence=float(d.get("confidence", d.get("score", 1.0))),
                       track_age=float(d.get("track_age", 1.0)),
                       class_priority=float(d.get("class_priority", 1.0)))
        if "x_center" in d and "width" in d and "height" in d:
            cx, cy = float(d["x_center"]), float(d["y_center"])
            w, h = float(d["width"]), float(d["height"])
            return cls(x=cx - 0.5 * w, y=cy - 0.5 * h, w=w, h=h,
                       confidence=float(d.get("confidence", 1.0)),
                       track_age=float(d.get("track_age", 1.0)),
                       class_priority=float(d.get("class_priority", 1.0)))
        if "xyxy" in d:
            x1, y1, x2, y2 = (float(v) for v in d["xyxy"])
            return cls.from_xyxy(x1, y1, x2, y2,
                                  confidence=float(d.get("score",
                                                          d.get("confidence", 1.0))))
        raise KeyError(f"Cannot build ObjectBox from dict keys {list(d)}")


# ---------------------------------------------------------------------------
# CTU grid helper (kept identical to phase1.field.importance_field.build_ctu_grid
# so the two pipelines stay aligned without a hidden import dependency).
# ---------------------------------------------------------------------------

def make_ctu_grid(frame_h: int, frame_w: int, ctu_size: int
                   ) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Return ``(grid_x, grid_y, n_rows, n_cols)`` of CTU centre coordinates.

    ``grid_x[r, c]`` is the x-coordinate of the centre of CTU at row ``r``,
    column ``c``. We use the same convention as Phase 1: rows are paginated
    by ``ceil(H / ctu_size)``, so the bottom row may be partially outside
    the frame (that's normal for non-multiples of 128).
    """
    n_cols = int(np.ceil(frame_w / ctu_size))
    n_rows = int(np.ceil(frame_h / ctu_size))
    col_centres = np.arange(n_cols, dtype=np.float64) * ctu_size + ctu_size * 0.5
    row_centres = np.arange(n_rows, dtype=np.float64) * ctu_size + ctu_size * 0.5
    grid_x, grid_y = np.meshgrid(col_centres, row_centres)
    return grid_x, grid_y, n_rows, n_cols


# ---------------------------------------------------------------------------
# Per-object overlap / occupancy gate
# ---------------------------------------------------------------------------

def _ctu_box_corners(n_rows: int, n_cols: int, ctu_size: int,
                      frame_h: int, frame_w: int
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return CTU rectangles ``(x1, y1, x2, y2)`` clipped to the frame.

    The returned arrays each have shape ``(n_rows, n_cols)`` and are the
    pixel coordinates of every CTU's *top-left* and *bottom-right*
    corner. Clipping to ``[0, W-1] × [0, H-1]`` is important so the
    bottom/right CTUs (which extend past the frame for non-multiple
    resolutions) report the correct *visible* area.
    """
    cols = np.arange(n_cols, dtype=np.float64)
    rows = np.arange(n_rows, dtype=np.float64)
    x1 = np.minimum(cols * ctu_size,                   frame_w).astype(np.float64)
    x2 = np.minimum((cols + 1) * ctu_size,             frame_w).astype(np.float64)
    y1 = np.minimum(rows * ctu_size,                   frame_h).astype(np.float64)
    y2 = np.minimum((rows + 1) * ctu_size,             frame_h).astype(np.float64)
    grid_x1 = np.broadcast_to(x1[None, :], (n_rows, n_cols))
    grid_x2 = np.broadcast_to(x2[None, :], (n_rows, n_cols))
    grid_y1 = np.broadcast_to(y1[:, None], (n_rows, n_cols))
    grid_y2 = np.broadcast_to(y2[:, None], (n_rows, n_cols))
    return grid_x1, grid_y1, grid_x2, grid_y2


def _intersection_area(box: ObjectBox,
                        ctu_x1: np.ndarray, ctu_y1: np.ndarray,
                        ctu_x2: np.ndarray, ctu_y2: np.ndarray
                        ) -> np.ndarray:
    bx1 = box.x; by1 = box.y
    bx2 = box.x + box.w; by2 = box.y + box.h
    iw = np.clip(np.minimum(ctu_x2, bx2) - np.maximum(ctu_x1, bx1), 0.0, None)
    ih = np.clip(np.minimum(ctu_y2, by2) - np.maximum(ctu_y1, by1), 0.0, None)
    return iw * ih


def occupancy_gate_per_object(
    box: ObjectBox,
    ctu_x1: np.ndarray, ctu_y1: np.ndarray,
    ctu_x2: np.ndarray, ctu_y2: np.ndarray,
    cfg: OGConfig,
) -> np.ndarray:
    """Compute ``G_{j,c}`` (the per-object occupancy gate) for one box."""
    inter = _intersection_area(box, ctu_x1, ctu_y1, ctu_x2, ctu_y2)
    ctu_area = np.maximum((ctu_x2 - ctu_x1) * (ctu_y2 - ctu_y1), 1e-9)
    obj_area = max(box.w * box.h, 1e-9)
    a_ctu = inter / ctu_area
    a_obj = inter / obj_area
    g = cfg.lambda_ctu * a_ctu + cfg.lambda_obj * a_obj
    return np.clip(g, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Distance kernel (legacy IPF kernel, kept identical so OG-IPF reduces
# cleanly to classical IPF when α_ctx = 1 and α_in = 0).
# ---------------------------------------------------------------------------

def distance_kernel_per_object(
    box: ObjectBox, grid_x: np.ndarray, grid_y: np.ndarray, cfg: OGConfig,
) -> np.ndarray:
    """Phase-1-style normalised distance kernel ``D_{j,c}``."""
    dx = (grid_x - box.x_center) / max(cfg.alpha_w * box.w + cfg.eps_d, 1e-6)
    dy = (grid_y - box.y_center) / max(cfg.alpha_h * box.h + cfg.eps_d, 1e-6)
    d_norm = np.sqrt(dx * dx + dy * dy)
    return 1.0 / (np.power(d_norm, cfg.beta) + cfg.eps_k)


# ---------------------------------------------------------------------------
# Public entry: compute_occupancy_utility
# ---------------------------------------------------------------------------

@dataclass
class OccupancyResult:
    """Per-frame OG-IPF outputs.

    Attributes
    ----------
    U
        Per-CTU utility ``U_c`` after object-aggregation. Shape
        ``(n_rows, n_cols)``, dtype float64. Used as the input to
        RD-log A+ in ``og_ipf.compute_og_a_plus_delta``.
    G_max
        Per-CTU occupancy gate aggregated as ``max_j G_{j,c}``. Used
        for the minimum-object-protection constraint downstream.
    G_per_object
        Per-object occupancy gate stack ``(n_objects, n_rows, n_cols)``
        — kept for diagnostics + ablation; ``None`` if no boxes.
    n_objects
        Number of boxes that contributed (after width/height > 0 filter).
    """

    U:           np.ndarray
    G_max:       np.ndarray
    G_per_object: Optional[np.ndarray]
    n_objects:   int


def _aggregate(stack: np.ndarray, cfg: OGConfig) -> np.ndarray:
    """``max`` or numerically-stable L_p aggregation across axis 0."""
    if stack.size == 0:
        raise ValueError("Cannot aggregate an empty stack")
    if cfg.aggregator == "max":
        return np.max(stack, axis=0)
    if cfg.aggregator == "lp":
        p = float(cfg.p_norm)
        if not np.isfinite(p):
            return np.max(stack, axis=0)
        if p <= 0:
            raise ValueError(f"p_norm must be > 0, got {p}")
        m = np.max(stack, axis=0)
        safe_m = np.where(m > 0.0, m, 1.0)
        normed = stack / safe_m[None, :, :]
        powsum = np.sum(np.power(normed, p), axis=0)
        agg = m * np.power(powsum, 1.0 / p)
        return np.where(m > 0.0, agg, 0.0)
    raise ValueError(f"Unknown aggregator: {cfg.aggregator!r}")


def compute_occupancy_utility(
    boxes: Sequence[ObjectBox],
    frame_h: int,
    frame_w: int,
    ctu_size: int = 128,
    cfg: Optional[OGConfig] = None,
    *,
    visible_h: Optional[int] = None,
    visible_w: Optional[int] = None,
    return_per_object: bool = False,
) -> OccupancyResult:
    """Compute the OG-IPF utility and occupancy maps for one frame.

    Parameters
    ----------
    boxes
        Iterable of :class:`ObjectBox`. Boxes with non-positive area
        are silently dropped (consistent with Phase 1's behaviour).
    frame_h, frame_w
        **Padded** frame size in pixels — used to build the CTU grid
        via :func:`make_ctu_grid`.  For VVC streams whose luma height
        is not a multiple of ``ctu_size``, ``frame_h`` is the padded
        multiple (e.g. 1152 for a 1080-line source).
    visible_h, visible_w
        **Visible** (unpadded) frame dimensions in pixels.  When
        supplied, CTU box corners are clipped to these bounds instead
        of the padded ``frame_h / frame_w`` bounds.  This prevents
        the bottom / right boundary CTUs from being assigned
        artificially large areas, which would under-weight any bboxes
        near the visible edge.  If ``None`` (default), falls back to
        ``frame_h / frame_w`` for backward compatibility.
    ctu_size
        Side length of one CTU in pixels (128 in our pipeline).
    cfg
        :class:`OGConfig`; defaults are calibrated to the user spec.
    return_per_object
        If ``True``, the per-object gate stack is retained and returned
        in :attr:`OccupancyResult.G_per_object`. Default: ``False`` so
        we don't blow memory on long sequences.

    Returns
    -------
    OccupancyResult
        ``U`` (per-CTU utility) and ``G_max`` (per-CTU occupancy gate
        across objects). ``U`` is *unnormalised* — downstream
        ``og_ipf.compute_og_a_plus_delta`` re-scales it via the
        K-weighted normaliser ``g`` (scale invariance).

    Notes
    -----
    Pure-Python list comprehension over objects is fine here: MOT17
    has < 50 boxes per frame on average, and the per-object gate /
    kernel are vectorised over CTUs. For a 9×15 grid + 50 boxes this
    is ≈ 2 ms / frame on a laptop.
    """
    cfg = cfg or OGConfig()
    # Padded dims drive the CTU grid (n_rows / n_cols); visible dims
    # clip individual CTU box corners so the bottom / right boundary
    # CTUs report their true visible area when computing overlap.
    clip_h = int(visible_h) if (visible_h is not None and visible_h > 0) else frame_h
    clip_w = int(visible_w) if (visible_w is not None and visible_w > 0) else frame_w
    grid_x, grid_y, n_rows, n_cols = make_ctu_grid(frame_h, frame_w, ctu_size)
    ctu_x1, ctu_y1, ctu_x2, ctu_y2 = _ctu_box_corners(
        n_rows, n_cols, ctu_size, clip_h, clip_w)

    # Filter zero-area boxes upfront — these would only inject NaN.
    valid_boxes: List[ObjectBox] = [b for b in boxes if b.w > 0 and b.h > 0]
    if not valid_boxes:
        return OccupancyResult(
            U=np.zeros((n_rows, n_cols), dtype=np.float64),
            G_max=np.zeros((n_rows, n_cols), dtype=np.float64),
            G_per_object=(None if not return_per_object
                          else np.zeros((0, n_rows, n_cols), dtype=np.float64)),
            n_objects=0,
        )

    g_stack = np.empty((len(valid_boxes), n_rows, n_cols), dtype=np.float64)
    u_stack = np.empty_like(g_stack)
    for j, box in enumerate(valid_boxes):
        g = occupancy_gate_per_object(box, ctu_x1, ctu_y1, ctu_x2, ctu_y2, cfg)
        d = distance_kernel_per_object(box, grid_x, grid_y, cfg)
        # Context score: smooth field outside the bbox only. ``(1 - g)``
        # also zeroes the context inside the bbox so we don't double-count
        # protection there (occupancy already dominates).
        c_score = (1.0 - g) * d
        # Per-object priority. The user spec defaults γ = 0 so size is
        # ignored; we keep the (w·h)^γ term as a hook for ablation.
        size_term = float((box.w * box.h) ** cfg.gamma) if cfg.gamma > 0 else 1.0
        priority  = box.class_priority * box.track_age * box.confidence * size_term
        u = priority * (cfg.alpha_in * g + cfg.alpha_ctx * c_score)
        g_stack[j] = g
        u_stack[j] = u

    g_max = _aggregate(g_stack, cfg)
    u_agg = _aggregate(u_stack, cfg)
    # Numerical floor: U_c never exactly zero (the analytic A+ takes
    # log of (U+ε), so a hard zero would cause log-warnings).
    u_agg = np.maximum(u_agg, cfg.eps_u)
    return OccupancyResult(
        U=u_agg,
        G_max=g_max,
        G_per_object=(g_stack if return_per_object else None),
        n_objects=len(valid_boxes),
    )


# ---------------------------------------------------------------------------
# Loader for per-frame box JSON files (written by save_boxes step)
# ---------------------------------------------------------------------------

def load_boxes_for_frame(boxes_dir, frame_idx: int) -> List[ObjectBox]:
    """Load a per-frame box JSON file written by ``save_boxes.py``.

    The file format is one JSON object per file::

        {
          "frame_idx": 0,
          "boxes": [
            {"xyxy": [x1, y1, x2, y2], "score": 0.92, "class": 0,
             "class_priority": 1.0, "track_age": 1.0},
            ...
          ]
        }

    Returns an empty list if the file is missing — callers should
    treat that as "no detections this frame" and produce a zero map.
    """
    from pathlib import Path
    import json as _json

    p = Path(boxes_dir) / f"boxes_{frame_idx:06d}.json"
    if not p.exists():
        return []
    payload = _json.loads(p.read_text(encoding="utf-8"))
    out: List[ObjectBox] = []
    for d in payload.get("boxes", []):
        try:
            out.append(ObjectBox.from_dict(d))
        except KeyError:
            continue
    return out


__all__ = [
    "OGConfig",
    "ObjectBox",
    "OccupancyResult",
    "make_ctu_grid",
    "occupancy_gate_per_object",
    "distance_kernel_per_object",
    "compute_occupancy_utility",
    "load_boxes_for_frame",
]
