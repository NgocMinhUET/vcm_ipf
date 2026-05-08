"""Hand-crafted rectangle-ROI baseline (M_rect) for the comprehensive bench.

Why this script exists
----------------------
``PROJECT_AUDIT.md`` §5.4 (Audit §1.4 + §2.4) asks: does our learned IPF
add anything over a *trivial* hand-crafted ROI rectangle? This script
generates exactly such a baseline:

    1. Run the same YOLOv8 used for evaluation on the uncompressed
       reference frames (cached once per sequence at
       ``<output_dir>/_reference_boxes/<seq>.json``).
    2. For each frame, compute the union of *expanded* detection boxes
       (expansion factor = 0.1 by default, matching ``roi_expansion``).
    3. Produce a per-CTU mask: 1 if any expanded box overlaps the CTU,
       else 0.
    4. Build the dQP map::
           dQP(c) = -delta_roi if mask(c) else +delta_bg
       Default ``delta_roi = 2``, ``delta_bg = 2`` so the map sums to zero.
       Optionally apply rate-neutral projection if a K_c grid is provided.
    5. Save under
       ``<phase1_root>/<run_prefix><seq>/M_rect/qp_vtm_delta/qp_NNNNNN.txt``
       so the standard pipeline picks it up via ``_find_qp_maps``.

This is the simplest possible "saliency-aware" QP allocation and serves
as the floor that any learned method must beat. If M4 / M_oracle do not
beat M_rect statistically, the IPF concept adds no value.

Usage
-----
::

    PYTHONPATH=src python scripts/build_rect_baseline.py \\
        --reference-boxes ~/Minh/ipf/phase2_outputs/pilot_v8b/_reference_boxes/MOT17-04-DPM.json \\
        --frames-dir ~/Minh/ipf/datasets/MOT17/MOT17/train/MOT17-04-DPM/img1 \\
        --width 1920 --height 1152 --ctu-size 128 \\
        --output-dir ~/Minh/ipf/phase1_outputs/multi_seq_MOT17-04-DPM/M_rect/qp_vtm_delta \\
        --delta-roi 2 --delta-bg 2 \\
        --expansion 0.1 \\
        --n-frames 50

The script is self-contained: no detector is invoked here, it just reads
the cached reference-boxes JSON. (If those don't exist yet, run
phase 2 once with the target sequence to populate the cache.)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("phase2.scripts.build_rect_baseline")


def _expand_box(
    box: Tuple[float, float, float, float],
    expansion: float,
    w: int,
    h: int,
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    bw = x2 - x1; bh = y2 - y1
    dx = bw * expansion / 2.0
    dy = bh * expansion / 2.0
    return (
        max(0.0, x1 - dx),
        max(0.0, y1 - dy),
        min(float(w), x2 + dx),
        min(float(h), y2 + dy),
    )


def _ctu_mask_for_frame(
    boxes: List[Tuple[float, float, float, float]],
    width: int,
    height: int,
    ctu_size: int,
    expansion: float,
) -> np.ndarray:
    n_rows = (height + ctu_size - 1) // ctu_size
    n_cols = (width + ctu_size - 1) // ctu_size
    mask = np.zeros((n_rows, n_cols), dtype=bool)
    for box in boxes:
        x1, y1, x2, y2 = _expand_box(box, expansion, width, height)
        c1 = max(0, int(x1) // ctu_size)
        c2 = min(n_cols - 1, int(x2 - 1) // ctu_size)
        r1 = max(0, int(y1) // ctu_size)
        r2 = min(n_rows - 1, int(y2 - 1) // ctu_size)
        if c2 < c1 or r2 < r1:
            continue
        mask[r1:r2 + 1, c1:c2 + 1] = True
    return mask


def _build_delta_grid(
    mask: np.ndarray,
    delta_roi: int,
    delta_bg: int,
) -> np.ndarray:
    delta = np.where(mask, -int(delta_roi), +int(delta_bg)).astype(np.int32)
    return delta


def _rate_neutral_project(
    delta: np.ndarray, K_grid: np.ndarray
) -> np.ndarray:
    """Closed-form scalar shift such that Σ K · 2^{-(δ + s)/6} = Σ K."""
    K = np.asarray(K_grid, dtype=np.float64)
    K = np.maximum(K, 1e-9)
    denom = float(np.sum(K) + 1e-9)
    weighted = float(np.sum(K * np.power(2.0, -delta / 6.0)))
    ratio = max(weighted / denom, 1e-12)
    s = 6.0 * np.log2(ratio)
    return delta.astype(np.float64) + s


def _write_delta_qp_file(
    delta: np.ndarray,
    path: Path,
    frame_idx: int,
    delta_min: int = -8,
    delta_max: int = 4,
) -> None:
    clipped = np.clip(np.rint(delta), delta_min, delta_max).astype(np.int32)
    n_rows, n_cols = clipped.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(
            f"# frame={frame_idx} rows={n_rows} cols={n_cols} "
            f"type=delta delta_min={delta_min} delta_max={delta_max}\n"
        )
        for r in range(n_rows):
            fh.write(" ".join(f"{clipped[r, c]:+d}" for c in range(n_cols)))
            fh.write("\n")


def _load_reference_boxes(path: Path) -> List[List[Tuple[float, float, float, float]]]:
    """Reference-boxes JSON shape: ``[[box, box, ...], [box, ...], ...]``
    where ``box`` is either ``[x1, y1, x2, y2]`` or
    ``{"box": [x1,y1,x2,y2], "score": s, "class": c}``.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    out: List[List[Tuple[float, float, float, float]]] = []
    for frame in data:
        boxes_this = []
        for entry in frame:
            if isinstance(entry, dict) and "box" in entry:
                b = entry["box"]
            else:
                b = entry
            if len(b) >= 4:
                boxes_this.append((float(b[0]), float(b[1]), float(b[2]), float(b[3])))
        out.append(boxes_this)
    return out


def build_rect_baseline(
    reference_boxes_json: Path,
    output_dir: Path,
    width: int,
    height: int,
    ctu_size: int = 128,
    expansion: float = 0.1,
    delta_roi: int = 2,
    delta_bg: int = 2,
    n_frames: int = 0,
    K_npz: Optional[Path] = None,
    delta_min: int = -8,
    delta_max: int = 4,
) -> int:
    """Generate per-frame relative dQP maps for the rectangle baseline."""
    boxes_per_frame = _load_reference_boxes(reference_boxes_json)
    if n_frames > 0:
        boxes_per_frame = boxes_per_frame[:n_frames]

    K_per_frame = None
    if K_npz is not None and K_npz.exists():
        loaded = np.load(K_npz)
        if "K_per_frame" in loaded.files:
            K_per_frame = loaded["K_per_frame"]
        elif "K" in loaded.files:
            K_per_frame = loaded["K"]

    n_written = 0
    for fi, boxes in enumerate(boxes_per_frame):
        mask = _ctu_mask_for_frame(boxes, width, height, ctu_size, expansion)
        delta = _build_delta_grid(mask, delta_roi, delta_bg)
        if K_per_frame is not None and fi < len(K_per_frame):
            delta = _rate_neutral_project(delta, K_per_frame[fi])
        out_path = output_dir / f"qp_{fi:06d}.txt"
        _write_delta_qp_file(delta, out_path, fi, delta_min, delta_max)
        n_written += 1

    logger.info("Wrote %d M_rect delta-QP files to %s", n_written, output_dir)
    return n_written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build M_rect hand-crafted ROI baseline")
    parser.add_argument("--reference-boxes", required=True, type=Path,
                        help="Cached reference-boxes JSON (from encode_pipeline)")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--ctu-size", type=int, default=128)
    parser.add_argument("--expansion", type=float, default=0.1)
    parser.add_argument("--delta-roi", type=int, default=2)
    parser.add_argument("--delta-bg", type=int, default=2)
    parser.add_argument("--n-frames", type=int, default=0)
    parser.add_argument("--K-npz", type=Path, default=None)
    parser.add_argument("--delta-min", type=int, default=-8)
    parser.add_argument("--delta-max", type=int, default=4)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except AttributeError:
        pass

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    build_rect_baseline(
        reference_boxes_json=args.reference_boxes,
        output_dir=args.output_dir,
        width=args.width, height=args.height,
        ctu_size=args.ctu_size,
        expansion=args.expansion,
        delta_roi=args.delta_roi, delta_bg=args.delta_bg,
        n_frames=args.n_frames,
        K_npz=args.K_npz,
        delta_min=args.delta_min, delta_max=args.delta_max,
    )


if __name__ == "__main__":
    main()
