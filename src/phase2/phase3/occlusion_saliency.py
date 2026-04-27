"""Phase 3 Stage B.1 — Occlusion saliency for task-driven CTU importance.

Reference: ``11_PHASE3_RESEARCH_PROTOCOL.md`` §3.1 + Zeiler & Fergus 2014.

For each frame *f* and each CTU *c*, we:

1. Run the detector once on the original frame and record the per-box
   confidences (baseline).
2. Replace the pixels inside CTU *c* with a Gaussian-blurred version
   (σ ≈ 8 px ≈ what high-QP coding looks like at QP ~ 50).
3. Re-run the detector and compute

       Φ_oracle(c) = Σ_{boxes b that overlap c}  max(0, conf_baseline(b) - conf_blurred(b))

   This is the cumulative confidence drop attributable to corrupting
   CTU *c*. It is our task-driven ground-truth importance map.

The output is one ``.npy`` per frame containing a 2-D array of shape
``(n_rows, n_cols)`` where ``n_rows = ceil(H/ctu_size)`` and
``n_cols = ceil(W/ctu_size)``. Both raw and percentile-normalized
versions are written so downstream code can decide whether to apply
the same EMA-percentile trick used for Φ_IPF.

Diagnostic side-effect: when ``--phi-ipf-dir`` is given, the script
loads the matching IPF field tensors saved by Phase 1 and prints the
Spearman correlation ρ(Φ_IPF, Φ_oracle) per frame plus a global value.
This is the "free check" promised in the Stage A discussion — it does
**not** gate execution but raises a WARN in the log if ρ < 0.30 so
the user can decide whether to pivot.

Run with::

    python -m phase2.phase3.occlusion_saliency \
        --frames-dir /data/MOT17/MOT17-04-DPM/img1 \
        --n-frames 50 \
        --output-dir ~/Minh/ipf/phase3_outputs/saliency/MOT17-04-DPM \
        --ctu-size 128 \
        --blur-sigma 8 \
        --batch-size 32 \
        --device cuda:0

Wall-clock budget: ~1.5 s/frame on A100, ~12 min for 50 × 3 sequences.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("phase2.phase3.occlusion_saliency")


# ---------------------------------------------------------------------------
# Geometry helpers.
# ---------------------------------------------------------------------------

def _ctu_grid(height: int, width: int, ctu_size: int) -> Tuple[int, int]:
    n_rows = (height + ctu_size - 1) // ctu_size
    n_cols = (width + ctu_size - 1) // ctu_size
    return n_rows, n_cols


def _ctu_box(r: int, c: int, ctu_size: int, h: int, w: int) -> Tuple[int, int, int, int]:
    """Return (x1, y1, x2, y2) for the CTU at grid position (r, c)."""
    x1 = c * ctu_size
    y1 = r * ctu_size
    x2 = min(x1 + ctu_size, w)
    y2 = min(y1 + ctu_size, h)
    return x1, y1, x2, y2


def _box_iou(b1: Tuple[float, float, float, float],
             b2: Tuple[float, float, float, float]) -> float:
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter + 1e-9
    return float(inter / union)


# ---------------------------------------------------------------------------
# Detector batch wrapper (YOLOv8 via Ultralytics).
# ---------------------------------------------------------------------------

@dataclass
class FrameDetections:
    boxes: np.ndarray   # (N, 4) xyxy
    scores: np.ndarray  # (N,)
    classes: np.ndarray  # (N,)


class YoloBatch:
    def __init__(self, model_path: str, conf: float, device: str) -> None:
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device

    def detect(self, images: List[np.ndarray]) -> List[FrameDetections]:
        """Run the model on a list of BGR uint8 images. Returns one entry per input."""
        results = self.model(images, conf=self.conf, device=self.device, verbose=False)
        out: List[FrameDetections] = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                out.append(FrameDetections(
                    boxes=np.zeros((0, 4), dtype=np.float32),
                    scores=np.zeros((0,), dtype=np.float32),
                    classes=np.zeros((0,), dtype=np.int32),
                ))
                continue
            xyxy = r.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            sc = r.boxes.conf.detach().cpu().numpy().astype(np.float32)
            cl = r.boxes.cls.detach().cpu().numpy().astype(np.int32)
            out.append(FrameDetections(boxes=xyxy, scores=sc, classes=cl))
        return out


# ---------------------------------------------------------------------------
# Saliency core.
# ---------------------------------------------------------------------------

def _blur_ctu(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, sigma: float) -> np.ndarray:
    """Return a copy of frame with a Gaussian blur applied to CTU [x1:x2, y1:y2]."""
    import cv2
    out = frame.copy()
    patch = out[y1:y2, x1:x2]
    if patch.size == 0:
        return out
    k = max(3, int(2 * round(2.0 * sigma) + 1))
    if k % 2 == 0:
        k += 1
    out[y1:y2, x1:x2] = cv2.GaussianBlur(patch, (k, k), sigmaX=sigma, sigmaY=sigma)
    return out


def _match_score(baseline: FrameDetections, perturbed: FrameDetections,
                 iou_thr: float = 0.5) -> List[Tuple[int, float]]:
    """For each baseline box, return (-1 or matched_idx, residual_conf)."""
    matched: List[Tuple[int, float]] = []
    used = np.zeros(len(perturbed.boxes), dtype=bool)
    for i, b in enumerate(baseline.boxes):
        best = -1
        best_iou = iou_thr
        for j, p in enumerate(perturbed.boxes):
            if used[j] or perturbed.classes[j] != baseline.classes[i]:
                continue
            iou = _box_iou(tuple(b.tolist()), tuple(p.tolist()))
            if iou > best_iou:
                best_iou = iou
                best = j
        if best >= 0:
            used[best] = True
            matched.append((best, float(perturbed.scores[best])))
        else:
            matched.append((-1, 0.0))
    return matched


def _saliency_for_frame(
    frame: np.ndarray,
    yolo: YoloBatch,
    ctu_size: int,
    sigma: float,
    batch_size: int,
) -> np.ndarray:
    """Compute per-CTU saliency for one frame. Returns (n_rows, n_cols) float32."""
    h, w = frame.shape[:2]
    n_rows, n_cols = _ctu_grid(h, w, ctu_size)
    sal = np.zeros((n_rows, n_cols), dtype=np.float32)

    baseline = yolo.detect([frame])[0]
    if len(baseline.boxes) == 0:
        # No detections to lose → uniform zero saliency (no task signal to preserve).
        return sal

    base_total = float(baseline.scores.sum())

    # Build the list of (r, c, perturbed_image) for every CTU.
    work: List[Tuple[int, int, np.ndarray]] = []
    for r in range(n_rows):
        for c in range(n_cols):
            x1, y1, x2, y2 = _ctu_box(r, c, ctu_size, h, w)
            work.append((r, c, _blur_ctu(frame, x1, y1, x2, y2, sigma)))

    # Batched detection.
    for start in range(0, len(work), batch_size):
        chunk = work[start:start + batch_size]
        imgs = [item[2] for item in chunk]
        dets = yolo.detect(imgs)
        for (r, c, _img), det in zip(chunk, dets):
            # Strict drop: total baseline confidence minus surviving confidence.
            matched = _match_score(baseline, det)
            survived = sum(score for _, score in matched)
            drop = max(0.0, base_total - survived)
            # Restrict credit: only attribute drop from boxes that overlap CTU.
            x1, y1, x2, y2 = _ctu_box(r, c, ctu_size, h, w)
            ctu_box_xyxy = (float(x1), float(y1), float(x2), float(y2))
            overlap_flag = False
            for b in baseline.boxes:
                if _box_iou(tuple(b.tolist()), ctu_box_xyxy) > 0.0:
                    overlap_flag = True
                    break
            sal[r, c] = drop if overlap_flag else 0.5 * drop  # background gets half-credit

    return sal


# ---------------------------------------------------------------------------
# Φ_IPF correlation diagnostic.
# ---------------------------------------------------------------------------

def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 4 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def _load_phi_ipf(phi_dir: Path, frame_idx: int,
                  n_rows: int, n_cols: int) -> Optional[np.ndarray]:
    """Try several conventional names. Phase 1 saves Φ as ``field_XXXXXX.npy``
    when ``output.save_field_npy=True``. We accept either a 2-D field tensor
    (re-aggregated to CTU) or a CTU-grid tensor."""
    candidates = [
        phi_dir / f"field_{frame_idx:06d}.npy",
        phi_dir / f"phi_{frame_idx:06d}.npy",
        phi_dir / f"importance_{frame_idx:06d}.npy",
    ]
    for path in candidates:
        if path.exists():
            arr = np.load(path)
            if arr.ndim == 2 and arr.shape == (n_rows, n_cols):
                return arr.astype(np.float64)
            if arr.ndim == 2:
                # Pixel-resolution field: pool to CTU by max.
                from math import ceil
                ctu_h = ceil(arr.shape[0] / n_rows)
                ctu_w = ceil(arr.shape[1] / n_cols)
                pooled = np.zeros((n_rows, n_cols), dtype=np.float64)
                for r in range(n_rows):
                    for c in range(n_cols):
                        block = arr[r * ctu_h:(r + 1) * ctu_h,
                                    c * ctu_w:(c + 1) * ctu_w]
                        pooled[r, c] = float(block.max() if block.size else 0.0)
                return pooled
    return None


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 occlusion-saliency generator")
    parser.add_argument("--frames-dir", required=True, help="Directory of .jpg frames")
    parser.add_argument("--output-dir", required=True, help="Output directory for .npy maps")
    parser.add_argument("--n-frames", type=int, default=50)
    parser.add_argument("--ctu-size", type=int, default=128)
    parser.add_argument("--blur-sigma", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--detector", default="yolov8n.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--phi-ipf-dir", default="",
                        help="Optional Phase 1 Φ-tensor directory for ρ diagnostic")
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="Sample every k-th frame (1 = all consecutive)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    frames_dir = Path(args.frames_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    import cv2
    paths = sorted([p for p in frames_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if args.frame_stride > 1:
        paths = paths[::args.frame_stride]
    paths = paths[:args.n_frames]
    if not paths:
        raise SystemExit(f"No frames found under {frames_dir}")

    yolo = YoloBatch(args.detector, args.confidence, args.device)

    rho_records: List[Tuple[int, float, float]] = []
    summary_phi_ipf: List[float] = []
    summary_phi_ora: List[float] = []
    saved = 0

    for idx, fpath in enumerate(paths):
        img = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Could not read %s — skip", fpath)
            continue
        sal = _saliency_for_frame(img, yolo,
                                  ctu_size=args.ctu_size,
                                  sigma=args.blur_sigma,
                                  batch_size=args.batch_size)
        np.save(output_dir / f"phi_oracle_{idx:06d}.npy", sal)

        # Percentile-normalized companion (mirrors Phase 1 normalization).
        p_lo = float(np.percentile(sal, 5.0))
        p_hi = float(np.percentile(sal, 95.0))
        denom = max(1e-9, p_hi - p_lo)
        sal_norm = np.clip((sal - p_lo) / denom, 0.0, 1.0)
        np.save(output_dir / f"phi_oracle_norm_{idx:06d}.npy", sal_norm)

        if args.phi_ipf_dir:
            phi_ipf = _load_phi_ipf(Path(args.phi_ipf_dir).expanduser().resolve(),
                                    idx, sal.shape[0], sal.shape[1])
            if phi_ipf is not None:
                rho = _spearman(phi_ipf.flatten(), sal.flatten())
                # Pearson too, useful as a sanity stat alongside rank-based.
                if float(np.std(phi_ipf)) > 0.0 and float(np.std(sal)) > 0.0:
                    pear = float(np.corrcoef(phi_ipf.flatten(), sal.flatten())[0, 1])
                else:
                    pear = float("nan")
                rho_records.append((idx, rho, pear))
                summary_phi_ipf.extend(phi_ipf.flatten().tolist())
                summary_phi_ora.extend(sal.flatten().tolist())
                logger.info("frame=%06d  ρ_spearman=%.3f  ρ_pearson=%.3f", idx, rho, pear)

        saved += 1

    logger.info("Saved %d frames to %s", saved, output_dir)

    # Final aggregate diagnostic.
    if rho_records:
        global_rho = _spearman(np.asarray(summary_phi_ipf), np.asarray(summary_phi_ora))
        per_frame = [r for _, r, _ in rho_records if not np.isnan(r)]
        med = float(np.median(per_frame)) if per_frame else float("nan")
        logger.info("=== Φ_IPF vs Φ_oracle correlation summary ===")
        logger.info("  per-frame ρ_spearman:  median=%.3f  min=%.3f  max=%.3f  N=%d",
                    med, float(np.min(per_frame)) if per_frame else float("nan"),
                    float(np.max(per_frame)) if per_frame else float("nan"),
                    len(per_frame))
        logger.info("  global  ρ_spearman:    %.3f  (over %d CTU samples)",
                    global_rho, len(summary_phi_ipf))
        if not np.isnan(global_rho) and global_rho < 0.30:
            logger.warning("ρ_global = %.3f < 0.30 — IPF is a POOR proxy for task "
                           "importance. Consider switching baseline to Φ_oracle direct "
                           "or revisiting the kernel/mass design.", global_rho)
        with open(output_dir / "rho_diagnostic.json", "w", encoding="utf-8") as f:
            json.dump({
                "global_spearman": global_rho,
                "median_spearman": med,
                "per_frame": [{"frame": i, "spearman": r, "pearson": p}
                               for i, r, p in rho_records],
                "n_frames": len(rho_records),
                "ctu_size": args.ctu_size,
                "blur_sigma": args.blur_sigma,
            }, f, indent=2)


if __name__ == "__main__":
    main()
