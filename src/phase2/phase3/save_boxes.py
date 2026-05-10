"""Phase 3 Stage C — persist per-frame YOLO bounding boxes for OG-IPF.

The legacy occlusion-saliency stage runs YOLO and writes only the
collapsed Φ map. OG-IPF needs the **boxes themselves** — so this
module re-runs the same detector on the same frame range and saves
one tiny JSON per frame::

    boxes_<frame>.json
    {
      "frame_idx": 0,
      "frame_size": [W, H],
      "ctu_size": 128,
      "boxes": [
        {"xyxy": [x1, y1, x2, y2], "score": 0.92, "class": 0,
         "class_priority": 1.0},
        ...
      ]
    }

The output directory is the **same** as the saliency directory, so
downstream code can find both with a single ``--saliency-dir`` flag.

We could have monkey-patched ``occlusion_saliency.py`` to dump boxes
as a side effect, but keeping it as a separate one-shot stage means:

* OG-IPF can be run against any pre-existing saliency dir without
  re-computing the (expensive) occlusion saliency,
* the box data and the saliency data are independent caches — if the
  user wants to re-detect with a different model / threshold, only
  the box step needs to be re-run.

Run::

    PYTHONPATH=src python -m phase2.phase3.save_boxes \\
        --frames-dir ~/Minh/ipf/datasets/MOT17/MOT17/train/MOT17-04-DPM/img1 \\
        --output-dir ~/Minh/ipf/phase3_outputs/saliency/MOT17-04-DPM \\
        --n-frames 50 \\
        --detector yolov8n.pt \\
        --confidence 0.25 \\
        --device cuda:0 \\
        --classes 0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger("phase2.phase3.save_boxes")


# ---------------------------------------------------------------------------
# Class priority
# ---------------------------------------------------------------------------

# COCO ID 0 = person → priority 1.0; everything else default 0.5 unless the
# user overrides via --class-priority. The file format leaves the field
# explicit on each box so downstream code never has to guess.
DEFAULT_CLASS_PRIORITY = {
    0: 1.0,    # person
    1: 0.7,    # bicycle
    2: 0.7,    # car
    3: 0.7,    # motorcycle
    5: 0.7,    # bus
    7: 0.7,    # truck
}


def _resolve_priority(coco_class: int,
                       overrides: Optional[dict] = None) -> float:
    if overrides and coco_class in overrides:
        return float(overrides[coco_class])
    return float(DEFAULT_CLASS_PRIORITY.get(coco_class, 0.5))


# ---------------------------------------------------------------------------
# Detector wrapper (kept very thin; mirrors occlusion_saliency.YoloBatch)
# ---------------------------------------------------------------------------

def _yolo_detect(model, frames: List[np.ndarray], conf: float, device: str,
                 classes: Optional[Sequence[int]]) -> List[dict]:
    """Single-batch YOLO call returning a list of (xyxy, scores, classes) dicts."""
    results = model(frames, conf=conf, device=device, verbose=False,
                    classes=list(classes) if classes else None)
    out: List[dict] = []
    for r in results:
        if not hasattr(r, "boxes") or r.boxes is None or len(r.boxes) == 0:
            out.append({"xyxy": [], "scores": [], "classes": []})
            continue
        xyxy = r.boxes.xyxy.cpu().numpy().astype(float).tolist()
        scores = r.boxes.conf.cpu().numpy().astype(float).tolist()
        cls    = r.boxes.cls.cpu().numpy().astype(int).tolist()
        out.append({"xyxy": xyxy, "scores": scores, "classes": cls})
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Persist per-frame YOLO boxes for OG-IPF.")
    ap.add_argument("--frames-dir", required=True,
                    help="Directory of input image frames (jpg/png).")
    ap.add_argument("--output-dir", required=True,
                    help="Where to write boxes_<frame>.json (typically the "
                         "same as the saliency directory).")
    ap.add_argument("--n-frames", type=int, default=50)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--ctu-size", type=int, default=128,
                    help="Recorded as metadata only; OG-IPF uses it via "
                         "compute_occupancy_utility.")
    ap.add_argument("--detector", default="yolov8n.pt")
    ap.add_argument("--confidence", type=float, default=0.25)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--classes", nargs="*", type=int, default=[0],
                    help="COCO class IDs to keep (default: [0] = person).")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--class-priority-json", default="",
                    help="Optional JSON file mapping {class_id: priority}.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    overrides: Optional[dict] = None
    if args.class_priority_json:
        with open(args.class_priority_json, "r", encoding="utf-8") as f:
            overrides = {int(k): float(v) for k, v in json.load(f).items()}

    frames_dir = Path(args.frames_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import — keep `save_boxes` importable in CPU-only sanity tests.
    import cv2
    from ultralytics import YOLO

    paths = sorted([p for p in frames_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if args.frame_stride > 1:
        paths = paths[::args.frame_stride]
    paths = paths[:args.n_frames]
    if not paths:
        raise SystemExit(f"No frames found under {frames_dir}")

    model = YOLO(args.detector)
    n_written = 0

    for batch_start in range(0, len(paths), args.batch_size):
        batch_paths = paths[batch_start:batch_start + args.batch_size]
        imgs = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in batch_paths]
        if any(im is None for im in imgs):
            for i, im in enumerate(imgs):
                if im is None:
                    logger.warning("Could not read %s — empty box list",
                                    batch_paths[i])
                    imgs[i] = np.zeros((1, 1, 3), dtype=np.uint8)
        det_lists = _yolo_detect(model, imgs, args.confidence, args.device,
                                  args.classes)
        for offset, det in enumerate(det_lists):
            idx = batch_start + offset
            h, w = imgs[offset].shape[:2]
            boxes_payload = []
            for (xyxy, score, cls) in zip(det["xyxy"], det["scores"], det["classes"]):
                if args.classes and cls not in args.classes:
                    continue
                boxes_payload.append({
                    "xyxy": [float(v) for v in xyxy],
                    "score": float(score),
                    "class": int(cls),
                    "class_priority": _resolve_priority(int(cls), overrides),
                })
            payload = {
                "frame_idx": idx,
                "frame_size": [int(w), int(h)],
                "ctu_size": int(args.ctu_size),
                "detector": args.detector,
                "confidence_threshold": float(args.confidence),
                "boxes": boxes_payload,
            }
            with open(output_dir / f"boxes_{idx:06d}.json", "w",
                       encoding="utf-8") as f:
                json.dump(payload, f)
            n_written += 1

    logger.info("save_boxes complete: %d files written under %s",
                 n_written, output_dir)


if __name__ == "__main__":
    main()
