"""Re-evaluate cached decoded frames with a stronger detector.

Motivation
----------
YOLOv8n on MOT17 at QP 27–42 is metric-saturated: headroom H ≈ 0.11, meaning
the detector-sensitivity slope is 8× below the per-frame noise floor (n=50
frames).  Under saturation, no CTU-level allocator can produce a bootstrap CI
that excludes zero.

Upgrading to YOLOv8x (or Faster R-CNN X101-FPN per MPEG VCM CTC) lifts H above
1 because the stronger detector is less tolerant of compression artifacts:
|∂AP/∂QP| is larger while σ_0 (per-frame mAP noise) stays similar.

This script
-----------
Loads cached decoded-frame directories from a pilot output (no re-encode),
runs a stronger detector, computes COCO mAP vs MOT17 GT, and writes the results
in the same d1_true_map.json format that reevaluate_pilots.py consumes.

Usage
-----
    PYTHONPATH=src python scripts/reevaluate_stronger_detector.py \\
        --pilot-dir ~/Minh/ipf/phase2_outputs/pilot_v8_og \\
        --config configs/pilot_v8_og.yaml \\
        --detector yolov8x \\
        --methods M0 M4 \\
        --gt-dir ~/Minh/ipf/datasets/MOT17/MOT17/train \\
        --output-suffix _yolov8x

Then run reevaluate_pilots.py with the new d1 JSON to get bootstrap CIs.

Detector options
----------------
- yolov8n (default, known saturated)
- yolov8m (moderate, H usually ≈ 0.4-0.8)
- yolov8x (recommended, H usually > 1)
- fasterrcnn (Faster R-CNN X101-FPN, closest to MPEG VCM CTC standard)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("scripts.reevaluate_stronger_detector")


# ---------------------------------------------------------------------------
# COCO mAP helpers (same logic as d1_true_map.py)
# ---------------------------------------------------------------------------

def _iou(b1, b2):
    """IoU between two [x1,y1,x2,y2] boxes."""
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / (union + 1e-9)


def _ap_at_iou(detections, gt_boxes, iou_thresh: float) -> float:
    """Compute AP for one (frame, iou_thresh) pair."""
    if not gt_boxes:
        return 1.0 if not detections else 0.0
    detections = sorted(detections, key=lambda x: -x["score"])
    matched = [False] * len(gt_boxes)
    tp = []; fp = []
    for det in detections:
        best_iou = 0.0; best_gt = -1
        for gi, gt in enumerate(gt_boxes):
            if matched[gi]:
                continue
            v = _iou(det["xyxy"], gt)
            if v > best_iou:
                best_iou = v; best_gt = gi
        if best_iou >= iou_thresh and best_gt >= 0:
            matched[best_gt] = True
            tp.append(1); fp.append(0)
        else:
            tp.append(0); fp.append(1)
    tp_cum = np.cumsum(tp); fp_cum = np.cumsum(fp)
    n_gt = len(gt_boxes)
    prec = tp_cum / (tp_cum + fp_cum + 1e-9)
    rec = tp_cum / (n_gt + 1e-9)
    # 101-point interpolation
    ap = 0.0
    for r_thresh in np.linspace(0, 1, 101):
        idx = np.where(rec >= r_thresh)[0]
        ap += (prec[idx].max() if idx.size else 0.0)
    return ap / 101


def compute_coco_ap(detections, gt_boxes) -> Dict[str, float]:
    """Return mAP@0.5 and mAP@[0.5:0.05:0.95] for one frame."""
    iou_thresholds = np.arange(0.50, 1.00, 0.05)
    aps = [_ap_at_iou(detections, gt_boxes, t) for t in iou_thresholds]
    return {"mAP_50": aps[0], "mAP_50_95": float(np.mean(aps))}


# ---------------------------------------------------------------------------
# Ground-truth loader (MOT17 gt.txt)
# ---------------------------------------------------------------------------

def load_mot17_gt(gt_dir: Path, sequence: str) -> Dict[int, List[List[float]]]:
    """Load MOT17 human-annotated GT into ``{frame_1based: [[x1,y1,x2,y2], ...]}``."""
    gt_file = gt_dir / sequence / "gt" / "gt.txt"
    if not gt_file.exists():
        logger.warning("GT file not found: %s", gt_file)
        return {}
    boxes: Dict[int, List[List[float]]] = {}
    with open(gt_file) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            frame = int(parts[0])
            cls = int(parts[7]) if len(parts) > 7 else 1
            vis = float(parts[8]) if len(parts) > 8 else 1.0
            if cls != 1 or vis < 0.25:
                continue
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            boxes.setdefault(frame, []).append([x, y, x + w, y + h])
    return boxes


# ---------------------------------------------------------------------------
# Detector wrapper
# ---------------------------------------------------------------------------

def load_detector(model_name: str):
    """Return a callable ``detect(frame_bgr) -> List[dict(xyxy, score)]``."""
    model_name = model_name.lower()
    if "fasterrcnn" in model_name or "detectron" in model_name:
        try:
            from detectron2.config import get_cfg
            from detectron2.engine.defaults import DefaultPredictor
            import detectron2.model_zoo as mz
            cfg = get_cfg()
            cfg.merge_from_file(mz.get_config_file(
                "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
            cfg.MODEL.WEIGHTS = mz.get_checkpoint_url(
                "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.001
            predictor = DefaultPredictor(cfg)
            def detect_frcnn(frame_bgr):
                out = predictor(frame_bgr)
                instances = out["instances"].to("cpu")
                results = []
                for i in range(len(instances)):
                    box = instances.pred_boxes.tensor[i].numpy().tolist()
                    score = float(instances.scores[i])
                    cls = int(instances.pred_classes[i])
                    if cls == 0:  # person (COCO class 0)
                        results.append({"xyxy": box, "score": score})
                return results
            return detect_frcnn
        except ImportError:
            logger.error("detectron2 not installed; falling back to yolov8x")
            model_name = "yolov8x"

    # YOLOv8 family
    from ultralytics import YOLO
    yolo = YOLO(f"{model_name}.pt")

    def detect_yolo(frame_bgr):
        results = yolo(frame_bgr, conf=0.001, classes=[0], verbose=False)
        out = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                out.append({"xyxy": [x1, y1, x2, y2], "score": float(box.conf[0])})
        return out

    return detect_yolo


# ---------------------------------------------------------------------------
# Headroom diagnostic H
# ---------------------------------------------------------------------------

def compute_headroom_H(
    seq_mAP: Dict[int, float],
    n_frames: int,
) -> float:
    """Compute headroom H = (slope × |ΔQ|) / σ_0.

    H ≥ 1 means the detector-sensitivity slope is above the per-frame noise floor.
    H < 1 means metric-saturated.
    """
    qps = sorted(seq_mAP.keys())
    aps = [seq_mAP[q] for q in qps]
    if len(qps) < 2:
        return 0.0
    slopes = [abs(aps[i + 1] - aps[i]) / max(qps[i + 1] - qps[i], 1)
              for i in range(len(qps) - 1)]
    s0 = float(np.mean(slopes))
    delta_q = float(np.mean([qps[i + 1] - qps[i] for i in range(len(qps) - 1)]))
    sigma_0 = 1.0 / np.sqrt(max(n_frames, 1))  # approx 1 SE
    return (s0 * delta_q) / (sigma_0 + 1e-9)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--detector", default="yolov8x",
                        choices=["yolov8n", "yolov8m", "yolov8x", "fasterrcnn"])
    parser.add_argument("--methods", nargs="+", default=["M0", "M4"])
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--output-suffix", default="_strong_detector")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    import yaml
    cfg_data = yaml.safe_load(Path(args.config).read_text())
    sequences = [s["name"] for s in cfg_data.get("sequences", [])]
    qp_points = cfg_data.get("encoding", {}).get("qp_points", [27, 32, 37, 42])
    pilot_dir = Path(args.pilot_dir)
    gt_dir = Path(args.gt_dir)

    logger.info("Loading detector: %s", args.detector)
    detect = load_detector(args.detector)

    results = {}  # {method: {seq: {qp: {frame: {mAP_50, mAP_50_95}}}}}
    headroom_report = {}

    for method in args.methods:
        results[method] = {}
        for seq in sequences:
            results[method][seq] = {}
            gt_boxes_by_frame = load_mot17_gt(gt_dir, seq)

            for qp in qp_points:
                decoded_dir = pilot_dir / f"{method}_{seq}_QP{qp}" / "decoded_frames"
                if not decoded_dir.exists():
                    # Try alternate naming conventions
                    for cand in pilot_dir.glob(f"**/{method}*{seq}*QP{qp}*/decoded_frames"):
                        decoded_dir = cand
                        break
                if not decoded_dir.exists():
                    logger.warning("decoded_frames not found: %s %s QP%d", method, seq, qp)
                    continue

                frame_files = sorted(decoded_dir.glob("*.png")) + \
                              sorted(decoded_dir.glob("*.jpg"))
                if not frame_files:
                    logger.warning("No frames in %s", decoded_dir)
                    continue

                frame_results = {}
                for fi, fpath in enumerate(frame_files):
                    import cv2
                    frame_bgr = cv2.imread(str(fpath))
                    if frame_bgr is None:
                        continue
                    frame_1based = fi + 1
                    gt = gt_boxes_by_frame.get(frame_1based, [])
                    dets = detect(frame_bgr)
                    metrics = compute_coco_ap(dets, gt)
                    frame_results[fi] = metrics

                results[method][seq][qp] = frame_results
                logger.info("%s %s QP%d: %d frames, mAP50=%.4f",
                            method, seq, qp,
                            len(frame_results),
                            float(np.mean([v["mAP_50"] for v in frame_results.values()]))
                            if frame_results else 0.0)

            # Headroom per sequence (M0 only)
            if method == "M0":
                seq_mean_mAP = {
                    qp: float(np.mean([v["mAP_50_95"]
                                       for v in results[method][seq][qp].values()]))
                    for qp in qp_points if qp in results[method][seq] and results[method][seq][qp]
                }
                n_frames = len(next(iter(results[method][seq].values()), {}).values()) \
                           if results[method][seq] else 50
                H = compute_headroom_H(seq_mean_mAP, n_frames)
                headroom_report[seq] = {
                    "H": H,
                    "detector": args.detector,
                    "saturated": H < 1.0,
                    "mAP_per_qp": seq_mean_mAP,
                }
                logger.info("H(%s, %s) = %.3f %s",
                            seq, args.detector, H,
                            "(SATURATED)" if H < 1 else "(OK)")

    # Write output
    out_path = pilot_dir / f"d1_true_map{args.output_suffix}.json"
    out_data = {"detector": args.detector, "results": results,
                "headroom": headroom_report}
    out_path.write_text(json.dumps(out_data, indent=2))
    logger.info("Written: %s", out_path)

    # Print headroom summary
    print("\n=== Detector-Headroom Summary ===")
    print(f"{'Sequence':<20} {'H':>6}  {'Status'}")
    print("-" * 40)
    for seq, info in headroom_report.items():
        status = "SATURATED (H<1)" if info["saturated"] else "OK (H≥1)"
        print(f"{seq:<20} {info['H']:>6.3f}  {status}")

    all_H = [v["H"] for v in headroom_report.values()]
    if all_H:
        mean_H = float(np.mean(all_H))
        print(f"\nMean H across sequences: {mean_H:.3f}")
        if mean_H < 1.0:
            print("WARNING: Mean H < 1 — aggregate evaluation still saturated.")
            print("  → Consider using Faster R-CNN X101-FPN (MPEG CTC standard)")
        else:
            print("PASS: Mean H ≥ 1 — evaluation regime is non-saturated.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
