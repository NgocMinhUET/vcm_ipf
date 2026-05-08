"""MOT17 ground-truth loader.

Replaces the YOLO-on-uncompressed pseudo-GT used in earlier evaluations
with the human-annotated bounding boxes shipped with MOT17. This removes
the most-cited weakness of the pipeline (PROJECT_AUDIT §1.5 #4): the
"ground truth" was YOLO's own noise on uncompressed frames, not real
objects, so any detector failure on compressed frames was conflated with
detector quirks on uncompressed ones.

MOT17 ``gt/gt.txt`` format
--------------------------
Each line is:

    frame_id, person_id, x, y, w, h, conf, class, visibility

* ``frame_id``: 1-indexed.
* ``x, y, w, h``: bbox in pixel coordinates of the original frame
  (1920×1080 for the sequences we use).
* ``conf``: ``1`` means the box is active GT (use it). ``0`` means the
  box should be ignored entirely (e.g. distractor regions).
* ``class``: 1 = pedestrian (the only class we care about for VCM-MOT
  evaluation; the encoded YUV is also detected as COCO class 0 = person,
  so we map MOT class 1 ↔ COCO class 0).
* ``visibility``: 0..1; we keep all visible boxes by default and let the
  caller filter on this if it wants stricter evaluation.

This loader returns ``DetectionResult`` objects compatible with the rest
of the evaluation stack (``compute_coco_map_from_detections`` etc.) so it
plugs in as a drop-in replacement for the pseudo-GT detection pass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("phase2.evaluation.mot17_gt")


# MOT17 → COCO class mapping. Class 1 in MOT is pedestrian; class 0 in
# COCO is person.
MOT17_PEDESTRIAN_CLASS = 1
COCO_PERSON_CLASS = 0


@dataclass
class _GTBox:
    frame_id: int       # 1-indexed in MOT17
    x: float
    y: float
    w: float
    h: float
    conf: int
    cls: int
    visibility: float


def _parse_gt_line(line: str) -> Optional[_GTBox]:
    parts = line.strip().split(",")
    if len(parts) < 9:
        return None
    try:
        return _GTBox(
            frame_id=int(parts[0]),
            x=float(parts[2]), y=float(parts[3]),
            w=float(parts[4]), h=float(parts[5]),
            conf=int(parts[6]), cls=int(parts[7]),
            visibility=float(parts[8]),
        )
    except ValueError:
        return None


def load_mot17_gt_for_sequence(
    sequence_root: Path,
    n_frames: int,
    *,
    min_visibility: float = 0.0,
    require_active: bool = True,
    coco_class: int = COCO_PERSON_CLASS,
):
    """Read ``<sequence_root>/gt/gt.txt`` and return one ``DetectionResult``
    per frame, aligned to the 0-indexed first-N-frames convention used by
    the encode pipeline.

    Parameters
    ----------
    sequence_root : path that contains ``gt/gt.txt`` and ``img1/``. For
        MOT17 this is e.g. ``~/datasets/MOT17/train/MOT17-04-DPM``.
    n_frames : number of frames evaluated by the pipeline (50 for our
        pilots). Frames not in the first ``n_frames`` are dropped so the
        list aligns 1-to-1 with the predictions.
    min_visibility : drop boxes whose ``visibility`` field is below this.
        ``0.0`` keeps everything; ``0.3`` is a common MOT-evaluation
        choice that excludes heavily-occluded annotations.
    require_active : if True (default), drop ``conf == 0`` rows (the
        "ignore" regions).
    coco_class : the class label to assign to each retained box. Defaults
        to ``0`` so the result is interchangeable with COCO-style YOLO
        detections.
    """
    from phase2.evaluation.task_accuracy import DetectionResult

    gt_path = (sequence_root / "gt" / "gt.txt").expanduser()
    if not gt_path.exists():
        raise FileNotFoundError(
            f"MOT17 ground-truth file not found: {gt_path}\n"
            f"Set --gt-source pseudo to fall back to YOLO-on-uncompressed."
        )

    by_frame: dict[int, List[_GTBox]] = {}
    n_total = n_kept = 0
    with gt_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            n_total += 1
            box = _parse_gt_line(line)
            if box is None:
                continue
            if require_active and box.conf == 0:
                continue
            if box.cls != MOT17_PEDESTRIAN_CLASS:
                continue
            if box.visibility < min_visibility:
                continue
            by_frame.setdefault(box.frame_id, []).append(box)
            n_kept += 1

    out: List[DetectionResult] = []
    for fi_zero in range(n_frames):
        frame_id = fi_zero + 1  # MOT17 is 1-indexed
        boxes = by_frame.get(frame_id, [])
        out.append(DetectionResult(
            frame_idx=fi_zero,
            n_detections=len(boxes),
            boxes=[(b.x, b.y, b.x + b.w, b.y + b.h) for b in boxes],
            scores=[1.0] * len(boxes),
            classes=[coco_class] * len(boxes),
        ))

    logger.info(
        "MOT17 GT loaded: %s — %d/%d rows kept across %d frames "
        "(avg %.1f boxes/frame, min_vis=%.2f, require_active=%s)",
        gt_path, n_kept, n_total, n_frames,
        sum(d.n_detections for d in out) / max(n_frames, 1),
        min_visibility, require_active,
    )
    return out


def derive_sequence_root_from_frames_dir(frames_dir: Path) -> Path:
    """``<root>/img1`` → ``<root>``. Used to locate the sibling ``gt/``."""
    p = Path(frames_dir).expanduser().resolve()
    if p.name == "img1":
        return p.parent
    return p
