"""Run and frame summary export in JSON and CSV formats."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from phase1.core.schemas import FrameResult, RunMetadata
from phase1.utils.log import get_logger

logger = get_logger("export.summary_writer")


def write_run_metadata(metadata: RunMetadata, output_dir: Path) -> None:
    """Write run metadata to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_json(output_dir / "run_metadata.json")
    logger.info("Run metadata saved")


def write_frame_summaries(
    results: list[FrameResult],
    output_dir: Path,
) -> None:
    """Write per-frame summary statistics to CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "frame_summaries.csv"

    if not results:
        return

    rows = [r.summary_dict() for r in results]
    fieldnames = list(rows[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Frame summaries: %d frames → %s", len(results), csv_path.name)


def write_aggregate_summary(
    results: list[FrameResult],
    metadata: RunMetadata,
    output_dir: Path,
) -> None:
    """Write aggregate statistics across all frames."""
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results:
        return

    n_objs = [r.n_objects for r in results]
    times = [r.processing_time_ms for r in results]

    qp_means = []
    for r in results:
        if r.final_qp_map is not None:
            qp_means.append(float(np.mean(r.final_qp_map)))

    summary = {
        "run_id": metadata.run_id,
        "total_frames": len(results),
        "objects_per_frame": {
            "mean": float(np.mean(n_objs)),
            "min": int(np.min(n_objs)),
            "max": int(np.max(n_objs)),
        },
        "frame_time_ms": {
            "mean": float(np.mean(times)),
            "p50": float(np.percentile(times, 50)),
            "p95": float(np.percentile(times, 95)),
        },
        "qp_mean_across_frames": float(np.mean(qp_means)) if qp_means else None,
        "total_time_s": metadata.total_time_s,
    }

    path = output_dir / "aggregate_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Aggregate summary saved")
