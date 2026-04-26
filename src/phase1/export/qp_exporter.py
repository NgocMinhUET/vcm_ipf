"""QP map export in multiple formats.

Supports:
  1. Absolute-QP VTM text format (legacy — pilot v1 compatibility).
  2. Delta-QP VTM text format (Phase 3 primary).
  3. NumPy binary and CSV formats for inspection.

The VTM patch reads per-CTU integer QPs. For the delta format we
annotate the file header so the Phase 2 encoder wrapper can detect
the format and compose with the run-time Q_base.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from phase1.utils.log import get_logger

logger = get_logger("export.qp_exporter")


# File header tags — Phase 2 reads these to dispatch on QP-map format.
HEADER_TAG_ABSOLUTE = "type=absolute"
HEADER_TAG_DELTA = "type=delta"


def export_qp_vtm(
    qp_map: np.ndarray,
    output_path: Path,
    frame_idx: int,
) -> None:
    """Export ABSOLUTE-QP map in VTM-compatible text format (legacy).

    Format: one integer QP value per CTU, raster-scan order,
    with header comment for metadata.

    Args:
        qp_map: 2D int array (n_rows, n_cols).
        output_path: Destination file path.
        frame_idx: Frame index for the header.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows, n_cols = qp_map.shape
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(
            f"# frame={frame_idx} rows={n_rows} cols={n_cols} "
            f"{HEADER_TAG_ABSOLUTE}\n"
        )
        for row in range(n_rows):
            line = " ".join(str(int(qp_map[row, col])) for col in range(n_cols))
            f.write(line + "\n")


def export_delta_qp_vtm(
    delta_map: np.ndarray,
    output_path: Path,
    frame_idx: int,
    delta_min: int = -8,
    delta_max: int = 4,
) -> None:
    """Export DELTA-QP (Phase 3 primary format) in VTM-compatible text.

    Values are rounded to signed integers, clipped to [delta_min, delta_max],
    and written in raster-scan order. The Phase 2 encoder wrapper detects
    the `type=delta` header tag and composes the absolute map at run time:

        Q_final(r, c) = clip(Q_base + delta(r, c), 1, 51)

    Args:
        delta_map: 2D float array (n_rows, n_cols) of signed QP offsets.
        output_path: Destination file path.
        frame_idx: Frame index for the header.
        delta_min: Lower clamp (inclusive). Default: -8.
        delta_max: Upper clamp (inclusive). Default: +4.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clipped = np.clip(np.rint(delta_map), delta_min, delta_max).astype(np.int32)
    n_rows, n_cols = clipped.shape

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(
            f"# frame={frame_idx} rows={n_rows} cols={n_cols} "
            f"{HEADER_TAG_DELTA} "
            f"delta_min={delta_min} delta_max={delta_max}\n"
        )
        for row in range(n_rows):
            line = " ".join(f"{clipped[row, col]:+d}" for col in range(n_cols))
            f.write(line + "\n")


def export_qp_csv(
    qp_map: np.ndarray,
    output_path: Path,
) -> None:
    """Export QP map as CSV for inspection."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(output_path), qp_map, fmt="%d", delimiter=",")


def export_qp_npy(
    qp_map: np.ndarray,
    output_path: Path,
) -> None:
    """Export QP map as NumPy binary."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), qp_map)
