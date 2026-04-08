"""QP map export in multiple formats.

Supports:
  1. VTM-compatible text format (for later CTU-level QP injection)
  2. NumPy binary format (for fast reload)
  3. CSV format (for human inspection)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from phase1.utils.log import get_logger

logger = get_logger("export.qp_exporter")


def export_qp_vtm(
    qp_map: np.ndarray,
    output_path: Path,
    frame_idx: int,
) -> None:
    """Export QP map in VTM-compatible text format.

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
        f.write(f"# frame={frame_idx} rows={n_rows} cols={n_cols}\n")
        for row in range(n_rows):
            line = " ".join(str(int(qp_map[row, col])) for col in range(n_cols))
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
