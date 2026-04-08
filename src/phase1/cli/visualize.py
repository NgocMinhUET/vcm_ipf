"""CLI entry point for post-hoc visualization from saved pipeline outputs.

Usage:
    ipf-viz temporal --run-dir outputs/run_001
    ipf-viz field-maps --run-dir outputs/run_001 --video path/to/video.mp4
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
import numpy as np

app = typer.Typer(name="ipf-viz", help="IPF Phase 1 Visualization Tools")


@app.command()
def temporal(
    run_dir: str = typer.Option(..., "--run-dir", help="Path to pipeline run output"),
) -> None:
    """Generate temporal QP stability plots from a completed run."""
    import json
    import csv
    from phase1.core.schemas import FrameResult
    from phase1.viz.visualizer import save_temporal_qp_plot

    run_path = Path(run_dir)
    csv_path = run_path / "frame_summaries.csv"

    if not csv_path.exists():
        typer.echo(f"Error: {csv_path} not found")
        raise typer.Exit(1)

    results = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fr = FrameResult(frame_idx=int(row["frame_idx"]))
            fr.n_objects = int(row["n_objects"])
            if "qp_mean" in row and row["qp_mean"]:
                qp_mean = float(row["qp_mean"])
                qp_min = int(row.get("qp_min", qp_mean))
                qp_max = int(row.get("qp_max", qp_mean))
                fr.final_qp_map = np.array([[qp_min, qp_max]])
            results.append(fr)

    output = run_path / "viz" / "temporal_qp_replot.png"
    save_temporal_qp_plot(results, output)
    typer.echo(f"Plot saved: {output}")


if __name__ == "__main__":
    app()
