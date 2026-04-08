"""CLI entry point for multi-method comparison.

Usage:
    ipf-compare run --config configs/default.yaml --video path/to/video.mp4 --run-id comp_001
    ipf-compare run --config configs/default.yaml --video path/to/video.mp4 --methods M0 M1 M4 M5
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(name="ipf-compare", help="Multi-method QP comparison pipeline")


@app.command()
def run(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config"),
    video: Optional[str] = typer.Option(None, "--video", "-v", help="Input video path"),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Run identifier"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Output directory"),
    max_frames: Optional[int] = typer.Option(None, "--max-frames", "-n", help="Limit frames"),
    device: Optional[str] = typer.Option(None, "--device", help="CUDA device"),
    methods: Optional[str] = typer.Option(
        "M0,M1,M4,M5,M6,M7,M8",
        "--methods", "-m",
        help="Comma-separated method IDs (e.g. M0,M1,M4,M5,M6,M7,M8)",
    ),
) -> None:
    """Run all specified methods on the same video and compare results."""
    from phase1.core.config import load_config
    from phase1.analysis.comparison_runner import ComparisonPipeline

    cfg = load_config(config)
    if video:
        cfg.video_path = video
    if run_id:
        cfg.run_id = run_id
    if output_dir:
        cfg.output_dir = output_dir
    if max_frames is not None:
        cfg.max_frames = max_frames
    if device:
        cfg.detector.device = device

    if not cfg.video_path:
        typer.echo("Error: No video path specified")
        raise typer.Exit(1)

    method_list = [m.strip() for m in methods.split(",")]
    typer.echo(f"Methods: {method_list}")

    pipeline = ComparisonPipeline(cfg, methods=method_list)
    summary = pipeline.run()

    typer.echo(f"\nComparison complete. Results at: {pipeline.run_dir}")
    typer.echo(f"See: comparison_table.txt and comparison_summary.json")


if __name__ == "__main__":
    app()
