"""CLI entry point for running the IPF Phase 1 pipeline.

Usage:
    ipf-run single --config configs/default.yaml --video path/to/video.mp4
    ipf-run batch  --config configs/default.yaml --video-dir path/to/videos/
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from phase1.core.config import load_config

app = typer.Typer(name="ipf-run", help="IPF Phase 1 Pipeline Runner")


@app.command()
def single(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config"),
    video: Optional[str] = typer.Option(None, "--video", "-v", help="Path to input video (overrides config)"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Output directory (overrides config)"),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Run identifier (overrides config)"),
    max_frames: Optional[int] = typer.Option(None, "--max-frames", "-n", help="Limit frames (overrides config)"),
    device: Optional[str] = typer.Option(None, "--device", help="CUDA device (e.g. cuda:0, cpu)"),
) -> None:
    """Process a single video through the IPF pipeline."""
    cfg = load_config(config)

    if video:
        cfg.video_path = video
    if output_dir:
        cfg.output_dir = output_dir
    if run_id:
        cfg.run_id = run_id
    if max_frames is not None:
        cfg.max_frames = max_frames
    if device:
        cfg.detector.device = device

    if not cfg.video_path:
        typer.echo("Error: No video path specified (use --video or set in config)")
        raise typer.Exit(1)

    from phase1.pipeline.runner import Phase1Pipeline

    pipeline = Phase1Pipeline(cfg)
    metadata = pipeline.run()
    typer.echo(f"\nDone: {metadata.processed_frames} frames in {metadata.total_time_s:.1f}s")
    typer.echo(f"Output: {Path(cfg.output_dir) / cfg.run_id}")


@app.command()
def batch(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config"),
    video_dir: str = typer.Option(..., "--video-dir", help="Directory containing videos"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Output root directory"),
    max_frames: Optional[int] = typer.Option(None, "--max-frames", "-n", help="Limit frames per video"),
    device: Optional[str] = typer.Option(None, "--device", help="CUDA device"),
) -> None:
    """Process all videos in a directory."""
    from phase1.core.constants import SUPPORTED_VIDEO_EXTENSIONS

    cfg = load_config(config)
    if output_dir:
        cfg.output_dir = output_dir
    if max_frames is not None:
        cfg.max_frames = max_frames
    if device:
        cfg.detector.device = device

    video_dir_path = Path(video_dir)
    videos = sorted([
        p for p in video_dir_path.iterdir()
        if p.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    ])

    if not videos:
        typer.echo(f"No videos found in: {video_dir}")
        raise typer.Exit(1)

    typer.echo(f"Found {len(videos)} videos in {video_dir}")

    from phase1.pipeline.runner import Phase1Pipeline

    for i, vpath in enumerate(videos):
        typer.echo(f"\n[{i+1}/{len(videos)}] Processing: {vpath.name}")
        cfg.video_path = str(vpath)
        cfg.run_id = f"batch_{vpath.stem}"

        pipeline = Phase1Pipeline(cfg)
        metadata = pipeline.run()
        typer.echo(f"  → {metadata.processed_frames} frames in {metadata.total_time_s:.1f}s")

    typer.echo(f"\nBatch complete: {len(videos)} videos processed")


if __name__ == "__main__":
    app()
