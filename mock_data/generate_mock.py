"""Generate synthetic test data for pipeline validation.

Creates a short synthetic video (or PNG frames as fallback) with
moving colored rectangles simulating tracked objects.

Usage:
    # Generate video (auto-fallbacks to .avi or PNG frames on Linux server)
    python mock_data/generate_mock.py

    # Force PNG frames output (most reliable on headless Linux)
    python mock_data/generate_mock.py --frames-only

    # Custom output
    python mock_data/generate_mock.py --output mock_data/test.mp4 --n-frames 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _build_frame(
    t: int,
    objects: list,
    width: int,
    height: int,
) -> np.ndarray:
    """Render one frame with moving objects."""
    frame = np.full((height, width, 3), 40, dtype=np.uint8)

    # Checkerboard background
    for y in range(0, height, 64):
        for x in range(0, width, 64):
            shade = 30 + ((x // 64 + y // 64) % 2) * 15
            frame[y:min(y+64, height), x:min(x+64, width)] = shade

    for obj in objects:
        cx, cy, bw, bh, vx, vy, color = obj

        # Move with small noise
        cx += vx + np.random.normal(0, 0.5)
        cy += vy + np.random.normal(0, 0.3)

        # Bounce off edges
        if cx - bw / 2 < 0 or cx + bw / 2 > width:
            vx = -vx
        if cy - bh / 2 < 0 or cy + bh / 2 > height:
            vy = -vy

        cx = float(np.clip(cx, bw / 2, width - bw / 2))
        cy = float(np.clip(cy, bh / 2, height - bh / 2))

        obj[0], obj[1], obj[4], obj[5] = cx, cy, vx, vy

        x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
        x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(
            frame, f"obj{objects.index(obj)}",
            (x1 + 4, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
        )

    cv2.putText(
        frame, f"Frame {t:03d}",
        (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2,
    )
    return frame


def _init_objects(n: int, width: int, height: int) -> list:
    np.random.seed(42)
    objects = []
    for _ in range(n):
        cx = float(np.random.randint(200, width - 200))
        cy = float(np.random.randint(200, height - 200))
        bw = int(np.random.randint(80, 200))
        bh = int(np.random.randint(100, 250))
        vx = float(np.random.uniform(-4, 4))
        vy = float(np.random.uniform(-3, 3))
        color = tuple(int(c) for c in np.random.randint(80, 240, 3))
        objects.append([cx, cy, bw, bh, vx, vy, color])
    return objects


def generate_png_frames(
    output_dir: Path,
    n_frames: int = 30,
    width: int = 1920,
    height: int = 1080,
    n_objects: int = 3,
) -> Path:
    """Save individual PNG frames — always works on any server."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    objects = _init_objects(n_objects, width, height)

    for t in range(n_frames):
        frame = _build_frame(t, objects, width, height)
        cv2.imwrite(str(output_dir / f"frame_{t:06d}.png"), frame)

    print(f"Created {n_frames} PNG frames in: {output_dir}")
    return output_dir


def generate_video(
    output_path: Path,
    n_frames: int = 30,
    width: int = 1920,
    height: int = 1080,
    fps: float = 30.0,
    n_objects: int = 3,
) -> Path:
    """Try to create a video file, fallback to PNG frames if codec unavailable."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try mp4v first, then XVID, then fallback to PNG frames
    attempts = [
        (output_path, cv2.VideoWriter_fourcc(*"mp4v")),
        (output_path.with_suffix(".avi"), cv2.VideoWriter_fourcc(*"XVID")),
        (output_path.with_suffix(".avi"), cv2.VideoWriter_fourcc(*"MJPG")),
    ]

    writer = None
    chosen_path = None
    for path, fourcc in attempts:
        w = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if w.isOpened():
            writer = w
            chosen_path = path
            break
        w.release()

    if writer is None or not writer.isOpened():
        print("WARNING: No video codec available — falling back to PNG frames.")
        frames_dir = output_path.parent / (output_path.stem + "_frames")
        return generate_png_frames(frames_dir, n_frames, width, height, n_objects)

    objects = _init_objects(n_objects, width, height)
    for t in range(n_frames):
        frame = _build_frame(t, objects, width, height)
        writer.write(frame)
    writer.release()

    # Verify file was actually written (mp4v can silently fail)
    if not chosen_path.exists() or chosen_path.stat().st_size < 1000:
        chosen_path.unlink(missing_ok=True)
        print("WARNING: Video file empty — falling back to PNG frames.")
        frames_dir = output_path.parent / (output_path.stem + "_frames")
        return generate_png_frames(frames_dir, n_frames, width, height, n_objects)

    print(f"Created video: {chosen_path} ({n_frames} frames, {width}x{height})")
    return chosen_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mock data for IPF testing")
    parser.add_argument("--output", default="mock_data/test.mp4", help="Output video path")
    parser.add_argument("--n-frames", type=int, default=30, help="Number of frames")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--n-objects", type=int, default=3)
    parser.add_argument(
        "--frames-only", action="store_true",
        help="Skip video, always save PNG frames (most reliable on servers)",
    )
    args = parser.parse_args()

    if args.frames_only:
        frames_dir = Path(args.output).parent / (Path(args.output).stem + "_frames")
        result = generate_png_frames(
            frames_dir, args.n_frames, args.width, args.height, args.n_objects
        )
    else:
        result = generate_video(
            Path(args.output), args.n_frames, args.width, args.height,
            30.0, args.n_objects,
        )

    print(f"\nUse this path as --video input:\n  {result}")
