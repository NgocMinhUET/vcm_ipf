"""Phase 3 Stage 1 — L_p-norm superposition ablation (Level 4 of §2).

Sweeps the superposition exponent p ∈ {1, 2, 4, ∞} across the MOT17 pilot
sequences using the Phase 1 pipeline, emitting per-CTU DELTA QP maps
(Q_base-agnostic, ready for Phase 2).

Design:
- The Phase 1 pipeline is invoked once per (p, sequence) pair.
- Each run writes to ``~/Minh/ipf/phase1_outputs_v2/ipf_lp_p{p}_{sequence}/``.
- Phase 2 configures its ``phase1_output_dir`` per p-value and picks up
  ``qp_vtm_delta/`` instead of ``qp_vtm/``.

This script is the minimal reproducible unit for Research Protocol §7.1 Stage 1.

Usage (run from the ``phase1/`` repo root on the server):

    PYTHONPATH=src python scripts/run_lp_ablation.py \
        --data-root ~/Minh/ipf/datasets/MOT17 \
        --output-dir ~/Minh/ipf/phase1_outputs_v2 \
        --max-frames 50 \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# Allow invocation from anywhere inside the repo.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phase1.core.config import load_config  # noqa: E402
from phase1.pipeline.runner import Phase1Pipeline  # noqa: E402


# MOT17 pilot sequences — matches Phase 1b completed set (§2 of PROJECT_STATE.md).
DEFAULT_SEQUENCES: List[str] = [
    "MOT17-02-DPM",
    "MOT17-04-DPM",
    "MOT17-09-DPM",
]

# p values to sweep. Keep as strings so 'inf' round-trips through the CLI.
DEFAULT_PS: List[str] = ["1", "2", "4", "inf"]


def _parse_p(p_str: str) -> float:
    if p_str.lower() in ("inf", "infinity", "max"):
        return math.inf
    return float(p_str)


def _p_tag(p: float) -> str:
    """Filename-safe representation of p."""
    if math.isinf(p):
        return "inf"
    if p == int(p):
        return str(int(p))
    return str(p).replace(".", "p")


def _resolve_video(data_root: Path, sequence: str) -> Path:
    """Find the video/frame directory for a MOT17 sequence."""
    mp4 = data_root / f"{sequence}.mp4"
    if mp4.is_file():
        return mp4
    # MOT17 extraction layout: <root>/MOT17/train/<seq>/img1/
    frame_dir = data_root / "MOT17" / "train" / sequence / "img1"
    if frame_dir.is_dir():
        return frame_dir
    raise FileNotFoundError(
        f"Neither {mp4} nor {frame_dir} found for sequence {sequence}"
    )


def run_one(
    base_config_path: Path,
    p: float,
    sequence: str,
    data_root: Path,
    output_dir: Path,
    max_frames: Optional[int],
    device: Optional[str],
) -> None:
    cfg = load_config(base_config_path)

    # Core academic override: superposition aggregator and its exponent.
    cfg.field.superposition = "lp"
    cfg.field.p_norm = p

    # ---------------------------------------------------------------
    # Directory layout contract with Phase 2 _find_qp_maps():
    #
    #   Phase 2 looks for:
    #     {phase1_output_dir}/{prefix}{sequence}/M4/qp_vtm_delta/
    #
    #   We achieve this by setting:
    #     output_dir = base_output_dir / f"ipf_lp_p{p}_{sequence}"
    #     run_id     = "M4"
    #
    #   Phase 1 runner writes to: output_dir / run_id / qp_vtm_delta/
    #   i.e.  base_output_dir / ipf_lp_p{p}_{sequence} / M4 / qp_vtm_delta/
    #
    # Phase 2 smoke.yaml uses:
    #   phase1_output_dir = ~/Minh/ipf/phase1_outputs_v2
    #   phase1_run_prefix = ipf_lp_pinf_         (for p=inf)
    # so it searches:
    #   phase1_outputs_v2 / ipf_lp_pinf_MOT17-04-DPM / M4 / qp_vtm_delta/
    # which matches exactly.
    # ---------------------------------------------------------------
    seq_run_dir = output_dir / f"ipf_lp_p{_p_tag(p)}_{sequence}"
    cfg.output_dir = str(seq_run_dir)
    cfg.run_id = "M4"

    cfg.video_path = str(_resolve_video(data_root, sequence))
    if max_frames is not None:
        cfg.max_frames = max_frames
    if device is not None:
        cfg.detector.device = device

    # Phase 3 demands the dQP export.
    cfg.output.save_qp_delta_vtm = True

    print(
        f"\n[{time.strftime('%H:%M:%S')}] --- Running p={_p_tag(p)} on {sequence} ---",
        flush=True,
    )
    print(f"    run_id   = {cfg.run_id}")
    print(f"    video    = {cfg.video_path}")
    print(f"    output   = {Path(cfg.output_dir) / cfg.run_id}")
    print(f"    dQP dir  = {Path(cfg.output_dir) / cfg.run_id}/qp_vtm_delta/  (Phase 2 path)")

    pipeline = Phase1Pipeline(cfg)
    meta = pipeline.run()
    print(
        f"    DONE: {meta.processed_frames} frames in {meta.total_time_s:.1f}s "
        f"(avg {meta.avg_frame_time_ms:.1f} ms/frame)",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3 L_p-norm superposition ablation launcher",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "ipf_lp.yaml"),
        help="Base config path (defaults to configs/ipf_lp.yaml).",
    )
    parser.add_argument(
        "--data-root",
        default="~/Minh/ipf/datasets/MOT17",
        help="MOT17 dataset root directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="~/Minh/ipf/phase1_outputs_v2",
        help="Root output directory (Phase 3 uses a separate tree from v1).",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=DEFAULT_SEQUENCES,
        help="Sequence IDs to process.",
    )
    parser.add_argument(
        "--p-values",
        nargs="+",
        default=DEFAULT_PS,
        help="p exponents (use 'inf' for hard max).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=50,
        help="Frames per sequence (Phase 3 pilot: 50).",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Detector device (e.g. cuda:0 | cpu).",
    )
    args = parser.parse_args()

    base_config = Path(args.config).expanduser().resolve()
    if not base_config.is_file():
        raise FileNotFoundError(f"Config file not found: {base_config}")

    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ps = [_parse_p(p) for p in args.p_values]

    total = len(ps) * len(args.sequences)
    print("=" * 70)
    print("Phase 3 Stage 1 — L_p-norm Ablation")
    print("=" * 70)
    print(f"Config:     {base_config}")
    print(f"Data root:  {data_root}")
    print(f"Output:     {output_dir}")
    print(f"Sequences:  {args.sequences}")
    print(f"p values:   {[_p_tag(p) for p in ps]}")
    print(f"Max frames: {args.max_frames}")
    print(f"Device:     {args.device}")
    print(f"Total runs: {total}")
    print("=" * 70)

    t0 = time.time()
    failed: List[Tuple[str, float, str]] = []
    done = 0
    for p in ps:
        for seq in args.sequences:
            try:
                run_one(
                    base_config_path=base_config,
                    p=p,
                    sequence=seq,
                    data_root=data_root,
                    output_dir=output_dir,
                    max_frames=args.max_frames,
                    device=args.device,
                )
                done += 1
            except Exception as exc:  # noqa: BLE001
                failed.append((seq, p, str(exc)))
                print(f"    FAILED: {exc}", flush=True)

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"Ablation complete: {done}/{total} runs OK in {elapsed/60:.1f} min")
    if failed:
        print(f"Failures: {len(failed)}")
        for seq, p, err in failed:
            print(f"  - p={_p_tag(p)}, seq={seq}: {err}")
    print("=" * 70)


if __name__ == "__main__":
    main()
