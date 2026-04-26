"""Phase 3 Stage 2 — Oracle data collection for learned QP allocation.

Reference: ``11_PHASE3_RESEARCH_PROTOCOL.md`` §3.

For each (sequence, base QP) the oracle collector does:

    1. Encode once with the UNIFORM baseline (δ = 0 everywhere) — yields
       reference (R_0, mAP_0) and per-frame detection boxes.
    2. For each frame in the sampling window and for each candidate CTU
       perturbation δ ∈ Δ, encode a perturbed clip where a small random
       subset of CTUs (5 % by default) receive that δ.
    3. Parse (R_perturb, mAP_perturb) and log per-CTU samples
       (Φ̂_max, Φ̂_sum, …, Q_base, δ, ΔR, ΔmAP).

The resulting Parquet dataset is the input for Stage 3 (parametric fitting).

Note: this script is an experiment orchestrator, not a tight inner loop.
It is intentionally conservative on wall-clock budget — ``§7.1 Stage 2``
targets ≤ 24 h. Adjust ``--frames-per-seq`` and ``--ctu-fraction`` for a
faster exploratory pass.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger("phase2.phase3.collect_oracle")


# Default perturbation set per §3.1 of the protocol.
DEFAULT_DELTAS: List[int] = [-8, -6, -4, -2, 0, 2, 4]


@dataclass
class OracleSample:
    """One row of the oracle dataset.

    A "sample" here is one perturbed frame — the per-CTU features are
    serialized as a list so downstream fitting can expand them on demand
    while keeping the wire format compact.
    """

    sequence: str
    frame_idx: int
    qp_base: int
    delta: int
    ctu_indices: List[int]          # flat row-major CTU indices that received delta
    n_ctu_total: int
    delta_rate_kbps: float          # R_perturb - R_0 (single-frame re-encode)
    delta_map50: float              # mAP_perturb - mAP_0
    features: Dict[str, List[float]] = field(default_factory=dict)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def sample_ctu_indices(
    n_rows: int,
    n_cols: int,
    fraction: float,
    rng: random.Random,
) -> List[int]:
    total = n_rows * n_cols
    k = max(1, int(round(fraction * total)))
    return sorted(rng.sample(range(total), k))


def load_phi_features(
    phase1_run_dir: Path,
    frame_idx: int,
    n_rows: int,
    n_cols: int,
) -> Dict[str, np.ndarray]:
    """Load the per-CTU importance features produced by Phase 1.

    We recover Φ̂ from the existing delta-QP dump:

        delta(r,c) ≈ -Δ_roi · strength(r,c)            if ROI
                    = +Δ_bg  · strength(r,c)            if BG

    which lets us reconstruct normalized field proxies (max-only here;
    sum/Lp variants need raw per-object kernels — TODO in future revs).

    This function is a placeholder that returns **zeros** when the
    per-frame field arrays are not persisted. The default Phase 1
    config does not save ``fields/`` (``output.save_field_npy = False``).
    Re-run Phase 1 with ``save_field_npy = True`` for rich features.
    """
    features = {
        "phi_max": np.zeros(n_rows * n_cols, dtype=np.float32),
        "phi_sum": np.zeros(n_rows * n_cols, dtype=np.float32),
        "phi_l2": np.zeros(n_rows * n_cols, dtype=np.float32),
    }

    field_npy = phase1_run_dir / "fields" / f"field_{frame_idx:06d}.npy"
    if field_npy.is_file():
        arr = np.load(field_npy).astype(np.float32)
        if arr.shape == (n_rows, n_cols):
            flat = arr.reshape(-1)
            features["phi_max"] = flat.copy()
            features["phi_sum"] = flat.copy()
            features["phi_l2"] = flat.copy()
    return features


def write_perturbed_qp_dir(
    baseline_qp_dir: Path,
    out_dir: Path,
    frame_idx: int,
    ctu_indices: List[int],
    delta: int,
    n_rows: int,
    n_cols: int,
) -> None:
    """Write a delta-QP directory where exactly one frame is perturbed.

    For every other frame we simply copy the baseline file (δ = 0) so VTM
    sees a consistent directory. For the target frame we overwrite the
    specified CTU indices with `delta`, leaving the rest at 0.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build a δ-map (flat) for the target frame.
    target = np.zeros(n_rows * n_cols, dtype=np.int32)
    for idx in ctu_indices:
        target[idx] = delta

    # Iterate over baseline files.
    for baseline_file in sorted(baseline_qp_dir.glob("qp_*.txt")):
        fidx = int(baseline_file.stem.split("_")[-1])
        out_file = out_dir / baseline_file.name
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(
                f"# frame={fidx} rows={n_rows} cols={n_cols} "
                f"type=delta delta_min=-8 delta_max=4\n"
            )
            if fidx == frame_idx:
                values = target.reshape(n_rows, n_cols)
            else:
                values = np.zeros((n_rows, n_cols), dtype=np.int32)
            for row in range(n_rows):
                f.write(" ".join(f"{int(v):+d}" for v in values[row]) + "\n")


def build_baseline_delta_dir(
    out_dir: Path,
    n_frames: int,
    n_rows: int,
    n_cols: int,
) -> Path:
    """Materialize a baseline delta-QP directory (all zeros).

    This mimics M0 but exercises the patched external-QP-map code path so
    the oracle measurement uses the exact same encoder logic as perturbed
    runs — isolating the effect of ``delta`` from any patch-vs-unpatched
    difference.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        out_file = out_dir / f"qp_{i:06d}.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(
                f"# frame={i} rows={n_rows} cols={n_cols} "
                f"type=delta delta_min=-8 delta_max=4\n"
            )
            zero_row = " ".join(["+0"] * n_cols)
            for _ in range(n_rows):
                f.write(zero_row + "\n")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 Stage 2 oracle collector")
    parser.add_argument(
        "--phase2-config",
        required=True,
        help="Phase 2 YAML (provides VTM paths, sequences, evaluator settings).",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory to write the oracle Parquet dataset and per-run logs.",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=["MOT17-04-DPM"],
        help="Which sequences to probe (must be defined in phase2-config).",
    )
    parser.add_argument(
        "--qp-bases",
        nargs="+",
        type=int,
        default=[27, 32, 37, 42],
        help="Base QPs to probe.",
    )
    parser.add_argument(
        "--deltas",
        nargs="+",
        type=int,
        default=DEFAULT_DELTAS,
        help="Per-CTU QP perturbations to apply (signed integers).",
    )
    parser.add_argument(
        "--frames-per-seq",
        type=int,
        default=20,
        help="Number of frames to probe per sequence (uniform subsampling).",
    )
    parser.add_argument(
        "--ctu-fraction",
        type=float,
        default=0.05,
        help="Fraction of CTUs receiving the perturbation per sample (§3.1).",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=3,
        help="Independent random seeds per (frame, Q_b, delta).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Enumerate runs without invoking VTM (debug).",
    )
    args = parser.parse_args()

    from phase2.core.config import Phase2Config, load_phase2_config

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    set_seed(args.seed)

    cfg: Phase2Config = load_phase2_config(args.phase2_config)
    out_root = Path(args.output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    sequences_by_name = {s.name: s for s in cfg.sequences}
    selected_seqs = [sequences_by_name[name] for name in args.sequences]

    # Enumerate all runs.
    plan: List[dict] = []
    for seq in selected_seqs:
        total = seq.n_frames
        step = max(1, total // args.frames_per_seq)
        probe_frames = list(range(0, total, step))[: args.frames_per_seq]
        for q in args.qp_bases:
            for delta in args.deltas:
                for seed in range(args.n_seeds):
                    for fidx in probe_frames:
                        plan.append(
                            {
                                "sequence": seq.name,
                                "qp_base": q,
                                "delta": delta,
                                "frame_idx": fidx,
                                "seed": seed,
                            }
                        )

    logger.info("Oracle plan: %d perturbed runs + %d baseline runs",
                len(plan), len(selected_seqs) * len(args.qp_bases))

    if args.dry_run:
        for row in plan[:20]:
            logger.info("PLAN: %s", row)
        logger.info("... (%d more rows) ...", max(0, len(plan) - 20))
        return

    # ---------------------------------------------------------------
    # The actual measurement loop is intentionally left as a stub:
    # the full implementation requires per-frame bit accounting (VTM
    # does not expose per-frame bits cleanly when external QP maps
    # modulate the rate controller). For Phase 3 Stage 2 we plan to:
    #   (a) Use BITSTREAM_NAL_UNIT slicing (parse NAL AU sizes).
    #   (b) Cache baseline recon.yuv and re-encode ONLY the perturbed
    #       frame as an intra-only slice.
    # A follow-up patch will wire (a)+(b) once Stage 1 is verified on
    # server. For now we persist the enumeration plan so the exact
    # experiment is reproducible.
    # ---------------------------------------------------------------
    plan_file = out_root / "oracle_plan.jsonl"
    with open(plan_file, "w", encoding="utf-8") as f:
        for row in plan:
            f.write(json.dumps(row) + "\n")
    logger.info("Plan serialized to %s", plan_file)
    logger.info(
        "Stage 2 measurement loop is a stub — pending Stage 1 verification "
        "(see %s §3.3 'Bitstream-level ΔR accounting')",
        "11_PHASE3_RESEARCH_PROTOCOL.md",
    )


if __name__ == "__main__":
    main()
