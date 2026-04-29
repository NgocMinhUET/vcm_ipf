"""Phase 3 Stage C — LiteQP end-to-end orchestrator.

Reads ``phase2/configs/phase3_liteqp.yaml`` and runs:

    1. occlusion_saliency       (per sequence, reuses Stage B output if present)
    2. rate_surrogate           (per sequence, reuses Stage B output if present)
    3. build_liteqp_dataset     (per sequence) → JSONL with δ_a+, δ*, residual
    4. concatenate datasets     → all_liteqp.jsonl
    5. train_liteqp_regressor   (joint, with LOSO CV) → liteqp_mlp.joblib
    6. apply_liteqp_model       (per sequence, --per-qp) → qp_vtm_delta_QP{n}/

Steps 1 and 2 are shared with Stage B and will be **skipped automatically**
if their outputs already exist under ``output_root/saliency`` and
``output_root/rate``.

Run::

    python phase2/scripts/run_phase3_liteqp_pipeline.py \
        --config phase2/configs/phase3_liteqp.yaml

Or to skip earlier stages::

    python phase2/scripts/run_phase3_liteqp_pipeline.py --start-step 3
"""

from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

LOG = logging.getLogger("phase3.liteqp.orchestrator")


def expand(p: Any) -> str:
    return str(Path(str(p)).expanduser())


def run(cmd: List[str], stage: str) -> None:
    LOG.info("[%s] %s", stage, " ".join(shlex.quote(c) for c in cmd))
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise SystemExit(f"[{stage}] failed (exit {res.returncode})")


# ---------------------------------------------------------------------------
# Steps 1 & 2 — reuse Stage B outputs if available
# ---------------------------------------------------------------------------

def step_saliency(cfg, seq, out_root: Path) -> None:
    target = out_root / "saliency" / seq["name"]
    if target.is_dir() and any(target.glob("phi_oracle_*.npy")):
        LOG.info("[saliency:%s] already exists, skipping", seq["name"])
        return
    sal = cfg.get("saliency", {})
    cmd = [
        sys.executable, "-m", "phase2.phase3.occlusion_saliency",
        "--frames-dir", expand(seq["frames_dir"]),
        "--output-dir", str(target),
        "--n-frames", str(seq["n_frames"]),
        "--ctu-size", str(sal.get("ctu_size", 128)),
        "--blur-sigma", str(sal.get("blur_sigma", 8.0)),
        "--batch-size", str(sal.get("batch_size", 32)),
        "--detector", str(sal.get("detector", "yolov8n.pt")),
        "--confidence", str(sal.get("confidence", 0.25)),
        "--device", str(sal.get("device", "cuda:0")),
    ]
    if seq.get("phi_ipf_dir"):
        cmd += ["--phi-ipf-dir", expand(Path(seq["phi_ipf_dir"]) / "fields")]
    run(cmd, f"saliency:{seq['name']}")


def step_rate_surrogate(cfg, seq, out_root: Path) -> None:
    target = out_root / "rate" / seq["name"] / "rate_surrogate.npz"
    if target.exists():
        LOG.info("[rate:%s] already exists, skipping", seq["name"])
        return
    cmd = [
        sys.executable, "-m", "phase2.phase3.rate_surrogate",
        "--frames-dir", expand(seq["frames_dir"]),
        "--pilot-summary", expand(cfg["pilot_summary"]),
        "--sequence", seq["name"],
        "--n-frames", str(seq["n_frames"]),
        "--ctu-size", "128",
        "--output", str(target),
    ]
    run(cmd, f"rate:{seq['name']}")


# ---------------------------------------------------------------------------
# Step 3 — build LiteQP dataset
# ---------------------------------------------------------------------------

def step_build_dataset(cfg, seq, out_root: Path) -> None:
    qp_list = [str(q) for q in cfg["qp_list"]]
    delta_list = [str(d) for d in cfg["delta_list"]]
    teach = cfg.get("teacher", {})
    out_path = out_root / "oracle_liteqp" / f"{seq['name']}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "phase2.phase3.build_liteqp_dataset",
        "--rate-npz", str(out_root / "rate" / seq["name"] / "rate_surrogate.npz"),
        "--saliency-dir", str(out_root / "saliency" / seq["name"]),
        "--frames-dir", expand(seq["frames_dir"]),
        "--sequence", seq["name"],
        "--qp-list", *qp_list,
        "--delta-list", *delta_list,
        "--lambda-task", str(teach.get("lambda_task", 5.0)),
        "--lambda-anchor", str(teach.get("lambda_anchor", 0.6)),
        "--eta", str(teach.get("eta", 0.06)),
        "--xi", str(teach.get("xi", 0.15)),
        "--residual-bound", str(cfg.get("residual_bound", 2.0)),
        "--output", str(out_path),
    ]
    run(cmd, f"build_liteqp:{seq['name']}")


def concatenate_datasets(cfg, out_root: Path) -> Path:
    target = out_root / "oracle_liteqp" / "all_liteqp.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as out:
        for seq in cfg["sequences"]:
            src = out_root / "oracle_liteqp" / f"{seq['name']}.jsonl"
            if not src.exists():
                LOG.warning("Missing dataset slice %s — skipping", src)
                continue
            with open(src, "r", encoding="utf-8") as f:
                for line in f:
                    out.write(line)
    LOG.info("Concatenated LiteQP dataset → %s", target)
    return target


# ---------------------------------------------------------------------------
# Step 5 — train MLP residual
# ---------------------------------------------------------------------------

def step_train(cfg, dataset_path: Path, out_root: Path) -> Path:
    train_cfg = cfg.get("train", {})
    model_path = out_root / "fit" / "liteqp_mlp.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "phase2.phase3.train_liteqp_regressor",
        "--oracle", str(dataset_path),
        "--output", str(model_path),
        "--residual-bound", str(cfg.get("residual_bound", 2.0)),
        "--max-iter", str(train_cfg.get("max_iter", 400)),
        "--cv", train_cfg.get("cv", "both"),
        "--n-bootstrap", str(train_cfg.get("n_bootstrap", 500)),
        "--bootstrap-alpha", str(train_cfg.get("bootstrap_alpha", 0.05)),
    ]
    run(cmd, "train_liteqp")
    return model_path


# ---------------------------------------------------------------------------
# Step 6 — apply model to generate per-QP delta maps
# ---------------------------------------------------------------------------

def step_apply(cfg, seq, model_path: Path, out_root: Path) -> None:
    apply_cfg = cfg.get("apply", {})
    qp_list = [str(q) for q in cfg["qp_list"]]
    nrow = (int(seq.get("height", 1152)) + 127) // 128
    ncol = (int(seq.get("width", 1920)) + 127) // 128
    target_dir = out_root / "learned" / f"liteqp_{seq['name']}" / "M4"
    cmd = [
        sys.executable, "-m", "phase2.phase3.apply_liteqp_model",
        "--mode", apply_cfg.get("mode", "liteqp"),
        "--model", str(model_path),
        "--rate-npz", str(out_root / "rate" / seq["name"] / "rate_surrogate.npz"),
        "--saliency-dir", str(out_root / "saliency" / seq["name"]),
        "--frames-dir", expand(seq["frames_dir"]),
        "--output-dir", str(target_dir),
        "--qp-list", *qp_list,
        "--n-frames", str(seq["n_frames"]),
        "--ctu-rows", str(nrow),
        "--ctu-cols", str(ncol),
        "--residual-bound", str(cfg.get("residual_bound", 2.0)),
        "--per-qp",
    ]
    run(cmd, f"apply_liteqp:{seq['name']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 Stage C (LiteQP) orchestrator")
    parser.add_argument("--config", default="phase2/configs/phase3_liteqp.yaml")
    parser.add_argument("--start-step", type=int, default=1,
                        help="1=saliency, 2=rate, 3=build_liteqp, 4=concat, "
                             "5=train, 6=apply")
    parser.add_argument("--end-step", type=int, default=6)
    parser.add_argument("--only-sequences", nargs="*", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))

    out_root = Path(expand(cfg["output_root"])).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    sequences = cfg.get("sequences", [])
    if args.only_sequences:
        sequences = [s for s in sequences if s["name"] in args.only_sequences]
        if not sequences:
            raise SystemExit(f"No sequences match --only-sequences {args.only_sequences}")
    LOG.info("Processing %d sequences → %s", len(sequences), out_root)

    if args.start_step <= 1 <= args.end_step:
        for seq in sequences:
            step_saliency(cfg, seq, out_root)
    if args.start_step <= 2 <= args.end_step:
        for seq in sequences:
            step_rate_surrogate(cfg, seq, out_root)
    if args.start_step <= 3 <= args.end_step:
        for seq in sequences:
            step_build_dataset(cfg, seq, out_root)

    dataset_path = out_root / "oracle_liteqp" / "all_liteqp.jsonl"
    if args.start_step <= 4 <= args.end_step:
        dataset_path = concatenate_datasets(cfg, out_root)

    model_path = out_root / "fit" / "liteqp_mlp.joblib"
    if args.start_step <= 5 <= args.end_step:
        model_path = step_train(cfg, dataset_path, out_root)

    if args.start_step <= 6 <= args.end_step:
        for seq in sequences:
            step_apply(cfg, seq, model_path, out_root)

    LOG.info("Stage C complete. Encode with phase2/configs/pilot_v4.yaml.")


if __name__ == "__main__":
    main()
