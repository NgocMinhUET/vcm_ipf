"""Phase 3 Stage B end-to-end orchestrator.

Reads ``phase2/configs/phase3_fit.yaml`` and runs:

    1. occlusion_saliency      (per sequence)
    2. rate_surrogate          (per sequence)
    3. build_oracle            (per sequence)
    4. fit_parametric          (global)
    5. train_dnn               (only if pre-registered criteria fail)
    6. apply_formula           (per sequence)

Each step is a separate ``subprocess.run`` so partial reruns are easy:
just delete the affected sub-directory under ``output_root`` and call
the script with ``--start-step <k>``.

Run with::

    python phase2/scripts/run_phase3_pipeline.py \
        --config phase2/configs/phase3_fit.yaml

Or to skip a step::

    python phase2/scripts/run_phase3_pipeline.py --start-step 3
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

LOG = logging.getLogger("phase3.orchestrator")


def expand(p: Any) -> str:
    return str(Path(str(p)).expanduser())


def run(cmd: List[str], stage: str) -> None:
    LOG.info("[%s] %s", stage, " ".join(shlex.quote(c) for c in cmd))
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise SystemExit(f"[{stage}] failed (exit {res.returncode})")


def step_saliency(cfg: Dict[str, Any], seq: Dict[str, Any], out_root: Path) -> None:
    sal = cfg.get("saliency", {})
    cmd = [
        sys.executable, "-m", "phase2.phase3.occlusion_saliency",
        "--frames-dir", expand(seq["frames_dir"]),
        "--output-dir", str(out_root / "saliency" / seq["name"]),
        "--phi-ipf-dir", expand(Path(seq["phi_ipf_dir"]) / "fields"),
        "--n-frames", str(seq["n_frames"]),
        "--ctu-size", str(sal.get("ctu_size", 128)),
        "--blur-sigma", str(sal.get("blur_sigma", 8.0)),
        "--batch-size", str(sal.get("batch_size", 32)),
        "--detector", str(sal.get("detector", "yolov8n.pt")),
        "--confidence", str(sal.get("confidence", 0.25)),
        "--device", str(sal.get("device", "cuda:0")),
    ]
    run(cmd, f"saliency:{seq['name']}")


def step_rate_surrogate(cfg: Dict[str, Any], seq: Dict[str, Any], out_root: Path) -> None:
    cmd = [
        sys.executable, "-m", "phase2.phase3.rate_surrogate",
        "--frames-dir", expand(seq["frames_dir"]),
        "--pilot-summary", expand(cfg["pilot_summary"]),
        "--sequence", seq["name"],
        "--n-frames", str(seq["n_frames"]),
        "--ctu-size", "128",
        "--output", str(out_root / "rate" / seq["name"] / "rate_surrogate.npz"),
    ]
    run(cmd, f"rate:{seq['name']}")


def step_build_oracle(cfg: Dict[str, Any], seq: Dict[str, Any], out_root: Path) -> None:
    qp_list = [str(q) for q in cfg["qp_list"]]
    delta_list = [str(d) for d in cfg["delta_list"]]
    oracle_cfg = cfg.get("oracle", {})
    out_path = out_root / "oracle" / f"{seq['name']}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "phase2.phase3.build_oracle",
        "--rate-npz", str(out_root / "rate" / seq["name"] / "rate_surrogate.npz"),
        "--saliency-dir", str(out_root / "saliency" / seq["name"]),
        "--phi-ipf-dir", expand(Path(seq["phi_ipf_dir"]) / "fields"),
        "--sequence", seq["name"],
        "--qp-list", *qp_list,
        "--delta-list", *delta_list,
        "--samples-per-cell", str(oracle_cfg.get("samples_per_cell", 200)),
        "--fps", str(seq.get("fps", 30)),
        "--eta", str(oracle_cfg.get("eta", 0.06)),
        "--xi", str(oracle_cfg.get("xi", 0.15)),
        "--output", str(out_path),
    ]
    run(cmd, f"oracle:{seq['name']}")


def concatenate_oracles(cfg: Dict[str, Any], out_root: Path) -> Path:
    target = out_root / "oracle" / "all.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as out:
        for seq in cfg["sequences"]:
            src = out_root / "oracle" / f"{seq['name']}.jsonl"
            if not src.exists():
                LOG.warning("Missing oracle slice %s — skipping", src)
                continue
            with open(src, "r", encoding="utf-8") as f:
                for line in f:
                    out.write(line)
    LOG.info("Concatenated oracle written to %s", target)
    return target


def step_fit(cfg: Dict[str, Any], oracle_path: Path, out_root: Path) -> Path:
    fit_cfg = cfg.get("fit", {})
    levels = fit_cfg.get("levels", ["L1-linear", "L2-asym", "L3-Qadapt", "L4-Lp", "L5-softmax"])
    fit_out = out_root / "fit" / "fit_results.json"
    fit_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "phase2.phase3.fit_parametric",
        "--oracle", str(oracle_path),
        "--out", str(fit_out),
        "--lambda-task", str(cfg.get("lambda_task", 7400.0)),
        "--n-bootstrap", str(fit_cfg.get("n_bootstrap", 200)),
        "--levels", *levels,
    ]
    if not fit_cfg.get("enable_loso", True):
        cmd.append("--no-cv")
    run(cmd, "fit_parametric")
    return fit_out


def parametric_winner(fit_out: Path) -> bool:
    with open(fit_out, "r", encoding="utf-8") as f:
        data = json.load(f)
    for r in data.get("results", []):
        ci_hi = (r.get("extra") or {}).get("val_bd_rate_task_ci_hi", 1.0)
        if r.get("val_bd_rate_task", 1.0) < 0.0 and ci_hi < 0.0:
            LOG.info("Parametric winner: %s (bd=%.4f, ci_hi=%.4f)",
                     r["level"], r["val_bd_rate_task"], ci_hi)
            return True
    LOG.warning("No parametric Level satisfies the pre-registered criteria")
    return False


def step_dnn(cfg: Dict[str, Any], oracle_path: Path, out_root: Path) -> None:
    dnn_cfg = cfg.get("dnn_fallback", {})
    cmd = [
        sys.executable, "-m", "phase2.phase3.train_dnn",
        "--oracle", str(oracle_path),
        "--output", str(out_root / "fit" / "dnn_l6.json"),
        "--lambda-task", str(cfg.get("lambda_task", 7400.0)),
        "--lambda-smooth", str(dnn_cfg.get("lambda_smooth", 0.05)),
        "--lambda-reg", str(dnn_cfg.get("lambda_reg", 1e-4)),
        "--restarts", str(dnn_cfg.get("restarts", 4)),
    ]
    run(cmd, "dnn_l6")


def step_apply(cfg: Dict[str, Any], seq: Dict[str, Any], fit_out: Path,
                out_root: Path) -> None:
    apply_cfg = cfg.get("apply", {})
    qp_list = [str(q) for q in cfg["qp_list"]]
    nrow = (int(seq["height"]) + 127) // 128
    ncol = (int(seq["width"]) + 127) // 128
    cmd = [
        sys.executable, "-m", "phase2.phase3.apply_formula",
        "--theta", str(fit_out),
        "--phi-ipf-dir", expand(Path(seq["phi_ipf_dir"]) / "fields"),
        "--output-dir", str(out_root / "learned" / f"learned_{seq['name']}" / "M4"),
        "--qp-list", *qp_list,
        "--n-frames", str(seq["n_frames"]),
        "--ctu-rows", str(nrow),
        "--ctu-cols", str(ncol),
        "--delta-min", str(apply_cfg.get("delta_min", -8)),
        "--delta-max", str(apply_cfg.get("delta_max", 4)),
    ]
    if apply_cfg.get("per_qp", False):
        cmd.append("--per-qp")
    run(cmd, f"apply:{seq['name']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 Stage B orchestrator")
    parser.add_argument("--config", default="phase2/configs/phase3_fit.yaml")
    parser.add_argument("--start-step", type=int, default=1,
                        help="1=saliency, 2=rate, 3=oracle, 4=fit, 5=dnn, 6=apply")
    parser.add_argument("--end-step", type=int, default=6)
    parser.add_argument("--only-sequences", nargs="*", default=None,
                        help="Restrict to a subset of sequences")
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
            step_build_oracle(cfg, seq, out_root)

    oracle_path = out_root / "oracle" / "all.jsonl"
    if args.start_step <= 4 <= args.end_step:
        oracle_path = concatenate_oracles(cfg, out_root)
        fit_out = step_fit(cfg, oracle_path, out_root)
    else:
        fit_out = out_root / "fit" / "fit_results.json"

    if args.start_step <= 5 <= args.end_step:
        if cfg.get("dnn_fallback", {}).get("enable_if_no_winner", True):
            if not parametric_winner(fit_out):
                step_dnn(cfg, oracle_path, out_root)
        else:
            LOG.info("DNN fallback disabled in config")

    if args.start_step <= 6 <= args.end_step:
        for seq in sequences:
            step_apply(cfg, seq, fit_out, out_root)

    LOG.info("Stage B complete. Encode with phase2/configs/pilot_v3.yaml.")


if __name__ == "__main__":
    main()
