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

**Versioning** (new in v2). If the config sets ``version: "v2"`` (or any
non-empty string), the suffix ``_v2`` is appended to:
* the dataset directory ``oracle_liteqp{sfx}/``
* the model bundle    ``fit/liteqp_mlp{sfx}.joblib``
* the per-sequence map directory ``learned/liteqp{sfx}_<seq>/``
This lets multiple LiteQP variants coexist without overwriting each other.
The shared caches (``saliency/`` and ``rate/``) are NOT versioned because
they only depend on raw frames + pilot_v1 M0 anchors.

**Per-sequence teacher overrides** (new in v2). ``teacher:`` provides the
default Lagrangian (λ_task, λ_anchor, η, ξ) used by the teacher. To tune
it per sequence (e.g. larger λ_task on elastic-mAP sequences such as
MOT17-09-DPM) add a ``teacher_overrides`` dict keyed by sequence name —
only the overridden keys need be present, the rest fall back to
``teacher:``.

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
# Versioning + per-sequence teacher overrides
# ---------------------------------------------------------------------------

def version_suffix(cfg: dict) -> str:
    """Return a ``_<version>`` filename suffix, or empty string when unset.

    A leading underscore is added automatically so callers can write
    ``f"liteqp_mlp{sfx}.joblib"`` and get either ``liteqp_mlp.joblib`` or
    ``liteqp_mlp_v2.joblib``.
    """
    raw = str(cfg.get("version", "")).strip()
    if not raw:
        return ""
    if raw.startswith("_"):
        return raw
    return f"_{raw}"


def teacher_for_sequence(cfg: dict, seq_name: str) -> Dict[str, float]:
    """Resolve teacher Lagrangian parameters for a given sequence.

    Looks up ``teacher_overrides[seq_name]`` if present, falling back to the
    global ``teacher:`` block for any missing keys. Returns a dict with
    ``lambda_task / lambda_anchor / eta / xi`` always populated.
    """
    base = dict(cfg.get("teacher", {}) or {})
    base.setdefault("lambda_task",   5.0)
    base.setdefault("lambda_anchor", 0.6)
    base.setdefault("eta",           0.06)
    base.setdefault("xi",            0.15)
    overrides = (cfg.get("teacher_overrides") or {}).get(seq_name, {}) or {}
    base.update({k: float(v) for k, v in overrides.items()})
    return base


# ---------------------------------------------------------------------------
# Action 4 — automatic per-sequence λ_task from M0 anchor elasticity
# ---------------------------------------------------------------------------

def resolve_auto_lambda(cfg: dict, out_root: Path) -> dict:
    """If ``auto_lambda.enabled``, compute λ per sequence from M0 elasticity
    and rewrite ``cfg["teacher_overrides"]`` accordingly.

    Returns the resolved auto-λ payload (also persisted to disk for
    reproducibility) — empty dict if auto-λ is disabled.

    Design choice (PROJECT_STATE §7.12): auto-λ is the SOLE source of
    per-sequence ``lambda_task`` overrides when enabled. Any
    pre-existing ``teacher_overrides[seq]["lambda_task"]`` is overwritten
    so that the user cannot accidentally combine manual + auto tuning
    (which would re-introduce the v5 cherry-picking critique). Other
    teacher keys (``lambda_anchor`` / ``eta`` / ``xi``) in
    ``teacher_overrides`` are left untouched.
    """
    auto_cfg = cfg.get("auto_lambda") or {}
    if not auto_cfg.get("enabled", False):
        return {}

    from phase2.phase3.auto_lambda import compute_lambdas, write_metadata

    src = auto_cfg.get("source")
    if not src:
        raise SystemExit("auto_lambda.enabled is true but no `source` given")
    src_path = Path(expand(src)).resolve()
    if not src_path.exists():
        raise SystemExit(f"auto_lambda.source not found: {src_path}")

    seq_names = [s["name"] for s in cfg.get("sequences", [])]
    qp_range = None
    if auto_cfg.get("qp_min") is not None and auto_cfg.get("qp_max") is not None:
        qp_range = (int(auto_cfg["qp_min"]), int(auto_cfg["qp_max"]))

    LOG.info("[auto_lambda] computing λ from %s (alpha=%s, clip=[%s, %s])",
             src_path,
             auto_cfg.get("alpha", 0.5),
             auto_cfg.get("lambda_min", 3.5),
             auto_cfg.get("lambda_max", 7.0))

    results = compute_lambdas(
        pilot_summary_path=src_path,
        sequences=seq_names,
        base_lambda=float(auto_cfg.get("base_lambda", 5.0)),
        alpha=float(auto_cfg.get("alpha", 0.5)),
        lambda_min=float(auto_cfg.get("lambda_min", 3.5)),
        lambda_max=float(auto_cfg.get("lambda_max", 7.0)),
        slope_mode=str(auto_cfg.get("slope_mode", "median")),
        qp_range=qp_range,
    )

    overrides = cfg.setdefault("teacher_overrides", {}) or {}
    for seq_name, payload in results.items():
        slot = overrides.setdefault(seq_name, {})
        slot["lambda_task"] = float(payload["lambda_task"])
        if payload.get("fallback"):
            LOG.warning("[auto_lambda] %s: FALLBACK to base_lambda=%.2f "
                        "(no anchor data found)",
                        seq_name, payload["lambda_task"])
        elif payload.get("clipped"):
            LOG.info("[auto_lambda] %s: elasticity=%.3f  raw=%.2f  "
                     "lambda_task=%.2f (CLIPPED)",
                     seq_name, payload["elasticity"],
                     payload["raw_lambda"], payload["lambda_task"])
        else:
            LOG.info("[auto_lambda] %s: elasticity=%.3f  "
                     "lambda_task=%.2f",
                     seq_name, payload["elasticity"], payload["lambda_task"])
    cfg["teacher_overrides"] = overrides

    sfx = version_suffix(cfg)
    meta_path = out_root / f"auto_lambda{sfx}.json"
    write_metadata(results, meta_path,
                    cfg_used={"source": str(src_path),
                              **{k: auto_cfg[k] for k in
                                 ("base_lambda", "alpha", "lambda_min",
                                  "lambda_max", "slope_mode") if k in auto_cfg},
                              "qp_range": qp_range,
                              "version": str(cfg.get("version", ""))})
    return results


# ---------------------------------------------------------------------------
# Steps 1 & 2 — reuse Stage B outputs if available
# ---------------------------------------------------------------------------

def step_save_boxes(cfg, seq, out_root: Path) -> None:
    """Persist per-frame YOLO boxes for OG-IPF (PROJECT_STATE §7.20).

    Idempotent: if every ``boxes_<frame>.json`` already exists, skip.
    """
    target = out_root / "saliency" / seq["name"]
    target.mkdir(parents=True, exist_ok=True)
    n = int(seq.get("n_frames", 50))
    needed = [target / f"boxes_{i:06d}.json" for i in range(n)]
    if all(p.exists() for p in needed):
        LOG.info("[boxes:%s] all %d files present, skipping", seq["name"], n)
        return
    sal = cfg.get("saliency", {})
    classes = cfg.get("og", {}).get("detector_classes", [0])
    cmd = [
        sys.executable, "-m", "phase2.phase3.save_boxes",
        "--frames-dir", expand(seq["frames_dir"]),
        "--output-dir", str(target),
        "--n-frames",   str(n),
        "--ctu-size",   str(sal.get("ctu_size", 128)),
        "--detector",   str(sal.get("detector", "yolov8n.pt")),
        "--confidence", str(sal.get("confidence", 0.25)),
        "--device",     str(sal.get("device", "cuda:0")),
        "--classes",    *[str(c) for c in classes],
    ]
    run(cmd, f"save_boxes:{seq['name']}")


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
    sfx = version_suffix(cfg)
    qp_list = [str(q) for q in cfg["qp_list"]]
    delta_list = [str(d) for d in cfg["delta_list"]]
    teach = teacher_for_sequence(cfg, seq["name"])
    LOG.info("[build_liteqp:%s] teacher = lambda_task=%.2f lambda_anchor=%.2f "
             "eta=%.3f xi=%.3f%s",
             seq["name"], teach["lambda_task"], teach["lambda_anchor"],
             teach["eta"], teach["xi"],
             "" if not (cfg.get("teacher_overrides") or {}).get(seq["name"])
             else "  (PER-SEQ OVERRIDE)")
    out_path = out_root / f"oracle_liteqp{sfx}" / f"{seq['name']}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "phase2.phase3.build_liteqp_dataset",
        "--rate-npz", str(out_root / "rate" / seq["name"] / "rate_surrogate.npz"),
        "--saliency-dir", str(out_root / "saliency" / seq["name"]),
        "--frames-dir", expand(seq["frames_dir"]),
        "--sequence", seq["name"],
        "--qp-list", *qp_list,
        "--delta-list", *delta_list,
        "--lambda-task", str(teach["lambda_task"]),
        "--lambda-anchor", str(teach["lambda_anchor"]),
        "--eta", str(teach["eta"]),
        "--xi", str(teach["xi"]),
        "--residual-bound", str(cfg.get("residual_bound", 2.0)),
        "--output", str(out_path),
    ]
    run(cmd, f"build_liteqp:{seq['name']}")


def concatenate_datasets(cfg, out_root: Path) -> Path:
    sfx = version_suffix(cfg)
    target = out_root / f"oracle_liteqp{sfx}" / f"all_liteqp{sfx}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as out:
        for seq in cfg["sequences"]:
            src = out_root / f"oracle_liteqp{sfx}" / f"{seq['name']}.jsonl"
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

def _backend(cfg: dict) -> str:
    """Resolve which residual model backend to use.

    ``cfg.train.backend`` ∈ {"mlp", "cnn"}. Default: "mlp" (legacy v3/v4
    behaviour). When set to "cnn", ``train.output_mode`` selects between
    "residual" (PROJECT_STATE §7.15 conservative variant) and "direct"
    (aggressive variant; no A+ prior at inference).
    """
    train_cfg = cfg.get("train", {}) or {}
    return str(train_cfg.get("backend", "mlp")).lower()


def step_train(cfg, dataset_path: Path, out_root: Path) -> Path:
    sfx = version_suffix(cfg)
    train_cfg = cfg.get("train", {}) or {}
    backend = _backend(cfg)

    if backend == "cnn":
        # CNN backend (PROJECT_STATE §7.15)
        model_path = out_root / "fit" / f"liteqp_cnn{sfx}.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-m", "phase2.phase3.train_liteqp_cnn",
            "--oracle", str(dataset_path),
            "--output", str(model_path),
            "--output-mode", str(train_cfg.get("output_mode", "residual")),
            "--residual-bound", str(cfg.get("residual_bound", 2.0)),
            "--epochs",      str(train_cfg.get("epochs", 100)),
            "--batch-size",  str(train_cfg.get("batch_size", 16)),
            "--lr",          str(train_cfg.get("lr", 1e-3)),
            "--weight-decay", str(train_cfg.get("weight_decay", 1e-4)),
            "--device",      str(train_cfg.get("device", "cpu")),
            "--seed",        str(train_cfg.get("seed", 20260506)),
            "--huber-delta", str(train_cfg.get("huber_delta", 0.5)),
            "--alpha-rnp",   str(train_cfg.get("alpha_rnp", 0.10)),
            "--alpha-tv",    str(train_cfg.get("alpha_tv",  0.005)),
            "--alpha-bound", str(train_cfg.get("alpha_bound", 0.10)),
            "--delta-max",   str(train_cfg.get("delta_max",  8.0)),
            "--cv",            train_cfg.get("cv", "both"),
            "--n-bootstrap",  str(train_cfg.get("n_bootstrap", 500)),
            "--bootstrap-alpha", str(train_cfg.get("bootstrap_alpha", 0.05)),
        ]
        if "output_bound" in train_cfg:
            cmd += ["--output-bound", str(train_cfg["output_bound"])]
        run(cmd, f"train_liteqp_cnn{sfx}")
        return model_path

    # Legacy MLP backend (default)
    model_path = out_root / "fit" / f"liteqp_mlp{sfx}.joblib"
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
    run(cmd, f"train_liteqp{sfx}")
    return model_path


# ---------------------------------------------------------------------------
# Step 6 — apply model to generate per-QP delta maps
# ---------------------------------------------------------------------------

def step_apply(cfg, seq, model_path: Path, out_root: Path) -> None:
    sfx = version_suffix(cfg)
    apply_cfg = cfg.get("apply", {}) or {}
    qaw = apply_cfg.get("q_aware_bound", {}) or {}
    train_cfg = cfg.get("train", {}) or {}
    backend = _backend(cfg)

    # Resolve --mode for apply_liteqp_model.py based on backend + output_mode.
    # Legacy MLP keeps the original "liteqp" / "a_plus" choices; OG modes
    # ("og_a_plus" / "og_liteqp") are only valid with backend == "mlp".
    if backend == "cnn":
        explicit_mode = apply_cfg.get("mode")
        if explicit_mode in ("cnn_residual", "cnn_direct"):
            mode = explicit_mode
        else:
            output_mode = str(train_cfg.get("output_mode", "residual"))
            mode = "cnn_residual" if output_mode == "residual" else "cnn_direct"
    else:
        mode = apply_cfg.get("mode", "liteqp")

    qp_list = [str(q) for q in cfg["qp_list"]]
    h = int(seq.get("height", 1152))
    w = int(seq.get("width", 1920))
    nrow = (h + 127) // 128
    ncol = (w + 127) // 128
    target_dir = out_root / "learned" / f"liteqp{sfx}_{seq['name']}" / "M4"
    cmd = [
        sys.executable, "-m", "phase2.phase3.apply_liteqp_model",
        "--mode", mode,
        "--model", str(model_path) if mode not in ("a_plus", "og_a_plus") else "",
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
    # Strip empty "--model" pair (analytic-only modes don't need a model).
    if "--model" in cmd:
        i = cmd.index("--model")
        if cmd[i + 1] == "":
            del cmd[i:i + 2]
    # OG-IPF arguments (PROJECT_STATE §7.20).
    if mode in ("og_a_plus", "og_liteqp"):
        og_cfg = cfg.get("og", {}) or {}
        boxes_dir = out_root / "saliency" / seq["name"]
        cmd += [
            "--boxes-dir", str(boxes_dir),
            "--frame-h",   str(h),
            "--frame-w",   str(w),
            "--ctu-size",  "128",
            "--og-lambda-ctu", str(og_cfg.get("lambda_ctu", 0.4)),
            "--og-lambda-obj", str(og_cfg.get("lambda_obj", 0.6)),
            "--og-gamma",       str(og_cfg.get("gamma",      0.0)),
            "--og-alpha-in",    str(og_cfg.get("alpha_in",   1.0)),
            "--og-alpha-ctx",   str(og_cfg.get("alpha_ctx",  0.3)),
            "--og-aggregator",  str(og_cfg.get("aggregator", "max")),
            "--og-p-norm",      str(og_cfg.get("p_norm",     4.0)),
            "--og-min-prot-floor", str(
                (og_cfg.get("min_protection") or {}).get("floor", 1.0)),
            "--og-min-prot-ceil",  str(
                (og_cfg.get("min_protection") or {}).get("ceil",  2.2)),
            "--og-min-prot-slope", str(
                (og_cfg.get("min_protection") or {}).get("slope", 0.08)),
            "--og-min-prot-eta",   str(
                (og_cfg.get("min_protection") or {}).get("eta",   0.7)),
            "--og-g-min-protect",  str(
                (og_cfg.get("min_protection") or {}).get("g_min_protect", 0.10)),
        ]
    # CNN inference device (default to whatever was used for training).
    if backend == "cnn":
        device = apply_cfg.get("device", train_cfg.get("device", "cpu"))
        cmd += ["--device", str(device)]
    # Action 3 — opt-in only (default OFF in pilot_v6 / v3 yaml).
    if bool(qaw.get("enabled", False)):
        cmd += [
            "--q-aware-bound",
            "--q-aware-slope", str(qaw.get("slope", 0.04)),
            "--q-aware-min",   str(qaw.get("lo",    1.4)),
            "--q-aware-max",   str(qaw.get("hi",    2.2)),
        ]
    run(cmd, f"apply_liteqp{sfx}:{seq['name']}")


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

    sfx = version_suffix(cfg)
    if sfx:
        LOG.info("Versioned run — suffix '%s' will be appended to "
                 "oracle/fit/learned dirs", sfx)

    # Action 4 — resolve auto-λ BEFORE the build_dataset step so
    # teacher_overrides is populated by the time step_build_dataset reads it.
    # (Disabled when ``auto_lambda.enabled`` is false / missing → no-op.)
    resolve_auto_lambda(cfg, out_root)

    if cfg.get("teacher_overrides"):
        LOG.info("Effective teacher_overrides:")
        for s, ov in cfg["teacher_overrides"].items():
            LOG.info("  %s = %s", s, ov)

    sequences = cfg.get("sequences", [])
    if args.only_sequences:
        sequences = [s for s in sequences if s["name"] in args.only_sequences]
        if not sequences:
            raise SystemExit(f"No sequences match --only-sequences {args.only_sequences}")
    LOG.info("Processing %d sequences → %s", len(sequences), out_root)

    apply_mode = (cfg.get("apply") or {}).get("mode", "liteqp")
    needs_boxes = apply_mode in ("og_a_plus", "og_liteqp")

    if args.start_step <= 1 <= args.end_step:
        for seq in sequences:
            step_saliency(cfg, seq, out_root)
            if needs_boxes:
                step_save_boxes(cfg, seq, out_root)
    if args.start_step <= 2 <= args.end_step:
        for seq in sequences:
            step_rate_surrogate(cfg, seq, out_root)

    # Steps 3–5 build the residual training set + train an MLP.
    # In analytic-only modes (a_plus, og_a_plus) the model is unused, so
    # we skip those steps entirely. The orchestrator log reflects this
    # decision so the user can re-enable training later by changing
    # ``apply.mode`` and re-running.
    is_analytic_only = apply_mode in ("a_plus", "og_a_plus")
    if is_analytic_only:
        LOG.info("[apply.mode=%s] analytic-only — skipping steps 3-5 "
                 "(no MLP residual is trained or needed).", apply_mode)

    if args.start_step <= 3 <= args.end_step and not is_analytic_only:
        for seq in sequences:
            step_build_dataset(cfg, seq, out_root)

    dataset_path = out_root / f"oracle_liteqp{sfx}" / f"all_liteqp{sfx}.jsonl"
    if args.start_step <= 4 <= args.end_step and not is_analytic_only:
        dataset_path = concatenate_datasets(cfg, out_root)

    model_path = out_root / "fit" / f"liteqp_mlp{sfx}.joblib"
    if args.start_step <= 5 <= args.end_step and not is_analytic_only:
        model_path = step_train(cfg, dataset_path, out_root)

    if args.start_step <= 6 <= args.end_step:
        for seq in sequences:
            step_apply(cfg, seq, model_path, out_root)

    encode_yaml = "phase2/configs/pilot_v5.yaml" if sfx == "_v2" \
        else "phase2/configs/pilot_v4.yaml"
    LOG.info("Stage C%s complete. Encode with %s.",
             sfx if sfx else "", encode_yaml)


if __name__ == "__main__":
    main()
