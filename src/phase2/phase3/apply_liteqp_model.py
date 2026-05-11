"""Phase 3 Stage C — Apply trained LiteQP residual model to generate dQP maps.

End-to-end mapping per (sequence, frame, Q_b):

    1.  φ_oracle = load(saliency / phi_oracle_<frame>.npy)
    2.  K_c      = load(rate_surrogate.npz, K_grids[frame])
    3.  δ_a+     = analytic_a_plus.compute_a_plus_delta(φ, K, Q_b)
                  → project_rate_neutral_exact(δ_a+, K)
    4.  features = (φ, K_norm, Q_b_norm, σ_Y, motion, τ_rel,
                    prev_δ, φ_neighbor_mean, φ_grad)         (per CTU)
    5.  r_hat    = clip(model.predict(features), -bound, +bound)
    6.  δ_pred   = δ_a+ + r_hat
    7.  δ_pred   = project_rate_neutral_exact(δ_pred, K)
    8.  δ_final  = clip(δ_pred,  -Δ_roi(Q_b), +Δ_bg(Q_b))
                  → clip(δ_final, [-8, +4])
                  → round to int
    9.  write Phase-2-compatible delta map (qp_<frame>.txt)

The same script also supports ``--mode a_plus`` (no model) for ablation
purposes — this is M4-A+ in the new naming scheme.

Run with::

    python -m phase2.phase3.apply_liteqp_model \
        --mode      liteqp \
        --model     ~/Minh/ipf/phase3_outputs/fit/liteqp_mlp.joblib \
        --rate-npz  ~/Minh/ipf/phase3_outputs/rate/MOT17-04-DPM/rate_surrogate.npz \
        --saliency-dir ~/Minh/ipf/phase3_outputs/saliency/MOT17-04-DPM \
        --frames-dir ~/Minh/ipf/datasets/MOT17/MOT17/train/MOT17-04-DPM/img1 \
        --output-dir ~/Minh/ipf/phase3_outputs/learned/liteqp_MOT17-04-DPM/M4 \
        --qp-list 27 32 37 42 \
        --n-frames 50 --ctu-rows 9 --ctu-cols 15
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from phase2.phase3.analytic_a_plus import (
    AnalyticAPlusConfig,
    compute_a_plus_delta,
    project_rate_neutral_clipped_exact,
    project_rate_neutral_exact,
    rate_neutral_residual,
    q_adaptive_bounds,
)
from phase2.phase3.train_liteqp_regressor import featurize_row, FEATURE_NAMES
from phase2.phase3.occupancy import (
    OGConfig,
    compute_occupancy_utility,
    load_boxes_for_frame,
)
from phase2.phase3.og_ipf import (
    MinProtectionConfig,
    apply_min_object_protection,
    compute_og_a_plus_delta,
    compute_og_diagnostics,
    min_protection_floor,
)

logger = logging.getLogger("phase2.phase3.apply_liteqp_model")


# ---------------------------------------------------------------------------
# Action 3 — Q-aware residual_bound (OPT-IN, default OFF)
# ---------------------------------------------------------------------------

def q_aware_residual_bound(qp: int, base: float = 2.0,
                            slope: float = 0.04,
                            lo: float = 1.4, hi: float = 2.2) -> float:
    """Residual bound that varies linearly with Q_base around Q=32.

        bound(Q) = clip(base − slope · (Q − 32),  lo,  hi)

    Sign of ``slope`` selects two opposite use-cases:

    *  ``slope > 0`` — TIGHTER at HIGH QP. (PROJECT_STATE §7.10/7.12.)
       Motivation: high-QP encodes are near the intra-prediction stability
       cliff — large MLP residuals there can collapse mAP. Low-QP encodes
       have spare bit budget and tolerate bigger swings. Default schedule
       (``slope=0.04, lo=1.4, hi=2.2``):

           QP=27 → 2.20    QP=32 → 2.00    QP=37 → 1.80    QP=42 → 1.60

    *  ``slope < 0`` — TIGHTER at LOW QP. (PROJECT_STATE §7.16, Path F.)
       Motivation: pilot_v8b (CNN-direct) regressed at QP=27 because
       redistribution doesn't help when M0 already has spare bits; it
       just costs mAP. Tightening δ at low QP forces the CNN to be
       conservative when there's nothing to gain. Recommended schedule
       (``slope=-0.10, base=2.0, lo=1.0, hi=3.0``):

           QP=27 → 1.50    QP=32 → 2.00    QP=37 → 2.50    QP=42 → 3.00

    Applied to:
      - ``cnn_residual``: clamps the CNN's residual r̂ → matching MLP semantics.
      - ``cnn_direct``  : clamps the FULL δ̂ output of the CNN. Same formula,
                           different target — kept under one flag for orthogonality.
    """
    return float(min(max(base - slope * (qp - 32.0), lo), hi))


# ---------------------------------------------------------------------------
# Helpers (kept consistent with build_liteqp_dataset.py)
# ---------------------------------------------------------------------------

def _percentile_norm(arr: np.ndarray, p_lo=5.0, p_hi=95.0) -> np.ndarray:
    lo = float(np.percentile(arr, p_lo))
    hi = float(np.percentile(arr, p_hi))
    return np.clip((arr - lo) / max(1e-9, hi - lo), 0.0, 1.0)


def _per_ctu_luma_std(frame_bgr: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    y = (0.114 * frame_bgr[..., 0]
         + 0.587 * frame_bgr[..., 1]
         + 0.299 * frame_bgr[..., 2]).astype(np.float64)
    h, w = y.shape
    bh = max(1, (h + n_rows - 1) // n_rows)
    bw = max(1, (w + n_cols - 1) // n_cols)
    out = np.zeros((n_rows, n_cols), dtype=np.float64)
    for r in range(n_rows):
        for c in range(n_cols):
            block = y[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw]
            out[r, c] = float(np.std(block)) if block.size else 0.0
    return out


def _per_ctu_temporal_diff(prev_bgr, curr_bgr, n_rows, n_cols) -> np.ndarray:
    yp = 0.114 * prev_bgr[..., 0] + 0.587 * prev_bgr[..., 1] + 0.299 * prev_bgr[..., 2]
    yc = 0.114 * curr_bgr[..., 0] + 0.587 * curr_bgr[..., 1] + 0.299 * curr_bgr[..., 2]
    diff = np.abs(yc.astype(np.float64) - yp.astype(np.float64))
    h, w = diff.shape
    bh = max(1, (h + n_rows - 1) // n_rows)
    bw = max(1, (w + n_cols - 1) // n_cols)
    out = np.zeros((n_rows, n_cols), dtype=np.float64)
    for r in range(n_rows):
        for c in range(n_cols):
            block = diff[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw]
            out[r, c] = float(np.mean(block)) if block.size else 0.0
    return out


def _phi_neighbor_mean(phi: np.ndarray) -> np.ndarray:
    h, w = phi.shape
    out = np.zeros_like(phi)
    for r in range(h):
        for c in range(w):
            r0, r1 = max(0, r - 1), min(h, r + 2)
            c0, c1 = max(0, c - 1), min(w, c + 2)
            out[r, c] = float(np.mean(phi[r0:r1, c0:c1]))
    return out


def _phi_gradient(phi: np.ndarray) -> np.ndarray:
    gh = np.zeros_like(phi); gw = np.zeros_like(phi)
    gh[:-1, :] = np.abs(phi[1:, :] - phi[:-1, :])
    gw[:, :-1] = np.abs(phi[:, 1:] - phi[:, :-1])
    return gh + gw


def _load_frame_bgr(frames_dir: Path, frame_idx: int) -> Optional[np.ndarray]:
    for ext in ("jpg", "jpeg", "png"):
        for offset in (1, 0):  # MOT17 is 1-indexed
            path = frames_dir / f"{frame_idx + offset:06d}.{ext}"
            if path.exists():
                try:
                    import cv2
                except ImportError:
                    return None
                return cv2.imread(str(path), cv2.IMREAD_COLOR)
    return None


def _write_delta_map(delta: np.ndarray, output_path: Path, frame_idx: int,
                     delta_min: int = -8, delta_max: int = 4) -> None:
    clipped = np.clip(np.rint(delta), delta_min, delta_max).astype(np.int32)
    n_rows, n_cols = clipped.shape
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(
            f"# frame={frame_idx} rows={n_rows} cols={n_cols} type=delta "
            f"delta_min={delta_min} delta_max={delta_max}\n"
        )
        for row in range(n_rows):
            f.write(" ".join(f"{clipped[row, col]:+d}" for col in range(n_cols)) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Apply LiteQP / A+ to make dQP maps")
    parser.add_argument("--mode", choices=["liteqp", "a_plus", "cnn_residual",
                                            "cnn_direct",
                                            "og_a_plus", "og_liteqp"],
                        required=True,
                        help='"liteqp"        = A+ + MLP residual on Φ_oracle; '
                             '"a_plus"        = analytic baseline on Φ_oracle; '
                             '"cnn_residual"  = A+ + CNN residual (PROJECT_STATE §7.15); '
                             '"cnn_direct"    = end-to-end CNN, NO A+ prior; '
                             '"og_a_plus"     = OG-IPF analytic + min-protection (NEW §7.20); '
                             '"og_liteqp"     = OG-A+ + MLP residual on OG features.')
    parser.add_argument("--model", default="",
                        help="Required if mode=liteqp/cnn_*; "
                             "joblib bundle (MLP) or .pt bundle (CNN)")
    parser.add_argument("--device", default="cpu",
                        help="Torch device for CNN inference (default cpu).")
    parser.add_argument("--rate-npz", required=True)
    parser.add_argument("--saliency-dir", required=True)
    parser.add_argument("--frames-dir", default="",
                        help="Optional; needed for σ_Y / motion features")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--qp-list", nargs="+", type=int, default=[27, 32, 37, 42])
    parser.add_argument("--n-frames", type=int, default=50)
    parser.add_argument("--ctu-rows", type=int, default=9)
    parser.add_argument("--ctu-cols", type=int, default=15)
    parser.add_argument("--residual-bound", type=float, default=2.0,
                        help="Constant cap on |MLP residual|. Used unless "
                             "--q-aware-bound is set, in which case this is "
                             "the *base* bound passed to q_aware_residual_bound().")
    parser.add_argument("--q-aware-bound", action="store_true",
                        help="(Action 3, OPT-IN). Make the residual bound a "
                             "function of Q_base via q_aware_residual_bound() — "
                             "tighter at high QP where the MLP can over-shoot. "
                             "Default OFF for clean comparison vs constant-bound "
                             "models (e.g. pilot_v4).")
    parser.add_argument("--q-aware-slope", type=float, default=0.04,
                        help="Slope per QP step around Q=32 when --q-aware-bound "
                             "is set. Default 0.04 (gentle: ±2.0 → ±1.6 at QP=42).")
    parser.add_argument("--q-aware-min", type=float, default=1.4,
                        help="Floor for q_aware_residual_bound. Default 1.4.")
    parser.add_argument("--q-aware-max", type=float, default=2.2,
                        help="Ceiling for q_aware_residual_bound. Default 2.2.")
    parser.add_argument("--per-qp", action="store_true",
                        help="Write a separate qp_vtm_delta_QP<n>/ per QP. If "
                             "absent, writes only one map (assumes Q_b=32).")

    # --- OG-IPF inputs (PROJECT_STATE §7.20) ---------------------------------
    parser.add_argument("--boxes-dir", default="",
                        help="Directory with per-frame YOLO boxes JSON "
                             "(written by `phase2.phase3.save_boxes`). "
                             "Required for --mode og_a_plus and og_liteqp.")
    parser.add_argument("--frame-h", type=int, default=0,
                        help="Frame height in pixels (used by OG-IPF for "
                             "CTU-grid clipping). If 0, derived from "
                             "ctu_rows × ctu_size.")
    parser.add_argument("--frame-w", type=int, default=0,
                        help="Frame width in pixels.")
    parser.add_argument("--ctu-size", type=int, default=128,
                        help="CTU side length in pixels (default 128).")
    parser.add_argument("--og-lambda-ctu", type=float, default=0.4)
    parser.add_argument("--og-lambda-obj", type=float, default=0.6)
    parser.add_argument("--og-gamma",      type=float, default=0.0,
                        help="Object size exponent γ ∈ [0, 0.25]. "
                             "0 = ignore size (default; reviewer-safe).")
    parser.add_argument("--og-alpha-in",   type=float, default=1.0)
    parser.add_argument("--og-alpha-ctx",  type=float, default=0.3)
    parser.add_argument("--og-aggregator", choices=["max", "lp"], default="max")
    parser.add_argument("--og-p-norm",     type=float, default=4.0)
    parser.add_argument("--og-min-prot-floor",  type=float, default=1.0)
    parser.add_argument("--og-min-prot-ceil",   type=float, default=2.2)
    parser.add_argument("--og-min-prot-slope",  type=float, default=0.08)
    parser.add_argument("--og-min-prot-eta",    type=float, default=0.7)
    parser.add_argument("--og-g-min-protect",   type=float, default=0.10,
                        help="CTUs with G ≤ this value are NOT hard-capped "
                             "at -δ_min·G^η — they keep the global +bg_bound "
                             "so the rate-neutral projection has slack. "
                             "Default 0.10 (10 %% overlap = roughly half a "
                             "CTU edge crossed).")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    rate_data = np.load(Path(args.rate_npz).expanduser().resolve())
    K_grids = rate_data["K_grids"]
    n_frames_avail, n_rows, n_cols = K_grids.shape
    if (n_rows, n_cols) != (args.ctu_rows, args.ctu_cols):
        logger.warning("CTU shape from K_grids %s != arg %s — using K_grids shape",
                       (n_rows, n_cols), (args.ctu_rows, args.ctu_cols))
        args.ctu_rows, args.ctu_cols = n_rows, n_cols

    sal_dir = Path(args.saliency_dir).expanduser().resolve()
    frames_dir = Path(args.frames_dir).expanduser().resolve() if args.frames_dir else None
    out_root = Path(args.output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # ── OG-IPF: validate boxes directory and build configs ─────────────────
    og_cfg: Optional[OGConfig] = None
    og_min_cfg: Optional[MinProtectionConfig] = None
    boxes_dir: Optional[Path] = None
    og_diagnostics_log: list[dict] = []  # one dict per frame across all QPs
    if args.mode in ("og_a_plus", "og_liteqp"):
        if not args.boxes_dir:
            raise SystemExit(
                f"--boxes-dir required when --mode {args.mode}. "
                "Run `phase2.phase3.save_boxes` first.")
        boxes_dir = Path(args.boxes_dir).expanduser().resolve()
        if not boxes_dir.is_dir():
            raise SystemExit(f"boxes-dir does not exist: {boxes_dir}")
        og_cfg = OGConfig(
            lambda_ctu=args.og_lambda_ctu,
            lambda_obj=args.og_lambda_obj,
            gamma=args.og_gamma,
            alpha_in=args.og_alpha_in,
            alpha_ctx=args.og_alpha_ctx,
            aggregator=args.og_aggregator,
            p_norm=args.og_p_norm,
        )
        og_min_cfg = MinProtectionConfig(
            intercept=args.og_min_prot_floor,
            slope=args.og_min_prot_slope,
            floor=args.og_min_prot_floor,
            ceil=args.og_min_prot_ceil,
            eta=args.og_min_prot_eta,
            g_min_protect=args.og_g_min_protect,
        )
        # Default frame-h/w from ctu_rows × ctu_size if not given.
        if args.frame_h <= 0:
            args.frame_h = args.ctu_rows * args.ctu_size
        if args.frame_w <= 0:
            args.frame_w = args.ctu_cols * args.ctu_size
        logger.info("OG-IPF mode=%s: boxes_dir=%s, frame=%dx%d, "
                     "λ_ctu=%.2f λ_obj=%.2f γ=%.2f α_in=%.2f α_ctx=%.2f "
                     "min_prot[floor=%.2f, ceil=%.2f, slope=%.3f, η=%.2f]",
                     args.mode, boxes_dir, args.frame_h, args.frame_w,
                     og_cfg.lambda_ctu, og_cfg.lambda_obj, og_cfg.gamma,
                     og_cfg.alpha_in, og_cfg.alpha_ctx,
                     og_min_cfg.floor, og_min_cfg.ceil,
                     og_min_cfg.slope, og_min_cfg.eta)

    # ── Load model (mode=liteqp / cnn_* / og_liteqp) ───────────────────────
    # ``pipeline`` (sklearn MLP) and ``cnn_model`` (PyTorch nn.Module) are
    # mutually exclusive — only one is populated based on ``--mode``.
    pipeline = None
    cnn_model = None
    cnn_bundle = None
    if args.mode in ("liteqp", "og_liteqp"):
        if not args.model:
            raise SystemExit(f"--model required when --mode {args.mode}")
        try:
            import joblib
        except ImportError as exc:
            raise SystemExit("joblib required: %s" % exc) from exc
        bundle = joblib.load(Path(args.model).expanduser().resolve())
        pipeline = bundle["model"]
        logger.info("Loaded LiteQP MLP bundle from %s", args.model)
        meta = bundle.get("meta", {})
        if meta.get("feature_names") and meta["feature_names"] != FEATURE_NAMES:
            logger.warning("Feature schema drift: model expects %s; current %s",
                           meta["feature_names"], FEATURE_NAMES)
    elif args.mode in ("cnn_residual", "cnn_direct"):
        if not args.model:
            raise SystemExit(f"--model required when --mode {args.mode}")
        from phase2.phase3.liteqp_cnn import load_bundle as _cnn_load_bundle
        cnn_model, cnn_bundle = _cnn_load_bundle(
            Path(args.model).expanduser().resolve())
        # Move model to requested device.
        try:
            import torch  # noqa: F401
            cnn_model.to(args.device)
        except Exception as exc:
            logger.warning("Could not move CNN to %s (%s); using CPU.",
                            args.device, exc)
            args.device = "cpu"
        # Sanity: ``--mode`` must match what the bundle was trained for.
        expected_mode = ("residual" if args.mode == "cnn_residual" else "direct")
        if cnn_bundle.output_mode != expected_mode:
            raise SystemExit(
                f"Bundle was trained with output_mode={cnn_bundle.output_mode!r}, "
                f"but --mode {args.mode} expects {expected_mode!r}.")
        logger.info("Loaded LiteQP-CNN bundle (mode=%s, n_input=%d, n_params=%d) "
                     "from %s", cnn_bundle.output_mode, cnn_bundle.n_input_channels,
                     sum(p.numel() for p in cnn_model.parameters()), args.model)

    # ── Pre-compute σ_Y / motion (only for liteqp mode) ────────────────────
    sigma_per_frame: List[np.ndarray] = []
    motion_per_frame: List[np.ndarray] = []
    n_frames = min(args.n_frames, n_frames_avail)
    if args.mode == "liteqp" and frames_dir is not None:
        prev_bgr = None
        for fi in range(n_frames):
            curr_bgr = _load_frame_bgr(frames_dir, fi)
            if curr_bgr is None:
                sigma_per_frame.append(np.ones((args.ctu_rows, args.ctu_cols)))
                motion_per_frame.append(np.zeros((args.ctu_rows, args.ctu_cols)))
            else:
                sigma_per_frame.append(_per_ctu_luma_std(
                    curr_bgr, args.ctu_rows, args.ctu_cols))
                if prev_bgr is None:
                    motion_per_frame.append(np.zeros((args.ctu_rows, args.ctu_cols)))
                else:
                    motion_per_frame.append(_per_ctu_temporal_diff(
                        prev_bgr, curr_bgr, args.ctu_rows, args.ctu_cols))
                prev_bgr = curr_bgr
        if sigma_per_frame:
            sa = np.stack(sigma_per_frame, axis=0)
            ma = np.stack(motion_per_frame, axis=0)
            sa = sa / max(1e-9, float(np.percentile(sa, 95)))
            ma = ma / max(1e-9, float(np.percentile(ma, 95)))
            sigma_per_frame = [sa[i] for i in range(n_frames)]
            motion_per_frame = [ma[i] for i in range(n_frames)]
    else:
        sigma_per_frame = [np.ones((args.ctu_rows, args.ctu_cols))
                           for _ in range(n_frames)]
        motion_per_frame = [np.zeros((args.ctu_rows, args.ctu_cols))
                            for _ in range(n_frames)]

    cfg = AnalyticAPlusConfig()

    # ── Per-QP loop ────────────────────────────────────────────────────────
    # ``--per-qp`` is mandatory because the analytic A+ formula and the MLP
    # residual are both Q-adaptive (different Q_b ⇒ different scale ⇒
    # different δ map). Running with a single Q_b would mis-calibrate every
    # other QP — exactly the failure mode that motivated Phase 3.
    if not args.per_qp:
        raise SystemExit(
            "LiteQP / A+ requires --per-qp to avoid Q_base calibration "
            "mismatch (the formula is Q-adaptive). Re-run with --per-qp."
        )
    qp_iter = args.qp_list
    n_total_written = 0
    rate_ratio_log: list[dict] = []  # populated below for metadata.json
    if args.q_aware_bound:
        logger.info("Q-AWARE RESIDUAL BOUND enabled "
                    "(base=%.2f, slope=%.2f, clip [%.2f, %.2f]):",
                    args.residual_bound, args.q_aware_slope,
                    args.q_aware_min, args.q_aware_max)
        for q in qp_iter:
            b = q_aware_residual_bound(q, base=args.residual_bound,
                                          slope=args.q_aware_slope,
                                          lo=args.q_aware_min,
                                          hi=args.q_aware_max)
            logger.info("  QP=%d → residual bound = ±%.2f", q, b)
    for qp in qp_iter:
        qp_dir = out_root / f"qp_vtm_delta_QP{qp}"
        qp_dir.mkdir(parents=True, exist_ok=True)

        prev_delta = np.zeros((args.ctu_rows, args.ctu_cols))
        roi_bound, bg_bound = q_adaptive_bounds(qp)
        # Action 3 (opt-in): per-Q residual cap.
        if args.q_aware_bound:
            residual_cap = q_aware_residual_bound(
                qp, base=args.residual_bound, slope=args.q_aware_slope,
                lo=args.q_aware_min, hi=args.q_aware_max)
        else:
            residual_cap = float(args.residual_bound)
        # Final clip bounds intersected with VVC legal range.
        clip_lo = max(-roi_bound, float(cfg.delta_min_clip))
        clip_hi = min(+bg_bound, float(cfg.delta_max_clip))
        ratios_pre_round, ratios_post_round = [], []
        for fi in range(n_frames):
            K_grid = K_grids[fi]
            K_norm = K_grid / (np.median(K_grid[K_grid > 0]) + 1e-9
                               if (K_grid > 0).any() else 1.0)
            G_max_grid: Optional[np.ndarray] = None  # only set in OG modes

            # ── Build phi_norm + (optionally) U_c / G_c_max ─────────────
            if args.mode in ("og_a_plus", "og_liteqp"):
                # OG-IPF: load per-frame YOLO boxes and compute U_c, G_c_max.
                # We *also* load Φ_oracle for the MLP feature stack in
                # og_liteqp; the MLP residual still operates on the same
                # 9-feature vector as plain liteqp for parity, with `phi`
                # replaced by U_c (normalised to [0,1] via percentile).
                boxes = load_boxes_for_frame(boxes_dir, fi)
                og = compute_occupancy_utility(
                    boxes,
                    frame_h=args.frame_h, frame_w=args.frame_w,
                    ctu_size=args.ctu_size, cfg=og_cfg,
                )
                # The util signal is unbounded; percentile-normalise to [0,1]
                # for the MLP feature vector and downstream consistency.
                U_norm = _percentile_norm(og.U)
                G_max_grid = og.G_max
                if og.U.shape != (args.ctu_rows, args.ctu_cols):
                    logger.warning(
                        "frame %06d: OG-IPF grid shape %s ≠ %s — skipping",
                        fi, og.U.shape, (args.ctu_rows, args.ctu_cols))
                    continue
                # 1) OG-A+ analytic prior on U_c (NOT on Φ).
                delta_a = compute_og_a_plus_delta(og.U, K_grid, qp, cfg)
                # 2) Min-object-protection — applied BEFORE projection so
                #    rate-neutrality is restored after the constraint nudges
                #    object-CTUs further negative.
                delta_a = apply_min_object_protection(
                    delta_a, og.G_max, qp, og_min_cfg)
                delta_a = project_rate_neutral_exact(delta_a, K_grid)
                phi_norm = U_norm  # used by og_liteqp's MLP features
            else:
                sal_path = sal_dir / f"phi_oracle_{fi:06d}.npy"
                if not sal_path.exists():
                    logger.warning("frame %06d: missing saliency, writing zeros", fi)
                    _write_delta_map(np.zeros((args.ctu_rows, args.ctu_cols)),
                                     qp_dir / f"qp_{fi:06d}.txt", fi)
                    continue
                phi_raw = np.load(sal_path).astype(np.float64)
                if phi_raw.shape != (args.ctu_rows, args.ctu_cols):
                    logger.warning("frame %06d: saliency shape %s ≠ %s — skipping",
                                   fi, phi_raw.shape, (args.ctu_rows, args.ctu_cols))
                    continue
                phi_norm = _percentile_norm(phi_raw)
                # 1) Analytic prior + exact rate-neutral projection
                #    NOTE: cnn_direct mode bypasses A+ entirely (delta_a is still
                #    computed for metadata purposes — useful for diagnostics).
                delta_a = compute_a_plus_delta(phi_norm, K_grid, qp, cfg)
                delta_a = project_rate_neutral_exact(delta_a, K_grid)

            if args.mode in ("liteqp", "og_liteqp"):
                # 2) Build features + predict residual (per-CTU MLP).
                #    The 9-feature vector is identical for both modes;
                #    only the ``phi`` channel changes meaning:
                #    * liteqp     → phi = Φ_oracle (occlusion saliency)
                #    * og_liteqp  → phi = U_c (normalised OG utility)
                phi_nbr = _phi_neighbor_mean(phi_norm)
                phi_grd = _phi_gradient(phi_norm)
                feats = []
                for r in range(args.ctu_rows):
                    for c in range(args.ctu_cols):
                        feats.append(featurize_row({
                            "phi":      float(phi_norm[r, c]),
                            "K_c_norm": float(K_norm[r, c]),
                            "q_base":   qp,
                            "q_base_norm": float((qp - 32.0) / 10.0),
                            "sigma_y":     float(sigma_per_frame[fi][r, c]),
                            "motion_proxy": float(motion_per_frame[fi][r, c]),
                            "temporal_reliability": 1.0,
                            "prev_delta":  float(prev_delta[r, c]),
                            "phi_neighbor_mean": float(phi_nbr[r, c]),
                            "phi_grad":    float(phi_grd[r, c]),
                        }))
                X = np.asarray(feats, dtype=np.float32)
                r_hat = pipeline.predict(X)
                r_hat = np.clip(r_hat, -residual_cap, +residual_cap)
                r_hat = r_hat.reshape(args.ctu_rows, args.ctu_cols)
                delta_pred = delta_a + r_hat
                # In OG modes, re-apply the min-protection AFTER residual
                # so the MLP cannot accidentally up-shift an object CTU
                # past 0. (The constraint is idempotent.)
                if args.mode == "og_liteqp" and G_max_grid is not None:
                    delta_pred = apply_min_object_protection(
                        delta_pred, G_max_grid, qp, og_min_cfg)
            elif args.mode in ("cnn_residual", "cnn_direct"):
                # 2) Build the spatial feature stack (7 channels) and run
                #    one CNN forward pass — much cheaper than the per-CTU
                #    MLP loop.
                from phase2.phase3.liteqp_cnn import (
                    make_input_planes as _cnn_make_input_planes,
                    cnn_predict as _cnn_predict,
                )
                phi_grd = _phi_gradient(phi_norm)
                planes = _cnn_make_input_planes(
                    phi=phi_norm, K_norm=K_norm,
                    sigma=sigma_per_frame[fi], motion=motion_per_frame[fi],
                    prev_delta=prev_delta,
                    q_base_norm=float((qp - 32.0) / 10.0),
                    phi_grad=phi_grd,
                )
                cnn_out = _cnn_predict(cnn_model, planes, device=args.device)
                if args.mode == "cnn_residual":
                    # ``residual_cap`` is the MLP-style residual bound (or the
                    # Q-aware schedule when --q-aware-bound is on).
                    cnn_out = np.clip(cnn_out, -residual_cap, +residual_cap)
                    delta_pred = delta_a + cnn_out
                else:  # cnn_direct: model output IS the δ map
                    if args.q_aware_bound:
                        # Path F (PROJECT_STATE §7.16): clip the FULL δ̂
                        # output by the same Q-aware schedule. Negative
                        # ``--q-aware-slope`` tightens at low QP, where
                        # pilot_v8b's CNN was over-aggressive.
                        delta_pred = np.clip(cnn_out, -residual_cap, +residual_cap)
                    else:
                        delta_pred = cnn_out
            elif args.mode == "og_a_plus":
                # OG-IPF analytic-only: delta_a already includes
                # min-object-protection from the OG branch above.
                delta_pred = delta_a
            else:  # a_plus mode (legacy Φ_oracle baseline)
                delta_pred = delta_a

            # 3) Clip-aware exact projection — preserves rate-neutrality
            #    *after* clipping (plain exact projection breaks once a CTU
            #    saturates at a bound). Returns an already-clipped map.
            #
            #    In OG modes the upper bound is **per-CTU**:
            #        upper_c = min(+bg_bound,  -δ_min(Q_b)·G_c^η)
            #                  if G_c > g_min_protect (default 0.1)
            #        upper_c = +bg_bound otherwise
            #    so the projection's global shift cannot push strongly
            #    protected CTUs back above zero, while weakly-overlapping
            #    CTUs (~20 % of the grid in MOT17-09) still have positive
            #    headroom for the projection to balance the rate sum.
            #    Without this threshold the rate ratio drifts > 5 % at
            #    QP=42 on MOT17-09 (PROJECT_STATE §7.20.3 fix).
            if args.mode in ("og_a_plus", "og_liteqp") and G_max_grid is not None:
                delta_min_q = min_protection_floor(qp, og_min_cfg)
                g_arr = np.clip(G_max_grid, 0.0, 1.0)
                g_thresh = float(og_min_cfg.g_min_protect)
                upper_per_ctu_protected = np.minimum(
                    clip_hi,
                    -delta_min_q * np.power(g_arr, og_min_cfg.eta),
                )
                upper_per_ctu = np.where(g_arr > g_thresh,
                                          upper_per_ctu_protected,
                                          clip_hi)
                delta_pred = project_rate_neutral_clipped_exact(
                    delta_pred, K_grid,
                    delta_min=clip_lo, delta_max=upper_per_ctu,
                )
            else:
                delta_pred = project_rate_neutral_clipped_exact(
                    delta_pred, K_grid,
                    delta_min=clip_lo, delta_max=clip_hi,
                )

            # Continuous-map ratio (should be 1.0 to machine precision).
            ratios_pre_round.append(rate_neutral_residual(delta_pred, K_grid))

            # 4) Final integer rounding (VVC requires integer δQP).
            #    In OG modes we cap the rounded value at floor(upper_per_ctu)
            #    so integer rounding cannot push protected CTUs back across
            #    zero either. (`floor(-0.616) = -1` ensures δ ≤ -1, which is
            #    safely ≤ -0.616 ⇒ (★) constraint holds.)
            if args.mode in ("og_a_plus", "og_liteqp") and G_max_grid is not None:
                upper_int = np.floor(upper_per_ctu)
                delta_int = np.minimum(
                    np.maximum(np.rint(delta_pred), float(cfg.delta_min_clip)),
                    upper_int,
                )
            else:
                delta_int = np.clip(np.rint(delta_pred),
                                    cfg.delta_min_clip, cfg.delta_max_clip)
            ratios_post_round.append(rate_neutral_residual(delta_int, K_grid))

            # OG diagnostics (PROJECT_STATE §7.20 / §9 of user spec).
            if args.mode in ("og_a_plus", "og_liteqp") and G_max_grid is not None:
                diag = compute_og_diagnostics(
                    delta_continuous=delta_pred,
                    delta_int=delta_int,
                    G_max=G_max_grid,
                    K=K_grid,
                    delta_min_clip=float(cfg.delta_min_clip),
                    delta_max_clip=float(cfg.delta_max_clip),
                    rate_ratio_continuous=ratios_pre_round[-1],
                    rate_ratio_rounded=ratios_post_round[-1],
                )
                og_diagnostics_log.append({
                    "frame_idx": fi, "qp_base": int(qp),
                    "n_ctus_total":            diag.n_ctus_total,
                    "pct_object_overlap":      diag.pct_object_overlap,
                    "pct_context":             diag.pct_context,
                    "pct_far_background":      diag.pct_far_background,
                    "mean_delta_object":       diag.mean_delta_object,
                    "mean_delta_context":      diag.mean_delta_context,
                    "mean_delta_far_bg":       diag.mean_delta_far_background,
                    "saturation_lower":        diag.saturation_lower,
                    "saturation_upper":        diag.saturation_upper,
                    "rate_ratio_continuous":   diag.rate_ratio_continuous,
                    "rate_ratio_rounded":      diag.rate_ratio_rounded,
                    "object_protection_violation": diag.object_protection_violation,
                })

            _write_delta_map(delta_int, qp_dir / f"qp_{fi:06d}.txt", fi)
            prev_delta = delta_int.astype(np.float64)
            n_total_written += 1

        rate_ratio_log.append({
            "q_base": int(qp),
            "n_frames": n_frames,
            "residual_cap": float(residual_cap),
            "rate_ratio_pre_round_mean":  float(np.mean(ratios_pre_round)),
            "rate_ratio_pre_round_max_abs_dev": float(np.max(np.abs(
                np.asarray(ratios_pre_round) - 1.0))),
            "rate_ratio_post_round_mean": float(np.mean(ratios_post_round)),
            "rate_ratio_post_round_max_abs_dev": float(np.max(np.abs(
                np.asarray(ratios_post_round) - 1.0))),
        })
        logger.info(
            "Q_b=%d → wrote %d maps; rate-ratio pre-round = %.6f "
            "(max dev %.2e), post-round = %.6f (max dev %.2e)",
            qp, n_frames,
            rate_ratio_log[-1]["rate_ratio_pre_round_mean"],
            rate_ratio_log[-1]["rate_ratio_pre_round_max_abs_dev"],
            rate_ratio_log[-1]["rate_ratio_post_round_mean"],
            rate_ratio_log[-1]["rate_ratio_post_round_max_abs_dev"],
        )

    # ── Persist metadata ───────────────────────────────────────────────────
    method_label = {
        "liteqp":       "M4-LiteQP",
        "a_plus":       "M4-A+",
        "cnn_residual": "M4-CNN-residual",
        "cnn_direct":   "M4-CNN-direct",
        "og_a_plus":    "M4-OG-A+",
        "og_liteqp":    "M4-OG-LiteQP",
    }[args.mode]
    meta = {
        "method":        method_label,
        "mode":          args.mode,
        "model_path":    args.model if args.mode not in ("a_plus", "og_a_plus") else None,
        "saliency_dir":  str(sal_dir),
        "rate_npz":      str(Path(args.rate_npz).resolve()),
        "frames_dir":    str(frames_dir) if frames_dir else None,
        "boxes_dir":     str(boxes_dir) if boxes_dir else None,
        "qp_list":       args.qp_list,
        "per_qp":        bool(args.per_qp),
        "n_frames":      n_frames,
        "ctu_grid":      [args.ctu_rows, args.ctu_cols],
        "residual_bound": float(args.residual_bound),
        "q_aware_bound": {
            "enabled":       bool(args.q_aware_bound),
            "base":          float(args.residual_bound),
            "slope":         float(args.q_aware_slope),
            "lo":            float(args.q_aware_min),
            "hi":            float(args.q_aware_max),
        },
        "og_config": (None if args.mode not in ("og_a_plus", "og_liteqp") else {
            "lambda_ctu": float(args.og_lambda_ctu),
            "lambda_obj": float(args.og_lambda_obj),
            "gamma":      float(args.og_gamma),
            "alpha_in":   float(args.og_alpha_in),
            "alpha_ctx":  float(args.og_alpha_ctx),
            "aggregator": str(args.og_aggregator),
            "p_norm":     float(args.og_p_norm),
            "min_protection": {
                "floor": float(args.og_min_prot_floor),
                "ceil":  float(args.og_min_prot_ceil),
                "slope": float(args.og_min_prot_slope),
                "eta":   float(args.og_min_prot_eta),
                "g_min_protect": float(args.og_g_min_protect),
            },
        }),
        "feature_names": FEATURE_NAMES,
        # Rate-neutrality monitoring — the **continuous, clipped** map is
        # rate-neutral by construction (pre_round_mean ≈ 1.0). The integer
        # map drifts by ~0.5–2 % typically; flag if > 5 %.
        "rate_neutral_log": rate_ratio_log,
        # Per-frame OG diagnostics (only populated in og_a_plus / og_liteqp).
        "og_diagnostics_per_frame": og_diagnostics_log,
    }
    with open(out_root / "liteqp_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("Wrote metadata to %s", out_root / "liteqp_metadata.json")
    logger.info("Total maps written: %d", n_total_written)

    # OG diagnostics summary CSV (PROJECT_STATE §7.20 / §9 of user spec).
    # Aggregates the per-frame OG stats by (qp_base) so the user can
    # eyeball the rate-ratio drift / object-protection violation at a
    # glance without parsing JSON.
    if args.mode in ("og_a_plus", "og_liteqp") and og_diagnostics_log:
        import csv
        from collections import defaultdict
        agg: dict = defaultdict(list)
        for row in og_diagnostics_log:
            agg[row["qp_base"]].append(row)
        csv_path = out_root / "og_diagnostics_summary.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "qp_base", "n_frames",
                "mean_pct_object_overlap",
                "mean_pct_context",
                "mean_pct_far_bg",
                "mean_delta_object",
                "mean_delta_context",
                "mean_delta_far_bg",
                "mean_saturation_lower",
                "mean_saturation_upper",
                "mean_rate_ratio_continuous",
                "mean_rate_ratio_rounded",
                "object_protection_violation_max",
            ])
            for qp_v in sorted(agg):
                rows = agg[qp_v]
                def _avg(key: str) -> float:
                    vals = [float(r[key]) for r in rows
                            if r[key] == r[key]]  # drop NaN
                    return float(np.mean(vals)) if vals else float("nan")
                w.writerow([
                    qp_v, len(rows),
                    _avg("pct_object_overlap"),
                    _avg("pct_context"),
                    _avg("pct_far_background"),
                    _avg("mean_delta_object"),
                    _avg("mean_delta_context"),
                    _avg("mean_delta_far_bg"),
                    _avg("saturation_lower"),
                    _avg("saturation_upper"),
                    _avg("rate_ratio_continuous"),
                    _avg("rate_ratio_rounded"),
                    max((float(r["object_protection_violation"])
                          for r in rows), default=0.0),
                ])
        logger.info("Wrote OG diagnostics summary to %s", csv_path)
        # Hard-fail soft warning: object-protection violation is the headline
        # invariant of the OG-IPF method. The acceptance criterion is < 5 %.
        worst = max(
            (float(r["object_protection_violation"])
             for r in og_diagnostics_log), default=0.0)
        if worst > 0.05:
            logger.warning(
                "OG-IPF: object_protection_violation = %.2f %% > 5 %% — "
                "either the min-protection schedule is too weak or the "
                "rate-neutral projection is fighting the constraint at "
                "+bg_bound. Investigate before claiming OG-IPF works.",
                worst * 100.0)
    # Soft warning if any QP shows large post-round drift.
    for entry in rate_ratio_log:
        drift = abs(entry["rate_ratio_post_round_mean"] - 1.0)
        if drift > 0.05:
            logger.warning("Q_b=%d post-round rate drift = %.2f %% (>5 %%)",
                           entry["q_base"], drift * 100.0)


if __name__ == "__main__":
    main()
