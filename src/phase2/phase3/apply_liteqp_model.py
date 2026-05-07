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
                                            "cnn_direct"],
                        required=True,
                        help='"liteqp"        = A+ + MLP residual; '
                             '"a_plus"        = analytic baseline; '
                             '"cnn_residual"  = A+ + CNN residual (PROJECT_STATE §7.15); '
                             '"cnn_direct"    = end-to-end CNN, NO A+ prior.')
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

    # ── Load model (mode=liteqp / cnn_*) ───────────────────────────────────
    # ``pipeline`` (sklearn MLP) and ``cnn_model`` (PyTorch nn.Module) are
    # mutually exclusive — only one is populated based on ``--mode``.
    pipeline = None
    cnn_model = None
    cnn_bundle = None
    if args.mode == "liteqp":
        if not args.model:
            raise SystemExit("--model required when --mode liteqp")
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
            K_grid = K_grids[fi]
            K_norm = K_grid / (np.median(K_grid[K_grid > 0]) + 1e-9
                               if (K_grid > 0).any() else 1.0)

            # 1) Analytic prior + exact rate-neutral projection
            #    NOTE: cnn_direct mode bypasses A+ entirely (delta_a is still
            #    computed for metadata purposes — useful for diagnostics).
            delta_a = compute_a_plus_delta(phi_norm, K_grid, qp, cfg)
            delta_a = project_rate_neutral_exact(delta_a, K_grid)

            if args.mode == "liteqp":
                # 2) Build features + predict residual (per-CTU MLP)
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
            else:  # a_plus mode
                delta_pred = delta_a

            # 3) Clip-aware exact projection — preserves rate-neutrality
            #    *after* clipping (plain exact projection breaks once a CTU
            #    saturates at a bound). Returns an already-clipped map.
            delta_pred = project_rate_neutral_clipped_exact(
                delta_pred, K_grid,
                delta_min=clip_lo, delta_max=clip_hi,
            )

            # Continuous-map ratio (should be 1.0 to machine precision).
            ratios_pre_round.append(rate_neutral_residual(delta_pred, K_grid))

            # 4) Final integer rounding (VVC requires integer δQP).
            #    NOTE: rounding can drift the rate-neutral ratio; we monitor it.
            delta_int = np.clip(np.rint(delta_pred),
                                cfg.delta_min_clip, cfg.delta_max_clip)
            ratios_post_round.append(rate_neutral_residual(delta_int, K_grid))

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
    }[args.mode]
    meta = {
        "method":        method_label,
        "mode":          args.mode,
        "model_path":    args.model if args.mode != "a_plus" else None,
        "saliency_dir":  str(sal_dir),
        "rate_npz":      str(Path(args.rate_npz).resolve()),
        "frames_dir":    str(frames_dir) if frames_dir else None,
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
        "feature_names": FEATURE_NAMES,
        # Rate-neutrality monitoring — the **continuous, clipped** map is
        # rate-neutral by construction (pre_round_mean ≈ 1.0). The integer
        # map drifts by ~0.5–2 % typically; flag if > 5 %.
        "rate_neutral_log": rate_ratio_log,
    }
    with open(out_root / "liteqp_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("Wrote metadata to %s", out_root / "liteqp_metadata.json")
    logger.info("Total maps written: %d", n_total_written)
    # Soft warning if any QP shows large post-round drift.
    for entry in rate_ratio_log:
        drift = abs(entry["rate_ratio_post_round_mean"] - 1.0)
        if drift > 0.05:
            logger.warning("Q_b=%d post-round rate drift = %.2f %% (>5 %%)",
                           entry["q_base"], drift * 100.0)


if __name__ == "__main__":
    main()
