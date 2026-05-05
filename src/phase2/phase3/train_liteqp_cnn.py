"""Train the LiteQP-CNN spatial residual / direct model.

Mirrors the design of ``train_liteqp_regressor.py`` but operates on
9×15 spatial samples instead of per-CTU flat rows.

LOSO (Leave-One-Sequence-Out) cross-validation + percentile bootstrap CI
are *retained* for parity with the MLP trainer — same evaluation
protocol means we can compare per-pixel MAE / bias directly.

Outputs:
    <out>.pt           PyTorch bundle (state_dict + scalar config)
    <out>.report.json  same keys as MLP report (loso, full_train_mae, …)
                       plus CNN-specific entries (n_params, output_mode,
                       per-component loss history).

Run with:
    python -m phase2.phase3.train_liteqp_cnn \
        --oracle ~/Minh/ipf/phase3_outputs/oracle_liteqp_v5r/all_liteqp.jsonl \
        --output ~/Minh/ipf/phase3_outputs/fit/liteqp_cnn_residual.pt \
        --output-mode residual --residual-bound 2.0 \
        --epochs 100 --batch-size 16 --lr 1e-3 \
        --device cuda:0 \
        --cv both --n-bootstrap 500 --bootstrap-alpha 0.05
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logger = logging.getLogger("phase2.phase3.train_liteqp_cnn")


# ---------------------------------------------------------------------------
# Bootstrap CI (identical interface to the MLP trainer)
# ---------------------------------------------------------------------------

def bootstrap_ci(values: np.ndarray, n_boot: int = 500, alpha: float = 0.05,
                 stat: str = "mae", seed: int = 0) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return {"point": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "n_boot": 0}
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v = values[idx]
        boots[b] = float(np.mean(np.abs(v))) if stat == "mae" \
            else float(np.mean(v))
    point = float(np.mean(np.abs(values))) if stat == "mae" \
        else float(np.mean(values))
    return {
        "point":  point,
        "ci_lo":  float(np.quantile(boots, alpha / 2.0)),
        "ci_hi":  float(np.quantile(boots, 1.0 - alpha / 2.0)),
        "n_boot": int(n_boot),
    }


# ---------------------------------------------------------------------------
# One training pass (used for LOSO folds and the final full-data fit)
# ---------------------------------------------------------------------------

def _evaluate_residuals(model, samples_va, output_mode: str,
                          residual_bound: float, device: str):
    """Per-pixel signed residual array on the validation samples."""
    import torch
    model.eval()
    out: List[np.ndarray] = []
    with torch.no_grad():
        for s in samples_va:
            x = torch.from_numpy(s.features).float().unsqueeze(0).to(device)
            pred = model(x).squeeze(0).squeeze(0).cpu().numpy()
            if output_mode == "residual":
                pred = np.clip(pred, -residual_bound, +residual_bound)
                target = s.target_residual[0]
            else:
                target = s.target_delta[0]
            out.append((pred - target).astype(np.float64).ravel())
    return np.concatenate(out) if out else np.array([], dtype=np.float64)


def fit_one_split(samples_tr, samples_va, *,
                   output_mode: str, output_bound: float,
                   residual_bound: float,
                   epochs: int, batch_size: int, lr: float,
                   weight_decay: float,
                   alpha_rnp: float, alpha_tv: float, alpha_bound: float,
                   huber_delta: float, delta_max: float,
                   seed: int, device: str):
    """Train one CNN. Returns (model, mae, bias, val_residuals, history)."""
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)

    from phase2.phase3.liteqp_cnn import (
        build_cnn_model, make_torch_dataset, compute_loss, LossWeights,
        per_pixel_sample_weight, count_params,
    )

    model = build_cnn_model(
        output_mode=output_mode, output_bound=output_bound,
    ).to(device)
    n_params = count_params(model)
    logger.info("CNN built — output_mode=%s, output_bound=%.2f, n_params=%d",
                output_mode, output_bound, n_params)

    weights = LossWeights(
        huber_delta=huber_delta, alpha_rnp=alpha_rnp, alpha_tv=alpha_tv,
        alpha_bound=alpha_bound, delta_max=delta_max,
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max(1, epochs))

    ds_tr = make_torch_dataset(samples_tr, output_mode)
    loader = torch.utils.data.DataLoader(
        ds_tr, batch_size=batch_size, shuffle=True, drop_last=False)

    history: List[Dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        epoch_acc = {"huber": 0.0, "rnp": 0.0, "tv": 0.0, "bound": 0.0,
                     "total": 0.0, "n": 0}
        for batch in loader:
            x = batch["x"].to(device)
            batch_dev = {
                "x":            x,
                "y":            batch["y"].to(device),
                "delta_a_plus": batch["delta_a_plus"].to(device),
                "K_grid":       batch["K_grid"].to(device),
            }
            sw = per_pixel_sample_weight(x)
            pred = model(x)
            loss, parts = compute_loss(pred, batch_dev, output_mode,
                                         weights, sample_weight=sw)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimiser.step()
            for k in ("huber", "rnp", "tv", "bound", "total"):
                epoch_acc[k] += float(parts[k]) * x.shape[0]
            epoch_acc["n"] += x.shape[0]
        scheduler.step()
        # Average over the epoch.
        ep_record = {k: epoch_acc[k] / max(1, epoch_acc["n"])
                     for k in ("huber", "rnp", "tv", "bound", "total")}
        ep_record["epoch"] = int(epoch)
        history.append(ep_record)
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            logger.info("  epoch=%3d  loss=%.4f  huber=%.4f  rnp=%.4f  tv=%.4f",
                        epoch, ep_record["total"], ep_record["huber"],
                        ep_record["rnp"], ep_record["tv"])

    val_residuals = _evaluate_residuals(model, samples_va, output_mode,
                                          residual_bound, device)
    if val_residuals.size:
        mae  = float(np.mean(np.abs(val_residuals)))
        bias = float(np.mean(val_residuals))
    else:
        mae, bias = float("nan"), float("nan")
    return model, mae, bias, val_residuals, history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the LiteQP-CNN spatial model")
    parser.add_argument("--oracle", required=True, nargs="+",
                        help="One or more all_liteqp.jsonl files to merge")
    parser.add_argument("--output", required=True,
                        help="Output bundle path (will end in .pt)")
    parser.add_argument("--report", default="")

    # Architecture
    parser.add_argument("--output-mode", choices=["residual", "direct"],
                        required=True)
    parser.add_argument("--output-bound", type=float, default=None,
                        help="Default = residual_bound for residual mode, "
                             "or 8.0 for direct mode.")
    parser.add_argument("--residual-bound", type=float, default=2.0,
                        help="Cap on training residuals (residual mode).")

    # Training
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--seed",       type=int,   default=20260506)

    # Loss weights
    parser.add_argument("--huber-delta", type=float, default=0.5)
    parser.add_argument("--alpha-rnp",   type=float, default=0.10)
    parser.add_argument("--alpha-tv",    type=float, default=0.005)
    parser.add_argument("--alpha-bound", type=float, default=0.10)
    parser.add_argument("--delta-max",   type=float, default=8.0)

    # CV
    parser.add_argument("--cv", choices=["loso", "full", "both"], default="both")
    parser.add_argument("--n-bootstrap",     type=int,   default=500)
    parser.add_argument("--bootstrap-alpha", type=float, default=0.05)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Output bound default depends on mode.
    if args.output_bound is None:
        args.output_bound = (args.residual_bound if args.output_mode == "residual"
                              else float(args.delta_max))

    # ── Load data ──────────────────────────────────────────────────────────
    from phase2.phase3.liteqp_cnn import (
        load_spatial_dataset, save_bundle, CNNBundle, count_params,
        CNN_INPUT_CHANNELS, CNN_N_INPUT,
    )
    paths = [Path(p).expanduser().resolve() for p in args.oracle]
    samples = load_spatial_dataset(paths, residual_bound=args.residual_bound)
    if not samples:
        raise SystemExit("No spatial samples loaded — check --oracle paths.")
    seqs = sorted({s.sequence for s in samples})
    logger.info("Loaded %d spatial samples from %d JSONL file(s); sequences: %s",
                len(samples), len(paths), seqs)
    logger.info("Feature schema (%d ch): %s", CNN_N_INPUT, CNN_INPUT_CHANNELS)

    report: Dict = {
        "schema_version":  1,
        "n_samples":       len(samples),
        "n_input_channels": CNN_N_INPUT,
        "feature_names":   list(CNN_INPUT_CHANNELS),
        "output_mode":     args.output_mode,
        "output_bound":    float(args.output_bound),
        "residual_bound":  float(args.residual_bound),
        "sequences":       seqs,
        "training": {
            "epochs":      int(args.epochs),
            "batch_size":  int(args.batch_size),
            "lr":          float(args.lr),
            "weight_decay": float(args.weight_decay),
            "seed":        int(args.seed),
            "device":      str(args.device),
            "huber_delta": float(args.huber_delta),
            "alpha_rnp":   float(args.alpha_rnp),
            "alpha_tv":    float(args.alpha_tv),
            "alpha_bound": float(args.alpha_bound),
        },
    }

    # ── LOSO ───────────────────────────────────────────────────────────────
    loso_pooled_residuals: List[np.ndarray] = []
    if args.cv in ("loso", "both") and len(seqs) >= 2:
        loso_results = []
        for held in seqs:
            samples_va = [s for s in samples if s.sequence == held]
            samples_tr = [s for s in samples if s.sequence != held]
            if not samples_tr or not samples_va:
                continue
            logger.info("[LOSO] held-out %s   (n_train=%d  n_val=%d)",
                        held, len(samples_tr), len(samples_va))
            _, mae, bias, val_resid, _hist = fit_one_split(
                samples_tr, samples_va,
                output_mode=args.output_mode,
                output_bound=float(args.output_bound),
                residual_bound=args.residual_bound,
                epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, weight_decay=args.weight_decay,
                alpha_rnp=args.alpha_rnp, alpha_tv=args.alpha_tv,
                alpha_bound=args.alpha_bound,
                huber_delta=args.huber_delta, delta_max=args.delta_max,
                seed=args.seed, device=args.device,
            )
            loso_pooled_residuals.append(val_resid)
            entry = {
                "held_out": held,
                "n_train":  len(samples_tr),
                "n_val":    len(samples_va),
                "mae":      mae,
                "bias":     bias,
            }
            if args.n_bootstrap > 0 and val_resid.size:
                entry["mae_ci"] = bootstrap_ci(
                    val_resid, n_boot=args.n_bootstrap,
                    alpha=args.bootstrap_alpha, stat="mae", seed=args.seed)
                entry["bias_ci"] = bootstrap_ci(
                    val_resid, n_boot=args.n_bootstrap,
                    alpha=args.bootstrap_alpha, stat="bias", seed=args.seed + 1)
                logger.info("[LOSO] %-15s  MAE=%.3f [%.3f, %.3f]  "
                            "bias=%+.3f [%+.3f, %+.3f]",
                            held, mae, entry["mae_ci"]["ci_lo"],
                            entry["mae_ci"]["ci_hi"], bias,
                            entry["bias_ci"]["ci_lo"], entry["bias_ci"]["ci_hi"])
            else:
                logger.info("[LOSO] %-15s  MAE=%.3f  bias=%+.3f", held, mae, bias)
            loso_results.append(entry)
        report["loso"] = loso_results
        if loso_results:
            report["loso_mean_mae"] = float(np.mean([r["mae"] for r in loso_results]))
        if loso_pooled_residuals and args.n_bootstrap > 0:
            pooled = np.concatenate(loso_pooled_residuals)
            report["loso_pooled_mae_ci"] = bootstrap_ci(
                pooled, n_boot=args.n_bootstrap,
                alpha=args.bootstrap_alpha, stat="mae", seed=args.seed + 2)
            report["loso_pooled_bias_ci"] = bootstrap_ci(
                pooled, n_boot=args.n_bootstrap,
                alpha=args.bootstrap_alpha, stat="bias", seed=args.seed + 3)
            logger.info("[LOSO] pooled  MAE=%.3f [%.3f, %.3f]  bias=%+.3f [%+.3f, %+.3f]",
                        report["loso_pooled_mae_ci"]["point"],
                        report["loso_pooled_mae_ci"]["ci_lo"],
                        report["loso_pooled_mae_ci"]["ci_hi"],
                        report["loso_pooled_bias_ci"]["point"],
                        report["loso_pooled_bias_ci"]["ci_lo"],
                        report["loso_pooled_bias_ci"]["ci_hi"])

    # ── Full-data fit (deployed model) ─────────────────────────────────────
    full_model = None
    full_history = None
    if args.cv in ("full", "both"):
        full_model, full_mae, full_bias, full_resid, full_history = fit_one_split(
            samples, samples,                  # train MAE on full set
            output_mode=args.output_mode,
            output_bound=float(args.output_bound),
            residual_bound=args.residual_bound,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, weight_decay=args.weight_decay,
            alpha_rnp=args.alpha_rnp, alpha_tv=args.alpha_tv,
            alpha_bound=args.alpha_bound,
            huber_delta=args.huber_delta, delta_max=args.delta_max,
            seed=args.seed, device=args.device,
        )
        report["full_train_mae"]  = full_mae
        report["full_train_bias"] = full_bias
        report["n_params"]        = count_params(full_model)
        if args.n_bootstrap > 0:
            report["full_train_mae_ci"]  = bootstrap_ci(
                full_resid, n_boot=args.n_bootstrap,
                alpha=args.bootstrap_alpha, stat="mae", seed=args.seed + 10)
            report["full_train_bias_ci"] = bootstrap_ci(
                full_resid, n_boot=args.n_bootstrap,
                alpha=args.bootstrap_alpha, stat="bias", seed=args.seed + 11)
        if full_history:
            report["full_train_history"] = full_history
        logger.info("Full-data fit  MAE=%.3f  bias=%+.3f  n_params=%d",
                    full_mae, full_bias, report["n_params"])

    # ── Persist bundle + report ────────────────────────────────────────────
    out_path = Path(args.output).expanduser().resolve()
    if out_path.suffix not in (".pt", ".bin"):
        out_path = out_path.with_suffix(".pt")
    if full_model is not None:
        bundle = CNNBundle(
            state_dict=full_model.state_dict(),
            output_mode=args.output_mode,
            output_bound=float(args.output_bound),
            n_input_channels=CNN_N_INPUT,
            hidden=16, n_groups=4, skip_phi_index=0,
            schema_version=1,
            feature_names=list(CNN_INPUT_CHANNELS),
            train_meta={
                "full_train_mae":  float(report.get("full_train_mae",  float("nan"))),
                "full_train_bias": float(report.get("full_train_bias", float("nan"))),
                "loso_mean_mae":   float(report.get("loso_mean_mae",   float("nan"))),
            },
        )
        save_bundle(bundle, out_path)

    report_path = Path(args.report).expanduser().resolve() if args.report \
        else out_path.with_suffix(".report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote report to %s", report_path)


if __name__ == "__main__":
    main()
