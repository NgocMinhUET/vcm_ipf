"""Phase 3 Stage C — LiteQP residual MLP trainer (with LOSO CV).

This module trains a tiny scikit-learn MLP that learns the **residual**
between the analytic RD-log A+ allocation and the per-CTU teacher label
δ_star (computed in :mod:`phase2.phase3.build_liteqp_dataset`).

Why residual learning?
----------------------
Plain end-to-end regression ``QP = MLP(features)`` is risky in our
small-data regime (3 sequences) because:

* mAP signals are noisy at the per-CTU scale.
* An unconstrained network can violate rate neutrality.
* Reviewers would rightfully ask "why does this CTU receive δ = −5?"
  with no principled justification.

Residual learning solves all three concerns simultaneously:

* The analytic A+ prior carries the rate-distortion logic.
* The MLP learns only a small correction (typically |r| ≤ 2 QP).
* The model is **interpretable** — you can plot the residual vs feature
  importance and see exactly which scenes the prior under- / over-shoots.

Cross-validation
----------------
Leave-One-Sequence-Out (LOSO) is mandatory because we only have three
sequences. Any method evaluated on the same sequences it was trained on
would be dismissed by reviewers as overfit. LOSO trains on 2 sequences
and reports residual MAE / r^2 on the held-out sequence.

Run with::

    python -m phase2.phase3.train_liteqp_regressor \
        --oracle  ~/Minh/ipf/phase3_outputs/oracle/all_liteqp.jsonl \
        --output  ~/Minh/ipf/phase3_outputs/fit/liteqp_mlp.joblib \
        --residual-bound 2.0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger("phase2.phase3.train_liteqp_regressor")


# ---------------------------------------------------------------------------
# Feature schema (must match apply_liteqp_model.py exactly)
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "phi",
    "K_c_norm",
    "q_base_norm",
    "sigma_y",
    "motion_proxy",
    "temporal_reliability",
    "prev_delta_norm",
    "phi_neighbor_mean",
    "phi_grad",
]


def featurize_row(r: dict) -> List[float]:
    return [
        float(r.get("phi", 0.0)),
        float(r.get("K_c_norm", 1.0)),
        float(r.get("q_base_norm", 0.0)),
        float(r.get("sigma_y", 0.0)),
        float(r.get("motion_proxy", 0.0)),
        float(r.get("temporal_reliability", 1.0)),
        float(r.get("prev_delta", 0.0)) / 8.0,
        float(r.get("phi_neighbor_mean", r.get("phi", 0.0))),
        float(r.get("phi_grad", 0.0)),
    ]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(paths: List[Path]) -> List[dict]:
    rows = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows


def build_arrays(rows: List[dict], residual_bound: float):
    """Return (X, y, sample_weight, sequence_id, qp_array)."""
    X = np.asarray([featurize_row(r) for r in rows], dtype=np.float32)
    delta_star = np.asarray([r["delta_star"] for r in rows], dtype=np.float64)
    delta_a    = np.asarray([r["delta_a_plus"] for r in rows], dtype=np.float64)
    y = np.clip(delta_star - delta_a, -residual_bound, residual_bound).astype(np.float32)
    # Sample weight: 0.2 + Φ + 0.1·K̃  (your spec §6.1)
    phi_v = np.asarray([float(r.get("phi", 0.0)) for r in rows], dtype=np.float32)
    k_v   = np.asarray([float(r.get("K_c_norm", 1.0)) for r in rows], dtype=np.float32)
    w = 0.2 + phi_v + 0.1 * k_v
    seq = np.asarray([r.get("sequence", "unknown") for r in rows])
    qp  = np.asarray([int(r.get("q_base", 0)) for r in rows])
    return X, y, w, seq, qp


# ---------------------------------------------------------------------------
# Trainer (sklearn MLP with our residual-bounded target).
# ---------------------------------------------------------------------------

def fit_one_split(X_tr, y_tr, w_tr, X_va, y_va, residual_bound: float,
                   max_iter: int, seed: int):
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.10,
            random_state=seed,
            n_iter_no_change=20,
        )),
    ])

    # sklearn's MLPRegressor does not support sample_weight natively, so
    # we approximate by oversampling high-weight rows to weighted-equal mass.
    sw_norm = w_tr / float(w_tr.mean())
    rng = np.random.default_rng(seed)
    p = sw_norm / sw_norm.sum()
    n_oversample = min(int(len(X_tr) * 1.2), 200_000)
    idx = rng.choice(len(X_tr), size=n_oversample, replace=True, p=p)
    pipeline.fit(X_tr[idx], y_tr[idx])

    # Bound residual to [-residual_bound, +residual_bound] post-hoc.
    pred_va = np.clip(pipeline.predict(X_va), -residual_bound, +residual_bound)
    mae = float(np.mean(np.abs(pred_va - y_va)))
    bias = float(np.mean(pred_va - y_va))
    return pipeline, mae, bias, pred_va


def bootstrap_ci(values: np.ndarray, n_boot: int = 500, alpha: float = 0.05,
                 stat: str = "mean", seed: int = 0) -> Dict[str, float]:
    """Percentile bootstrap CI on a 1-D residual array.

    Parameters
    ----------
    values
        Per-sample residuals (signed, NOT abs). MAE-bootstrap = bootstrap
        of ``mean(|x|)``; bias-bootstrap = bootstrap of ``mean(x)``.
    n_boot
        Number of bootstrap resamples.
    alpha
        Two-sided significance level (default 0.05 → 95 % CI).
    stat
        Either ``"mae"`` (bootstrap mean absolute error) or
        ``"bias"`` (bootstrap mean signed residual).
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return {"point": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"),
                "n_boot": 0}
    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = values[idx]
        if stat == "mae":
            boot[b] = float(np.mean(np.abs(sample)))
        elif stat == "bias":
            boot[b] = float(np.mean(sample))
        else:
            raise ValueError(f"unknown stat={stat}")
    if stat == "mae":
        point = float(np.mean(np.abs(values)))
    else:
        point = float(np.mean(values))
    return {
        "point": point,
        "ci_lo": float(np.quantile(boot, alpha / 2)),
        "ci_hi": float(np.quantile(boot, 1 - alpha / 2)),
        "n_boot": int(n_boot),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train LiteQP residual MLP")
    parser.add_argument("--oracle", nargs="+", required=True,
                        help="One or more JSONL files from build_liteqp_dataset.py")
    parser.add_argument("--output", required=True,
                        help="Output joblib bundle (model + meta)")
    parser.add_argument("--residual-bound", type=float, default=2.0)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--cv", choices=["loso", "none", "both"], default="both",
                        help='"loso" = report LOSO splits only; "none" = full-data fit only; '
                             '"both" = LOSO splits + full-data fit (default)')
    parser.add_argument("--report", default="",
                        help="Optional JSON report path (default: alongside output)")
    parser.add_argument("--n-bootstrap", type=int, default=500,
                        help="Bootstrap resamples for MAE/bias CI (default 500). "
                             "Set 0 to disable.")
    parser.add_argument("--bootstrap-alpha", type=float, default=0.05,
                        help="Two-sided significance level for bootstrap CI "
                             "(default 0.05 = 95%% CI).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    rows = load_jsonl([Path(p).expanduser().resolve() for p in args.oracle])
    if not rows:
        raise SystemExit("No rows loaded from --oracle")
    logger.info("Loaded %d rows from %d files", len(rows), len(args.oracle))

    X, y, w, seq, qp = build_arrays(rows, args.residual_bound)
    seqs = sorted(set(seq.tolist()))
    logger.info("Sequences in data: %s", seqs)
    logger.info("Residual stats — mean=%.3f std=%.3f min=%.2f max=%.2f",
                float(np.mean(y)), float(np.std(y)),
                float(np.min(y)), float(np.max(y)))

    report: Dict = {
        "n_rows": int(len(X)),
        "feature_names": FEATURE_NAMES,
        "residual_bound": float(args.residual_bound),
        "sequences": seqs,
    }

    # ── LOSO cross-validation ──────────────────────────────────────────────
    if args.cv in ("loso", "both") and len(seqs) >= 2:
        loso_results = []
        loso_pooled_residuals = []  # signed residuals from all hold-outs
        for held in seqs:
            mask_va = (seq == held)
            mask_tr = ~mask_va
            if mask_tr.sum() == 0:
                continue
            _, mae, bias, pred_va = fit_one_split(
                X[mask_tr], y[mask_tr], w[mask_tr],
                X[mask_va], y[mask_va],
                args.residual_bound, args.max_iter, args.seed,
            )
            residuals = (pred_va - y[mask_va]).astype(np.float64)
            loso_pooled_residuals.append(residuals)
            entry = {
                "held_out": held,
                "n_train": int(mask_tr.sum()),
                "n_val":   int(mask_va.sum()),
                "mae":     mae,
                "bias":    bias,
            }
            # Per-sequence bootstrap CI
            if args.n_bootstrap > 0:
                entry["mae_ci"] = bootstrap_ci(
                    residuals, n_boot=args.n_bootstrap,
                    alpha=args.bootstrap_alpha, stat="mae", seed=args.seed,
                )
                entry["bias_ci"] = bootstrap_ci(
                    residuals, n_boot=args.n_bootstrap,
                    alpha=args.bootstrap_alpha, stat="bias", seed=args.seed + 1,
                )
                logger.info(
                    "LOSO held-out=%-15s  MAE=%.3f [%.3f, %.3f]  "
                    "bias=%+.3f [%+.3f, %+.3f]  (n_va=%d)",
                    held, mae, entry["mae_ci"]["ci_lo"], entry["mae_ci"]["ci_hi"],
                    bias, entry["bias_ci"]["ci_lo"], entry["bias_ci"]["ci_hi"],
                    int(mask_va.sum()),
                )
            else:
                logger.info("LOSO held-out=%-15s  MAE=%.3f  bias=%+.3f  (n_va=%d)",
                            held, mae, bias, int(mask_va.sum()))
            loso_results.append(entry)
        report["loso"] = loso_results
        report["loso_mean_mae"] = float(np.mean([r["mae"] for r in loso_results]))

        # Pooled bootstrap across all hold-out residuals.
        if loso_pooled_residuals and args.n_bootstrap > 0:
            pooled = np.concatenate(loso_pooled_residuals)
            report["loso_pooled_mae_ci"] = bootstrap_ci(
                pooled, n_boot=args.n_bootstrap,
                alpha=args.bootstrap_alpha, stat="mae", seed=args.seed + 2,
            )
            report["loso_pooled_bias_ci"] = bootstrap_ci(
                pooled, n_boot=args.n_bootstrap,
                alpha=args.bootstrap_alpha, stat="bias", seed=args.seed + 3,
            )
            logger.info(
                "LOSO pooled  MAE=%.3f [%.3f, %.3f]  bias=%+.3f [%+.3f, %+.3f]",
                report["loso_pooled_mae_ci"]["point"],
                report["loso_pooled_mae_ci"]["ci_lo"],
                report["loso_pooled_mae_ci"]["ci_hi"],
                report["loso_pooled_bias_ci"]["point"],
                report["loso_pooled_bias_ci"]["ci_lo"],
                report["loso_pooled_bias_ci"]["ci_hi"],
            )

    # ── Full-data fit (used to deploy) ─────────────────────────────────────
    if args.cv in ("none", "both"):
        full_pipe, full_mae, full_bias, full_pred = fit_one_split(
            X, y, w, X, y,                 # train MAE on full set (deployment)
            args.residual_bound, args.max_iter, args.seed,
        )
        report["full_train_mae"] = full_mae
        report["full_train_bias"] = full_bias
        if args.n_bootstrap > 0:
            full_resid = (full_pred - y).astype(np.float64)
            report["full_train_mae_ci"]  = bootstrap_ci(
                full_resid, n_boot=args.n_bootstrap,
                alpha=args.bootstrap_alpha, stat="mae", seed=args.seed + 10,
            )
            report["full_train_bias_ci"] = bootstrap_ci(
                full_resid, n_boot=args.n_bootstrap,
                alpha=args.bootstrap_alpha, stat="bias", seed=args.seed + 11,
            )
        logger.info("Full-data fit  train MAE=%.3f  bias=%+.3f", full_mae, full_bias)
    else:
        full_pipe = None

    # ── Persist ────────────────────────────────────────────────────────────
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if full_pipe is not None:
        try:
            import joblib
        except ImportError as exc:
            raise SystemExit("joblib required to save model: %s" % exc) from exc
        joblib.dump({"model": full_pipe, "meta": report}, out_path)
        logger.info("Saved model bundle to %s", out_path)

    report_path = Path(args.report).expanduser().resolve() if args.report \
        else out_path.with_suffix(".report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote report %s", report_path)


if __name__ == "__main__":
    main()
