"""Phase 3 Stage B.6 — Level 6 lightweight DNN fallback.

Reference: ``11_PHASE3_RESEARCH_PROTOCOL.md`` §2 (Level 6) and §4.4
(decision tree). This Level is invoked **only** if Levels 1–5 fail to
satisfy the pre-registered success criteria, to keep the design as
academically conservative as possible (parametric > DNN).

Architecture (≤ 2 000 parameters):

    features (K=7) → Linear(K, 16) → SiLU
                  → Linear(16, 8)  → SiLU
                  → Linear(8, 1)   → tanh × delta_max  → δ̂

Total trainable parameters ≈ 7·16 + 16 + 16·8 + 8 + 8·1 + 1 = 273.

Loss: weighted Lagrangian + total-variation regularizer + L2 weight
decay:

    L = L_lagrangian(δ̂) + λ_smooth · TV(δ̂) + λ_reg · ‖θ‖²

The TV term acts as a CTU-grid smoothness prior so the resulting δ-map
is encoder-friendly (avoids 1-CTU-wide oscillations).

Implementation note: we use a NumPy-only forward / backward pass so the
fallback runs without PyTorch on the analysis box. Optimization uses
``scipy.optimize.minimize`` (L-BFGS-B with finite differences); for
small θ this converges in seconds.

Run with::

    python -m phase2.phase3.train_dnn \
        --oracle ~/Minh/ipf/phase3_outputs/oracle/MOT17-04-DPM.jsonl \
        --output ~/Minh/ipf/phase3_outputs/fit/MOT17-04-DPM/dnn_l6.json \
        --lambda-task 7400 --lambda-smooth 0.05 --lambda-reg 1e-4
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from phase2.phase3.fit_parametric import (
    OracleData,
    fit_map_surrogate,
    fit_rate_surrogate,
    load_oracle,
)

logger = logging.getLogger("phase2.phase3.train_dnn")

DELTA_MAX = 4.0
DELTA_MIN = -8.0


# ---------------------------------------------------------------------------
# DNN structure: 2-hidden-layer MLP, swish/silu activation.
# ---------------------------------------------------------------------------

def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _unpack(theta: np.ndarray, K: int) -> Dict[str, np.ndarray]:
    """θ → named weight tensors. Layout below."""
    o = 0
    W1 = theta[o:o + K * 16].reshape(K, 16); o += K * 16
    b1 = theta[o:o + 16];                     o += 16
    W2 = theta[o:o + 16 * 8].reshape(16, 8);  o += 16 * 8
    b2 = theta[o:o + 8];                      o += 8
    W3 = theta[o:o + 8 * 1].reshape(8, 1);    o += 8
    b3 = theta[o:o + 1];                      o += 1
    assert o == len(theta), f"unpack consumed {o} of {len(theta)}"
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}


def _theta_size(K: int) -> int:
    return K * 16 + 16 + 16 * 8 + 8 + 8 * 1 + 1


def _forward(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    K = X.shape[1]
    p = _unpack(theta, K)
    h1 = _silu(X @ p["W1"] + p["b1"])
    h2 = _silu(h1 @ p["W2"] + p["b2"])
    z = h2 @ p["W3"] + p["b3"]
    # Centered around zero, then expand to [DELTA_MIN, DELTA_MAX].
    # Use a half-range scaling per side so positive and negative deltas can
    # both reach their respective bounds.
    pos = (DELTA_MAX) * np.tanh(np.clip(z, 0, None))
    neg = (DELTA_MIN) * np.tanh(np.clip(-z, 0, None))
    return (pos + neg).reshape(-1)


# ---------------------------------------------------------------------------
# Loss components.
# ---------------------------------------------------------------------------

def _lagrangian(delta: np.ndarray, oracle: OracleData,
                r_hat, map_hat, lambda_task: float) -> float:
    keys = list(zip(oracle.sequence.tolist(), oracle.q_base.tolist()))
    total, n = 0.0, 0
    for key in set(keys):
        seq, q = key
        mask = np.array([k == key for k in keys])
        r = r_hat(oracle.features[mask], delta[mask], q_base=q, seq=seq).mean()
        m = map_hat(oracle.features[mask], delta[mask], q_base=q, seq=seq).mean()
        total += float(r - lambda_task * m)
        n += 1
    return total / max(1, n)


def _tv_grid(delta: np.ndarray, frame_idx: np.ndarray,
             rows: np.ndarray, cols: np.ndarray) -> float:
    """Total variation on the inferred CTU grid, summed across frames.

    We rely on (frame_idx, rows, cols) columns of the oracle so that we
    can rebuild the per-frame δ surface for TV evaluation. If the meta
    columns are missing, returns 0 (regularization disabled).
    """
    if rows is None or cols is None or frame_idx is None:
        return 0.0
    tv = 0.0
    nf = 0
    for fi in np.unique(frame_idx):
        m = frame_idx == fi
        if m.sum() < 4:
            continue
        nrow = int(rows[m].max()) + 1
        ncol = int(cols[m].max()) + 1
        grid = np.full((nrow, ncol), np.nan)
        for r, c, d in zip(rows[m], cols[m], delta[m]):
            grid[int(r), int(c)] = d
        # Only finite cells contribute; skip NaN edges.
        a = np.diff(grid, axis=0)
        b = np.diff(grid, axis=1)
        tv += float(np.nansum(np.abs(a))) + float(np.nansum(np.abs(b)))
        nf += 1
    return tv / max(1, nf)


def make_loss(oracle: OracleData, r_hat, map_hat,
              lambda_task: float, lambda_smooth: float, lambda_reg: float,
              meta_rows: np.ndarray, meta_cols: np.ndarray, meta_frame: np.ndarray):
    K = oracle.features.shape[1]

    def loss(theta: np.ndarray) -> float:
        delta = _forward(theta, oracle.features)
        delta = np.clip(delta, DELTA_MIN, DELTA_MAX)
        L = _lagrangian(delta, oracle, r_hat, map_hat, lambda_task)
        if lambda_smooth > 0.0:
            L += lambda_smooth * _tv_grid(delta, meta_frame, meta_rows, meta_cols)
        if lambda_reg > 0.0:
            L += lambda_reg * float(np.sum(theta * theta))
        return L

    return loss, _theta_size(K)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def _load_meta_columns(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Best-effort recovery of (frame_idx, row, col) columns from JSONL/Parquet."""
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError:
            return np.array([]), np.array([]), np.array([])
        t = pq.read_table(path)
        d = t.to_pydict()
    else:
        d = {"frame_idx": [], "row": [], "col": []}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                d["frame_idx"].append(row.get("frame_idx", -1))
                d["row"].append(row.get("row", -1))
                d["col"].append(row.get("col", -1))
    return (np.asarray(d.get("frame_idx", []), dtype=np.int32),
            np.asarray(d.get("row", []), dtype=np.int32),
            np.asarray(d.get("col", []), dtype=np.int32))


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 Level 6 lightweight DNN trainer")
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lambda-task", type=float, default=7400.0)
    parser.add_argument("--lambda-smooth", type=float, default=0.05)
    parser.add_argument("--lambda-reg", type=float, default=1e-4)
    parser.add_argument("--restarts", type=int, default=4,
                        help="Number of random restarts for L-BFGS-B")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise SystemExit(f"scipy required: {exc}")

    oracle = load_oracle(Path(args.oracle))
    r_hat = fit_rate_surrogate(oracle)
    map_hat = fit_map_surrogate(oracle)
    meta_frame, meta_rows, meta_cols = _load_meta_columns(
        Path(args.oracle).expanduser().resolve()
    )

    loss, dim = make_loss(oracle, r_hat, map_hat,
                          args.lambda_task, args.lambda_smooth, args.lambda_reg,
                          meta_rows, meta_cols, meta_frame)
    logger.info("DNN θ dimension = %d  oracle N = %d", dim, len(oracle.delta))

    rng = np.random.default_rng(args.seed)
    best = None
    for trial in range(args.restarts):
        theta0 = 0.1 * rng.standard_normal(dim)
        try:
            res = minimize(loss, theta0, method="L-BFGS-B",
                           options={"maxiter": 500, "ftol": 1e-7})
        except Exception as exc:  # noqa: BLE001
            logger.warning("restart %d failed: %s", trial, exc)
            continue
        logger.info("restart=%d  loss=%.4f  converged=%s", trial, res.fun, res.success)
        if best is None or res.fun < best["fun"]:
            best = {"fun": float(res.fun), "theta": res.x.tolist(),
                    "converged": bool(res.success), "n_iter": int(res.nit)}

    if best is None:
        raise SystemExit("All DNN restarts failed.")

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "L6-mlp-2x16x8",
            "loss": best["fun"],
            "theta": best["theta"],
            "theta_dim": dim,
            "lambda_task": args.lambda_task,
            "lambda_smooth": args.lambda_smooth,
            "lambda_reg": args.lambda_reg,
            "delta_clip": [DELTA_MIN, DELTA_MAX],
            "feature_dim": oracle.features.shape[1],
            "n_oracle": int(len(oracle.delta)),
            "converged": best["converged"],
        }, f, indent=2)
    logger.info("Wrote %s  (loss=%.4f)", out_path, best["fun"])


if __name__ == "__main__":
    main()
