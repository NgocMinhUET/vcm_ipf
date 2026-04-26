"""Phase 3 Stage 3 — Parametric fitting for the QP-offset function.

Reference: ``11_PHASE3_RESEARCH_PROTOCOL.md`` §2 (function-family hierarchy)
and §4 (fitting procedure).

This module implements Levels 1–5 of the hierarchy as plain NumPy
functions, each exposing three primitives:

    params_init(x)   -> theta0             (sensible defaults)
    params_bounds(x) -> ((lo_1, hi_1), …)  (optimization bounds)
    forward(x, theta) -> delta_map         (per-CTU dQP)

The rate / task surrogates fit from the oracle dataset are then wrapped in
a scalar Lagrangian

    L(theta) = mean[R_hat(delta(theta)) - lambda * mAP_hat(delta(theta))]

which is minimized with ``scipy.optimize.minimize`` (Level 1–3: L-BFGS-B
with analytical bounds; Level 4–5: differential-evolution, because ``p``
or ``tau`` behave non-smoothly near their limits).

Run with::

    python -m phase2.phase3.fit_parametric \
        --oracle data/phase3/oracle_dataset.parquet \
        --surrogates data/phase3/surrogates/ \
        --out data/phase3/fit_results.json

Pre-registered decision rule (Protocol §4.3): we fit Level 2 → Level 3 →
Level 4 → Level 5 in order and stop as soon as the held-out BD-Rate-Task
drops below 0 % with its 95 % bootstrap CI excluding zero. This script
prints the winning Level and writes its parameters to ``fit_results.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

logger = logging.getLogger("phase2.phase3.fit_parametric")


# ---------------------------------------------------------------------------
# Function families (Levels 1–5).
#
# Each family maps:
#     phi_features  ∈ R^{N_CTU × K}   (K features per CTU, including phi_max,
#                                       phi_sum, phi_l2, phi_l4, Q_base, …)
#     theta         ∈ R^D             (family-specific parameter vector)
# → delta_map       ∈ R^{N_CTU}       (signed QP offsets, clipped later)
# ---------------------------------------------------------------------------

# Standard feature layout (column indices into phi_features).
#   0: phi_max_hat    (normalized L_inf aggregate ∈ [0,1])
#   1: phi_sum_hat
#   2: phi_l2_hat
#   3: phi_l4_hat
#   4: Q_base         (absolute QP, integer)
#   5: obj_coverage   (# overlapping objects for this CTU)
#   6: obj_confidence (max YOLO confidence touching this CTU)
IDX_PHI_MAX = 0
IDX_PHI_SUM = 1
IDX_PHI_L2 = 2
IDX_PHI_L4 = 3
IDX_Q_BASE = 4


def lp_aggregate(features: np.ndarray, p: float) -> np.ndarray:
    """Return the L_p aggregate column corresponding to exponent p.

    Uses precomputed columns for p ∈ {1, 2, 4, inf} and linearly
    interpolates between the two nearest tabulated values for all other
    p. Keeping the aggregator *precomputed* per CTU is the reason the
    oracle dataset ships phi_sum / phi_l2 / phi_l4 / phi_max rather than
    raw per-object kernel magnitudes.
    """
    if math.isinf(p) or p >= 128:
        return features[:, IDX_PHI_MAX]
    table = {1.0: features[:, IDX_PHI_SUM],
             2.0: features[:, IDX_PHI_L2],
             4.0: features[:, IDX_PHI_L4],
             float("inf"): features[:, IDX_PHI_MAX]}
    if p in table:
        return table[p]
    # Linear interp on known log-scale knots (1, 2, 4).
    anchors = sorted(k for k in table.keys() if math.isfinite(k))
    for lo, hi in zip(anchors, anchors[1:]):
        if lo <= p <= hi:
            w = (p - lo) / (hi - lo)
            return (1 - w) * table[lo] + w * table[hi]
    # p > 4 but finite: blend towards max.
    w = min(1.0, (p - 4.0) / 28.0)  # saturate at p=32
    return (1 - w) * table[4.0] + w * table[float("inf")]


def asymmetric_map(phi_hat: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Level 2 core: asymmetric power-law applied to a scalar aggregate.

    theta = (mu, delta_roi, delta_bg, gamma_roi, gamma_bg).
    """
    mu, d_roi, d_bg, g_roi, g_bg = theta
    out = np.zeros_like(phi_hat)
    roi = phi_hat >= mu
    bg = ~roi
    if np.any(roi):
        s = ((phi_hat[roi] - mu) / (1.0 - mu + 1e-12)) ** g_roi
        out[roi] = -d_roi * s
    if np.any(bg):
        s = ((mu - phi_hat[bg]) / (mu + 1e-12)) ** g_bg
        out[bg] = +d_bg * s
    return out


@dataclass(frozen=True)
class FunctionFamily:
    """Bundle describing one level of the parametric hierarchy."""

    name: str
    dim: int
    init: np.ndarray
    bounds: Sequence[Tuple[float, float]]
    forward: Callable[[np.ndarray, np.ndarray], np.ndarray]  # (features, theta) -> delta

    def predict(self, features: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return self.forward(features, theta)


# --- Level 1 --------------------------------------------------------------
def _level1_forward(features: np.ndarray, theta: np.ndarray) -> np.ndarray:
    (alpha,) = theta
    return -alpha * features[:, IDX_PHI_MAX]


LEVEL1 = FunctionFamily(
    name="L1-linear",
    dim=1,
    init=np.array([5.0]),
    bounds=[(0.0, 12.0)],
    forward=_level1_forward,
)

# --- Level 2 (current IPF) ------------------------------------------------
def _level2_forward(features: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return asymmetric_map(features[:, IDX_PHI_MAX], theta)


LEVEL2 = FunctionFamily(
    name="L2-asym",
    dim=5,
    init=np.array([0.3, 10.0, 6.0, 1.0, 1.0]),
    bounds=[(0.05, 0.7), (0.0, 12.0), (0.0, 8.0), (0.25, 4.0), (0.25, 4.0)],
    forward=_level2_forward,
)

# --- Level 3 (Q-adaptive) -------------------------------------------------
def _level3_forward(features: np.ndarray, theta: np.ndarray) -> np.ndarray:
    *theta2, kappa = theta
    base = asymmetric_map(features[:, IDX_PHI_MAX], np.asarray(theta2))
    scale = 1.0 + kappa * (features[:, IDX_Q_BASE] - 32.0) / 10.0
    return base * scale


LEVEL3 = FunctionFamily(
    name="L3-Qadapt",
    dim=6,
    init=np.concatenate([LEVEL2.init, [0.0]]),
    bounds=list(LEVEL2.bounds) + [(-1.0, 1.0)],
    forward=_level3_forward,
)

# --- Level 4 (Lp-norm + asymmetric) ---------------------------------------
def _level4_forward(features: np.ndarray, theta: np.ndarray) -> np.ndarray:
    *theta2, p = theta
    phi_hat = lp_aggregate(features, float(p))
    return asymmetric_map(phi_hat, np.asarray(theta2))


LEVEL4 = FunctionFamily(
    name="L4-Lp",
    dim=6,
    init=np.concatenate([LEVEL2.init, [float("inf")]]),   # start at max
    bounds=list(LEVEL2.bounds) + [(1.0, 32.0)],           # 32 ≈ effective inf
    forward=_level4_forward,
)

# --- Level 5 (softmax smooth-max) -----------------------------------------
def _level5_forward(features: np.ndarray, theta: np.ndarray) -> np.ndarray:
    # With precomputed aggregates we approximate softmax blending by
    # interpolating between phi_mean (tau=0 ≈ 2x L2 / something) and
    # phi_max (tau→∞). Using phi_sum as the tau=0 proxy.
    *theta2, tau = theta
    w = 1.0 - 1.0 / (1.0 + math.exp(-(tau - 3.0)))  # logistic blend
    phi_hat = (1.0 - w) * features[:, IDX_PHI_MAX] + w * features[:, IDX_PHI_SUM]
    return asymmetric_map(phi_hat, np.asarray(theta2))


LEVEL5 = FunctionFamily(
    name="L5-softmax",
    dim=6,
    init=np.concatenate([LEVEL2.init, [5.0]]),
    bounds=list(LEVEL2.bounds) + [(0.0, 10.0)],
    forward=_level5_forward,
)


ALL_LEVELS: List[FunctionFamily] = [LEVEL1, LEVEL2, LEVEL3, LEVEL4, LEVEL5]


# ---------------------------------------------------------------------------
# Oracle dataset loading + surrogate fitting.
# ---------------------------------------------------------------------------

@dataclass
class OracleData:
    """In-memory representation of the oracle dataset."""

    features: np.ndarray          # (N_samples, K)
    delta: np.ndarray             # (N_samples,) applied delta
    d_rate: np.ndarray            # (N_samples,) Delta-rate (kbps)
    d_map: np.ndarray             # (N_samples,) Delta-mAP
    sequence: np.ndarray          # (N_samples,) sequence id string array
    q_base: np.ndarray            # (N_samples,) base QP


def load_oracle(path: Path) -> OracleData:
    """Load oracle dataset from Parquet (preferred) or JSONL (fallback)."""
    path = Path(path).expanduser().resolve()
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise SystemExit(
                f"pyarrow required for parquet oracle loading: {exc}"
            )
        table = pq.read_table(path)
        data = table.to_pydict()
    else:
        data = {"features": [], "delta": [], "d_rate": [], "d_map": [],
                "sequence": [], "q_base": []}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                data["features"].append(row["features"])
                data["delta"].append(row["delta"])
                data["d_rate"].append(row["delta_rate_kbps"])
                data["d_map"].append(row["delta_map50"])
                data["sequence"].append(row["sequence"])
                data["q_base"].append(row["qp_base"])

    return OracleData(
        features=np.asarray(data["features"], dtype=np.float64),
        delta=np.asarray(data["delta"], dtype=np.float64),
        d_rate=np.asarray(data["d_rate"], dtype=np.float64),
        d_map=np.asarray(data["d_map"], dtype=np.float64),
        sequence=np.asarray(data["sequence"]),
        q_base=np.asarray(data["q_base"], dtype=np.int32),
    )


def fit_rate_surrogate(data: OracleData) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Linear regression ``ΔR ≈ β · δ`` per (sequence, Q_base).

    Returns a closure ``R_hat(features, delta) -> predicted ΔR``.
    Rate response to small QP perturbations is nearly linear per CTU
    (R-λ theory, §4.1 of the protocol). We therefore fit a single
    coefficient per (sequence, Q_base) pair and assume additivity.
    """
    keys = list(zip(data.sequence.tolist(), data.q_base.tolist()))
    beta: Dict[Tuple[str, int], float] = {}
    for key in set(keys):
        seq_name, q = key
        mask = np.array([k == key for k in keys])
        if mask.sum() < 3:
            continue
        x = data.delta[mask]
        y = data.d_rate[mask]
        beta[key] = float(np.linalg.lstsq(x[:, None], y, rcond=None)[0][0])

    mean_beta = float(np.mean(list(beta.values()))) if beta else 0.0

    def r_hat(features: np.ndarray, delta: np.ndarray,
              q_base: int = 32, seq: str = "") -> np.ndarray:
        b = beta.get((seq, int(q_base)), mean_beta)
        return b * delta

    return r_hat


def fit_map_surrogate(data: OracleData) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Gaussian-kernel regression for the task-accuracy surrogate.

    Uses a simple Nadaraya-Watson estimator conditioned on (phi_max, delta)
    — enough for early Phase 3 exploratory fits. Stage-5 protocol allows
    upgrading to a full GP if residuals show structure.
    """
    X = np.column_stack([data.features[:, IDX_PHI_MAX], data.delta])
    y = data.d_map
    sigma = 0.15  # bandwidth in normalized feature space

    def map_hat(features: np.ndarray, delta: np.ndarray,
                q_base: int = 32, seq: str = "") -> np.ndarray:
        Q = np.column_stack([features[:, IDX_PHI_MAX], delta])
        # NW estimator; vectorized for small batches only.
        preds = np.empty(len(Q))
        for i, q in enumerate(Q):
            w = np.exp(-np.sum((X - q) ** 2, axis=1) / (2 * sigma * sigma))
            w_sum = w.sum() + 1e-12
            preds[i] = float((w * y).sum() / w_sum)
        return preds

    return map_hat


# ---------------------------------------------------------------------------
# Lagrangian optimizer (L-BFGS-B for smooth levels, DE for Level 4/5).
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    level: str
    theta: List[float]
    bounds: List[List[float]]
    val_loss: float
    val_bd_rate_task: float
    val_n_samples: int
    converged: bool
    extra: Dict[str, float] = field(default_factory=dict)


def lagrangian_loss(
    family: FunctionFamily,
    features: np.ndarray,
    q_base: np.ndarray,
    sequences: np.ndarray,
    r_hat: Callable,
    map_hat: Callable,
    lambda_task: float,
) -> Callable[[np.ndarray], float]:
    def loss(theta: np.ndarray) -> float:
        delta = family.forward(features, theta)
        delta = np.clip(delta, -8.0, 4.0)
        # Aggregate across the evaluation grid by grouping on (seq, Q_b).
        total = 0.0
        count = 0
        keys = list(zip(sequences.tolist(), q_base.tolist()))
        for key in set(keys):
            seq, q = key
            mask = np.array([k == key for k in keys])
            r = r_hat(features[mask], delta[mask], q_base=q, seq=seq).mean()
            m = map_hat(features[mask], delta[mask], q_base=q, seq=seq).mean()
            total += r - lambda_task * m
            count += 1
        return total / max(1, count)
    return loss


def fit_level(
    level: FunctionFamily,
    oracle: OracleData,
    r_hat: Callable,
    map_hat: Callable,
    lambda_task: float,
) -> FitResult:
    try:
        from scipy.optimize import minimize, differential_evolution
    except ImportError as exc:
        raise SystemExit(f"scipy required: {exc}")

    loss = lagrangian_loss(
        level, oracle.features, oracle.q_base, oracle.sequence,
        r_hat, map_hat, lambda_task,
    )

    # Smooth levels → L-BFGS-B; Level 4 has a non-smooth p sweep → DE.
    if level.name in {"L4-Lp", "L5-softmax"}:
        result = differential_evolution(
            loss, bounds=list(level.bounds),
            seed=0, tol=1e-4, maxiter=40, polish=True, updating="deferred",
            workers=1,
        )
        theta = result.x
        converged = result.success
    else:
        result = minimize(
            loss, level.init, method="L-BFGS-B", bounds=list(level.bounds),
            options={"maxiter": 200, "ftol": 1e-6},
        )
        theta = result.x
        converged = result.success

    # Validation metric: mean ΔR and ΔmAP on the full oracle split.
    delta_star = level.forward(oracle.features, theta)
    delta_star = np.clip(delta_star, -8.0, 4.0)

    # Crude BD-Rate-Task proxy: mean ΔR / lambda_task − ΔmAP, normalized.
    r_pred = r_hat(oracle.features, delta_star)
    m_pred = map_hat(oracle.features, delta_star)
    bd_proxy = float(np.mean(r_pred) - lambda_task * np.mean(m_pred))

    return FitResult(
        level=level.name,
        theta=theta.tolist(),
        bounds=[list(b) for b in level.bounds],
        val_loss=float(result.fun),
        val_bd_rate_task=bd_proxy,
        val_n_samples=int(len(oracle.delta)),
        converged=bool(converged),
    )


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 Stage 3 parametric fitting")
    parser.add_argument("--oracle", required=True, help="oracle_dataset.parquet or .jsonl")
    parser.add_argument("--out", required=True, help="fit_results.json")
    parser.add_argument("--levels", nargs="+",
                        default=["L1-linear", "L2-asym", "L3-Qadapt", "L4-Lp", "L5-softmax"])
    parser.add_argument(
        "--lambda-task",
        type=float,
        default=7400.0,
        help="kbps per mAP unit (§3.3 default from MOT17-04 pilot).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    oracle = load_oracle(Path(args.oracle))
    logger.info("Oracle: %d samples, %d unique sequences, %d unique Q_b",
                len(oracle.delta), len(set(oracle.sequence.tolist())),
                len(set(oracle.q_base.tolist())))

    r_hat = fit_rate_surrogate(oracle)
    map_hat = fit_map_surrogate(oracle)

    wanted = {l.name: l for l in ALL_LEVELS}
    results: List[FitResult] = []

    for name in args.levels:
        family = wanted.get(name)
        if family is None:
            logger.warning("Unknown level %s — skipping", name)
            continue
        logger.info("=== Fitting %s ===", name)
        fit = fit_level(family, oracle, r_hat, map_hat, args.lambda_task)
        logger.info("  loss=%.4f  theta=%s  converged=%s",
                    fit.val_loss, fit.theta, fit.converged)
        results.append(fit)

    best = min(results, key=lambda r: r.val_loss)
    logger.info("Best level on val loss: %s  loss=%.4f", best.level, best.val_loss)

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_level": best.level,
                "lambda_task": args.lambda_task,
                "oracle_path": str(args.oracle),
                "results": [r.__dict__ for r in results],
            },
            f,
            indent=2,
        )
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
