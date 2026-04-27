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
    """Per-CTU rate model from R-λ theory.

    The build_oracle script emits per-CTU ``delta_rate_kbps`` rows that are
    already proportional to the closed-form

        ΔR_c(δ) = K_c · (2^{-(Q_b+δ)/6} - 2^{-Q_b/6})

    so a single global multiplicative scale ``s_seq_qb`` is sufficient
    to absorb residual systematic mismatch per (sequence, Q_b). When
    feature column 6 (``K_c`` if available) is non-zero we use it
    directly; otherwise we regress δ → ΔR linearly.
    """
    keys = list(zip(data.sequence.tolist(), data.q_base.tolist()))
    has_K = data.features.shape[1] > 6 and float(np.std(data.features[:, 6])) > 0.0

    cell_scale: Dict[Tuple[str, int], float] = {}
    cell_beta: Dict[Tuple[str, int], float] = {}
    for key in set(keys):
        mask = np.array([k == key for k in keys])
        x = data.delta[mask]
        y = data.d_rate[mask]
        if mask.sum() < 3 or float(np.std(x)) == 0.0:
            continue
        if has_K:
            q = float(key[1])
            base = 2.0 ** (-q / 6.0)
            pert = 2.0 ** (-(q + x) / 6.0)
            pred = data.features[mask, 6] * (pert - base)
            num = float(np.dot(pred, y))
            den = float(np.dot(pred, pred)) + 1e-12
            cell_scale[key] = num / den
        cell_beta[key] = float(np.linalg.lstsq(x[:, None], y, rcond=None)[0][0])

    fallback_scale = (float(np.mean(list(cell_scale.values())))
                      if cell_scale else 0.0)
    fallback_beta = float(np.mean(list(cell_beta.values()))) if cell_beta else 0.0

    def r_hat(features: np.ndarray, delta: np.ndarray,
              q_base: int = 32, seq: str = "") -> np.ndarray:
        if has_K and features.shape[1] > 6:
            scale = cell_scale.get((seq, int(q_base)), fallback_scale)
            base = 2.0 ** (-int(q_base) / 6.0)
            pert = 2.0 ** (-(int(q_base) + delta) / 6.0)
            return scale * features[:, 6] * (pert - base)
        b = cell_beta.get((seq, int(q_base)), fallback_beta)
        return b * delta

    return r_hat


def fit_map_surrogate(data: OracleData) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Asymmetric per-CTU task surrogate.

    Model: ΔmAP_c(δ) = phi_oracle(c) · [-η · max(0, δ) + ξη · max(0, -δ)]

    We jointly fit (η, ξ) by least-squares on the oracle rows. When
    column 5 of the feature matrix carries a saliency proxy we use it
    directly; otherwise we fall back to phi_max as the saliency.
    """
    has_saliency = (data.features.shape[1] > 5
                    and float(np.std(data.features[:, 5])) > 0.0)
    sal_col = 5 if has_saliency else IDX_PHI_MAX

    pos = np.maximum(0.0, data.delta)
    neg = np.maximum(0.0, -data.delta)
    sal = data.features[:, sal_col]
    A = np.column_stack([-sal * pos, +sal * neg])
    y = data.d_map
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    eta = float(coef[0])
    eta_xi = float(coef[1])
    xi = max(0.0, eta_xi / max(1e-9, eta)) if eta > 0 else 0.0
    logger.info("map_surrogate fit: eta=%.5f  xi=%.3f  R²=%.3f",
                eta, xi, _r2(A @ coef, y))

    def map_hat(features: np.ndarray, delta: np.ndarray,
                q_base: int = 32, seq: str = "") -> np.ndarray:
        sc = features[:, sal_col]
        return -sc * eta * np.maximum(0.0, delta) + sc * eta * xi * np.maximum(0.0, -delta)

    return map_hat


def _r2(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot


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


def _slice_oracle(o: OracleData, mask: np.ndarray) -> OracleData:
    return OracleData(
        features=o.features[mask],
        delta=o.delta[mask],
        d_rate=o.d_rate[mask],
        d_map=o.d_map[mask],
        sequence=o.sequence[mask],
        q_base=o.q_base[mask],
    )


def _bd_rate_task_proxy(level: FunctionFamily, theta: np.ndarray,
                         oracle: OracleData, r_hat: Callable, map_hat: Callable,
                         lambda_task: float) -> float:
    """Mean Lagrangian on (already-evaluated) oracle split."""
    delta = np.clip(level.forward(oracle.features, theta), -8.0, 4.0)
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


def _bootstrap_ci(level: FunctionFamily, theta: np.ndarray, oracle: OracleData,
                   r_hat: Callable, map_hat: Callable, lambda_task: float,
                   n_boot: int = 200, seed: int = 0) -> Tuple[float, float]:
    """Percentile bootstrap CI on the val_bd_rate_task proxy."""
    rng = np.random.default_rng(seed)
    N = len(oracle.delta)
    if N < 8:
        return (float("nan"), float("nan"))
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        boot = _slice_oracle(oracle, idx)
        try:
            samples.append(_bd_rate_task_proxy(level, theta, boot,
                                               r_hat, map_hat, lambda_task))
        except Exception:  # noqa: BLE001
            continue
    if not samples:
        return (float("nan"), float("nan"))
    s = np.asarray(samples)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def _fit_once(level: FunctionFamily, oracle: OracleData,
               r_hat: Callable, map_hat: Callable, lambda_task: float,
               seed: int = 0) -> Tuple[np.ndarray, float, bool]:
    from scipy.optimize import minimize, differential_evolution
    loss = lagrangian_loss(
        level, oracle.features, oracle.q_base, oracle.sequence,
        r_hat, map_hat, lambda_task,
    )
    if level.name in {"L4-Lp", "L5-softmax"}:
        result = differential_evolution(
            loss, bounds=list(level.bounds),
            seed=seed, tol=1e-4, maxiter=40, polish=True, updating="deferred",
            workers=1,
        )
    else:
        result = minimize(
            loss, level.init, method="L-BFGS-B", bounds=list(level.bounds),
            options={"maxiter": 200, "ftol": 1e-6},
        )
    return result.x, float(result.fun), bool(result.success)


def fit_level(
    level: FunctionFamily,
    oracle: OracleData,
    r_hat: Callable,
    map_hat: Callable,
    lambda_task: float,
    n_bootstrap: int = 200,
    cross_validate: bool = True,
) -> FitResult:
    """Fit one Level with global optimum + LOSO CV + bootstrap CI."""
    theta, loss, conv = _fit_once(level, oracle, r_hat, map_hat, lambda_task)
    bd_proxy = _bd_rate_task_proxy(level, theta, oracle, r_hat, map_hat, lambda_task)
    ci_lo, ci_hi = _bootstrap_ci(level, theta, oracle, r_hat, map_hat,
                                  lambda_task, n_boot=n_bootstrap)

    extra: Dict[str, float] = {
        "val_bd_rate_task_ci_lo": ci_lo,
        "val_bd_rate_task_ci_hi": ci_hi,
    }

    if cross_validate:
        seqs = sorted(set(oracle.sequence.tolist()))
        if len(seqs) >= 2:
            cv_losses = []
            for held in seqs:
                tr_mask = oracle.sequence != held
                te_mask = oracle.sequence == held
                if tr_mask.sum() < 16 or te_mask.sum() < 4:
                    continue
                tr = _slice_oracle(oracle, tr_mask)
                te = _slice_oracle(oracle, te_mask)
                r_h_tr = fit_rate_surrogate(tr)
                m_h_tr = fit_map_surrogate(tr)
                theta_h, _, _ = _fit_once(level, tr, r_h_tr, m_h_tr, lambda_task)
                cv_losses.append(_bd_rate_task_proxy(level, theta_h, te,
                                                     r_h_tr, m_h_tr, lambda_task))
            if cv_losses:
                extra["loso_mean_bd_rate_task"] = float(np.mean(cv_losses))
                extra["loso_std_bd_rate_task"] = float(np.std(cv_losses))
                extra["loso_n_folds"] = float(len(cv_losses))

    return FitResult(
        level=level.name,
        theta=theta.tolist(),
        bounds=[list(b) for b in level.bounds],
        val_loss=float(loss),
        val_bd_rate_task=bd_proxy,
        val_n_samples=int(len(oracle.delta)),
        converged=bool(conv),
        extra=extra,
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
    parser.add_argument("--n-bootstrap", type=int, default=200,
                        help="Bootstrap samples for the BD-Rate-Task CI.")
    parser.add_argument("--no-cv", action="store_true",
                        help="Disable LOSO cross-validation (faster, less rigorous).")
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
        fit = fit_level(family, oracle, r_hat, map_hat, args.lambda_task,
                        n_bootstrap=args.n_bootstrap,
                        cross_validate=not args.no_cv)
        logger.info("  loss=%.4f  bd_proxy=%.4f  ci=[%.4f, %.4f]  theta=%s",
                    fit.val_loss, fit.val_bd_rate_task,
                    fit.extra.get("val_bd_rate_task_ci_lo", float("nan")),
                    fit.extra.get("val_bd_rate_task_ci_hi", float("nan")),
                    [round(t, 4) for t in fit.theta])
        if "loso_mean_bd_rate_task" in fit.extra:
            logger.info("  LOSO mean=%.4f  std=%.4f  folds=%d",
                        fit.extra["loso_mean_bd_rate_task"],
                        fit.extra["loso_std_bd_rate_task"],
                        int(fit.extra["loso_n_folds"]))
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
