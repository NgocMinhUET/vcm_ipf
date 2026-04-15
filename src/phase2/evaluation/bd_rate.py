"""Bjontegaard Delta (BD) metric computation.

Implements the standard BD-Rate and BD-PSNR metrics for comparing
rate-distortion performance between coding methods, following the
ITU-T J.340/J.348 recommendation.

BD-Rate: Percentage bitrate saving at equal quality (negative = saving).
BD-PSNR: Quality gain at equal bitrate (positive = improvement).
BD-Task: Adapted BD metric using task accuracy instead of PSNR.

The computation uses piecewise cubic interpolation on log-rate vs
distortion (or task metric) curves, requiring at least 4 RD points.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

logger = logging.getLogger("phase2.evaluation.bd_rate")


@dataclass
class RDPoint:
    """A single rate-distortion operating point."""
    bitrate_kbps: float
    psnr_y: float
    psnr_roi: float = 0.0
    mAP50: float = 0.0
    mAP50_95: float = 0.0


@dataclass
class BDResult:
    """Result of BD metric computation between two methods."""
    anchor_method: str
    test_method: str
    sequence: str
    n_points: int
    bd_rate_psnr: float
    bd_psnr: float
    bd_rate_roi: float
    bd_rate_task: float
    bd_task: float
    valid: bool
    error_msg: str = ""


def _cubic_poly_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit a cubic polynomial y = a*x^3 + b*x^2 + c*x + d.

    Returns coefficients [a, b, c, d].
    """
    return np.polyfit(x, y, 3)


def _cubic_poly_integrate(coeffs: np.ndarray, x_lo: float, x_hi: float) -> float:
    """Integrate a cubic polynomial over [x_lo, x_hi].

    For polynomial a*x^3 + b*x^2 + c*x + d, the integral is:
        a/4*x^4 + b/3*x^3 + c/2*x^2 + d*x evaluated from x_lo to x_hi
    """
    a, b, c, d = coeffs
    def F(x):
        return a/4*x**4 + b/3*x**3 + c/2*x**2 + d*x
    return F(x_hi) - F(x_lo)


def compute_bd_rate(
    anchor_points: List[RDPoint],
    test_points: List[RDPoint],
) -> float:
    """Compute BD-Rate (percentage bitrate saving at equal PSNR).

    Negative value means test method saves bitrate vs anchor.

    Args:
        anchor_points: RD points for the reference/anchor method.
        test_points: RD points for the test method.

    Returns:
        BD-Rate in percent (e.g., -10.0 means 10% saving).
    """
    if len(anchor_points) < 4 or len(test_points) < 4:
        logger.warning("BD-Rate requires >= 4 RD points, got %d/%d",
                       len(anchor_points), len(test_points))
        return 0.0

    anchor_rate = np.array([math.log10(p.bitrate_kbps) for p in anchor_points])
    anchor_psnr = np.array([p.psnr_y for p in anchor_points])
    test_rate = np.array([math.log10(p.bitrate_kbps) for p in test_points])
    test_psnr = np.array([p.psnr_y for p in test_points])

    return _bd_rate_core(anchor_rate, anchor_psnr, test_rate, test_psnr)


def compute_bd_psnr(
    anchor_points: List[RDPoint],
    test_points: List[RDPoint],
) -> float:
    """Compute BD-PSNR (quality gain at equal bitrate).

    Positive value means test method has higher PSNR than anchor.
    """
    if len(anchor_points) < 4 or len(test_points) < 4:
        return 0.0

    anchor_rate = np.array([math.log10(p.bitrate_kbps) for p in anchor_points])
    anchor_psnr = np.array([p.psnr_y for p in anchor_points])
    test_rate = np.array([math.log10(p.bitrate_kbps) for p in test_points])
    test_psnr = np.array([p.psnr_y for p in test_points])

    return _bd_psnr_core(anchor_rate, anchor_psnr, test_rate, test_psnr)


def compute_bd_rate_roi(
    anchor_points: List[RDPoint],
    test_points: List[RDPoint],
) -> float:
    """Compute BD-Rate using ROI PSNR instead of full-frame PSNR.

    This is the key VCM metric: bitrate saving at equal ROI quality.
    """
    if len(anchor_points) < 4 or len(test_points) < 4:
        return 0.0

    anchor_rate = np.array([math.log10(p.bitrate_kbps) for p in anchor_points])
    anchor_psnr = np.array([p.psnr_roi for p in anchor_points])
    test_rate = np.array([math.log10(p.bitrate_kbps) for p in test_points])
    test_psnr = np.array([p.psnr_roi for p in test_points])

    if np.any(anchor_psnr == 0) or np.any(test_psnr == 0):
        logger.warning("ROI PSNR contains zeros, BD-Rate-ROI may be invalid")
        return 0.0

    return _bd_rate_core(anchor_rate, anchor_psnr, test_rate, test_psnr)


def compute_bd_rate_task(
    anchor_points: List[RDPoint],
    test_points: List[RDPoint],
    metric: str = "mAP50",
) -> float:
    """Compute BD-Rate using task accuracy instead of PSNR.

    This measures bitrate saving at equal task (detection) accuracy.
    """
    if len(anchor_points) < 4 or len(test_points) < 4:
        return 0.0

    anchor_rate = np.array([math.log10(p.bitrate_kbps) for p in anchor_points])
    test_rate = np.array([math.log10(p.bitrate_kbps) for p in test_points])

    if metric == "mAP50":
        anchor_quality = np.array([p.mAP50 for p in anchor_points])
        test_quality = np.array([p.mAP50 for p in test_points])
    else:
        anchor_quality = np.array([p.mAP50_95 for p in anchor_points])
        test_quality = np.array([p.mAP50_95 for p in test_points])

    if np.any(anchor_quality == 0) and np.any(test_quality == 0):
        return 0.0

    return _bd_rate_core(anchor_rate, anchor_quality, test_rate, test_quality)


def compute_bd_task(
    anchor_points: List[RDPoint],
    test_points: List[RDPoint],
    metric: str = "mAP50",
) -> float:
    """Compute BD-Task: task accuracy gain at equal bitrate.

    Positive value means test method achieves higher mAP than anchor.
    """
    if len(anchor_points) < 4 or len(test_points) < 4:
        return 0.0

    anchor_rate = np.array([math.log10(p.bitrate_kbps) for p in anchor_points])
    test_rate = np.array([math.log10(p.bitrate_kbps) for p in test_points])

    if metric == "mAP50":
        anchor_quality = np.array([p.mAP50 for p in anchor_points])
        test_quality = np.array([p.mAP50 for p in test_points])
    else:
        anchor_quality = np.array([p.mAP50_95 for p in anchor_points])
        test_quality = np.array([p.mAP50_95 for p in test_points])

    return _bd_psnr_core(anchor_rate, anchor_quality, test_rate, test_quality)


def _bd_rate_core(
    anchor_log_rate: np.ndarray,
    anchor_quality: np.ndarray,
    test_log_rate: np.ndarray,
    test_quality: np.ndarray,
) -> float:
    """Core BD-Rate computation using piecewise cubic interpolation.

    Fits cubic polynomials log_rate = f(quality) for both anchor and test,
    then computes the average difference.

    Returns percentage bitrate difference (negative = saving).
    """
    idx_a = np.argsort(anchor_quality)
    anchor_quality = anchor_quality[idx_a]
    anchor_log_rate = anchor_log_rate[idx_a]

    idx_t = np.argsort(test_quality)
    test_quality = test_quality[idx_t]
    test_log_rate = test_log_rate[idx_t]

    q_min = max(anchor_quality[0], test_quality[0])
    q_max = min(anchor_quality[-1], test_quality[-1])

    if q_max <= q_min:
        logger.warning("No overlapping quality range for BD-Rate")
        return 0.0

    # Fit cubic: log_rate = f(quality)
    poly_a = _cubic_poly_fit(anchor_quality, anchor_log_rate)
    poly_t = _cubic_poly_fit(test_quality, test_log_rate)

    int_a = _cubic_poly_integrate(poly_a, q_min, q_max)
    int_t = _cubic_poly_integrate(poly_t, q_min, q_max)

    avg_diff = (int_t - int_a) / (q_max - q_min)

    bd_rate = (10.0 ** avg_diff - 1.0) * 100.0
    return float(bd_rate)


def _bd_psnr_core(
    anchor_log_rate: np.ndarray,
    anchor_quality: np.ndarray,
    test_log_rate: np.ndarray,
    test_quality: np.ndarray,
) -> float:
    """Core BD-PSNR computation.

    Fits cubic polynomials quality = f(log_rate) for both methods,
    then computes the average quality difference.

    Returns quality difference (positive = test is better).
    """
    idx_a = np.argsort(anchor_log_rate)
    anchor_log_rate = anchor_log_rate[idx_a]
    anchor_quality = anchor_quality[idx_a]

    idx_t = np.argsort(test_log_rate)
    test_log_rate = test_log_rate[idx_t]
    test_quality = test_quality[idx_t]

    r_min = max(anchor_log_rate[0], test_log_rate[0])
    r_max = min(anchor_log_rate[-1], test_log_rate[-1])

    if r_max <= r_min:
        return 0.0

    poly_a = _cubic_poly_fit(anchor_log_rate, anchor_quality)
    poly_t = _cubic_poly_fit(test_log_rate, test_quality)

    int_a = _cubic_poly_integrate(poly_a, r_min, r_max)
    int_t = _cubic_poly_integrate(poly_t, r_min, r_max)

    bd_psnr = (int_t - int_a) / (r_max - r_min)
    return float(bd_psnr)


def compute_full_bd_metrics(
    anchor_method: str,
    test_method: str,
    sequence: str,
    anchor_points: List[RDPoint],
    test_points: List[RDPoint],
) -> BDResult:
    """Compute all BD metrics between two methods for one sequence."""
    n_a = len(anchor_points)
    n_t = len(test_points)

    if n_a < 4 or n_t < 4:
        return BDResult(
            anchor_method=anchor_method,
            test_method=test_method,
            sequence=sequence,
            n_points=min(n_a, n_t),
            bd_rate_psnr=0, bd_psnr=0, bd_rate_roi=0,
            bd_rate_task=0, bd_task=0,
            valid=False,
            error_msg=f"Insufficient RD points: anchor={n_a}, test={n_t}",
        )

    try:
        bd_rate = compute_bd_rate(anchor_points, test_points)
        bd_psnr = compute_bd_psnr(anchor_points, test_points)
        bd_rate_roi = compute_bd_rate_roi(anchor_points, test_points)
        bd_rate_task = compute_bd_rate_task(anchor_points, test_points)
        bd_task = compute_bd_task(anchor_points, test_points)

        return BDResult(
            anchor_method=anchor_method,
            test_method=test_method,
            sequence=sequence,
            n_points=min(n_a, n_t),
            bd_rate_psnr=bd_rate,
            bd_psnr=bd_psnr,
            bd_rate_roi=bd_rate_roi,
            bd_rate_task=bd_rate_task,
            bd_task=bd_task,
            valid=True,
        )
    except Exception as e:
        logger.error("BD computation error (%s vs %s on %s): %s",
                     test_method, anchor_method, sequence, e)
        return BDResult(
            anchor_method=anchor_method,
            test_method=test_method,
            sequence=sequence,
            n_points=min(n_a, n_t),
            bd_rate_psnr=0, bd_psnr=0, bd_rate_roi=0,
            bd_rate_task=0, bd_task=0,
            valid=False,
            error_msg=str(e),
        )
