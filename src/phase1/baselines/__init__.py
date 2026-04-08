"""Baseline QP map generation methods for fair comparison.

Methods follow the Experiment Matrix:
    M0: Uniform QP (VVC anchor)
    M1: Binary ROI (hard mask, fixed delta)
    M5: Gaussian heatmap ROI
    M6: Exponential distance-decay map
    M7: Distance-transform weighting map
    M8: Blurred ROI mask map

All methods output CTU-level QP maps in the same format as IPF (M4),
enabling direct, fair comparison.
"""
