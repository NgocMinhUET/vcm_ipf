"""Tests for control modules: normalizer, QP mapper, bounded dynamics."""

import numpy as np
import pytest

from phase1.core.config import NormalizationConfig, QPMappingConfig, BoundedDynamicsConfig
from phase1.control.normalizer import TemporalNormalizer
from phase1.control.qp_mapper import map_field_to_qp
from phase1.control.bounded_dynamics import BoundedQPController


class TestTemporalNormalizer:
    def test_output_range(self):
        norm = TemporalNormalizer(NormalizationConfig())
        field = np.random.rand(9, 15) * 100
        result = norm.normalize(field)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_zero_field(self):
        norm = TemporalNormalizer(NormalizationConfig())
        field = np.zeros((5, 5))
        result = norm.normalize(field)
        assert result.shape == (5, 5)

    def test_temporal_smoothing(self):
        norm = TemporalNormalizer(NormalizationConfig(rho=0.9))

        # First frame: high values
        f1 = np.ones((5, 5)) * 100
        r1 = norm.normalize(f1)

        # Second frame: very different values → should be smoothed
        f2 = np.ones((5, 5)) * 1000
        r2 = norm.normalize(f2)

        # The normalizer boundaries should not jump immediately
        assert norm._a is not None

    def test_reset(self):
        norm = TemporalNormalizer(NormalizationConfig())
        norm.normalize(np.ones((3, 3)))
        assert norm._a is not None
        norm.reset()
        assert norm._a is None


class TestQPMapper:
    def test_roi_gets_lower_qp(self):
        cfg = QPMappingConfig(qp_base=32, delta_roi=10, mu=0.3)
        field = np.array([[0.0, 0.5, 1.0]])
        qp = map_field_to_qp(field, cfg)

        # High importance (1.0) → lower QP
        # Low importance (0.0) → higher QP
        assert qp[0, 2] < qp[0, 0]

    def test_base_at_threshold(self):
        cfg = QPMappingConfig(qp_base=32, mu=0.5)
        field = np.array([[0.5]])
        qp = map_field_to_qp(field, cfg)
        # At mu boundary, ROI strength = 0 → QP = qp_base
        assert abs(qp[0, 0] - 32.0) < 0.01

    def test_qp_range(self):
        cfg = QPMappingConfig(qp_base=32, delta_roi=10, delta_bg=6)
        field = np.linspace(0, 1, 20).reshape(4, 5)
        qp = map_field_to_qp(field, cfg)
        assert np.min(qp) >= 32 - 10 - 0.01
        assert np.max(qp) <= 32 + 6 + 0.01


class TestBoundedDynamics:
    def test_first_frame_passthrough(self):
        ctrl = BoundedQPController(BoundedDynamicsConfig(eta=1.0, delta_slew=100))
        raw = np.array([[25.0, 35.0]], dtype=np.float64)
        out = ctrl.apply(raw)
        np.testing.assert_array_equal(out, np.array([[25, 35]]))

    def test_slew_rate_limiting(self):
        ctrl = BoundedQPController(BoundedDynamicsConfig(eta=1.0, delta_slew=2))

        # Frame 1: QP = 30
        f1 = np.array([[30.0]])
        ctrl.apply(f1)

        # Frame 2: sudden jump to 40 → should be clamped to 32
        f2 = np.array([[40.0]])
        out = ctrl.apply(f2)
        assert out[0, 0] <= 32

    def test_clamp_range(self):
        ctrl = BoundedQPController(BoundedDynamicsConfig(qp_min=15, qp_max=45, eta=1.0, delta_slew=100))
        raw = np.array([[5.0, 60.0]])
        out = ctrl.apply(raw)
        assert out[0, 0] >= 15
        assert out[0, 1] <= 45

    def test_reset(self):
        ctrl = BoundedQPController(BoundedDynamicsConfig())
        ctrl.apply(np.array([[30.0]]))
        assert ctrl._q_bar_prev is not None
        ctrl.reset()
        assert ctrl._q_bar_prev is None
