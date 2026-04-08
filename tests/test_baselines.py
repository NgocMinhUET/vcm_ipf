"""Tests for baseline QP methods."""

import numpy as np
import pytest

from phase1.core.config import QPMappingConfig, CTUConfig, BoundedDynamicsConfig
from phase1.core.schemas import ObjectState
from phase1.baselines.qp_methods import (
    M0_UniformQP, M1_BinaryROI, M5_GaussianHeatmap,
    M6_ExponentialDecay, M7_DistanceTransform, M8_BlurredROI,
    create_method,
)

QP_CFG = QPMappingConfig(qp_base=32, delta_roi=10, delta_bg=6, mu=0.3)
CTU_CFG = CTUConfig(ctu_size=128)
BD_CFG = BoundedDynamicsConfig(qp_min=10, qp_max=51)
FRAME_H, FRAME_W = 1080, 1920


def make_objects():
    return [
        ObjectState(track_id=1, frame_idx=0, x_center=500, y_center=400,
                     width=100, height=200, confidence=0.9, class_priority=1.0, track_age=1.0),
        ObjectState(track_id=2, frame_idx=0, x_center=1500, y_center=600,
                     width=150, height=250, confidence=0.8, class_priority=1.0, track_age=1.0),
    ]


class TestM0:
    def test_uniform(self):
        m = M0_UniformQP(QP_CFG, CTU_CFG, BD_CFG)
        imp, qp = m.compute_qp_map(make_objects(), FRAME_H, FRAME_W)
        assert np.all(qp == 32)
        assert np.all(imp == 0)


class TestM1:
    def test_binary_has_roi(self):
        m = M1_BinaryROI(QP_CFG, CTU_CFG, BD_CFG)
        imp, qp = m.compute_qp_map(make_objects(), FRAME_H, FRAME_W)
        assert np.any(imp == 1.0)  # some ROI CTUs
        assert np.any(imp == 0.0)  # some BG CTUs
        assert np.min(qp) < 32    # ROI gets lower QP
        assert np.max(qp) > 32    # BG gets higher QP

    def test_no_objects(self):
        m = M1_BinaryROI(QP_CFG, CTU_CFG, BD_CFG)
        imp, qp = m.compute_qp_map([], FRAME_H, FRAME_W)
        assert np.all(imp == 0)


class TestM5:
    def test_gaussian_smooth(self):
        m = M5_GaussianHeatmap(QP_CFG, CTU_CFG, BD_CFG)
        imp, qp = m.compute_qp_map(make_objects(), FRAME_H, FRAME_W)
        # Gaussian should produce intermediate values (not just 0/1)
        unique_vals = np.unique(np.round(imp, 2))
        assert len(unique_vals) > 2


class TestM6:
    def test_exponential(self):
        m = M6_ExponentialDecay(QP_CFG, CTU_CFG, BD_CFG)
        imp, qp = m.compute_qp_map(make_objects(), FRAME_H, FRAME_W)
        assert imp.max() > 0
        assert np.any(qp < 32)


class TestM7:
    def test_distance_transform(self):
        m = M7_DistanceTransform(QP_CFG, CTU_CFG, BD_CFG)
        imp, qp = m.compute_qp_map(make_objects(), FRAME_H, FRAME_W)
        assert imp.max() > 0
        # Distance transform should give smooth gradient
        unique_vals = np.unique(np.round(imp, 2))
        assert len(unique_vals) > 3


class TestM8:
    def test_blurred(self):
        m = M8_BlurredROI(QP_CFG, CTU_CFG, BD_CFG)
        imp, qp = m.compute_qp_map(make_objects(), FRAME_H, FRAME_W)
        assert imp.max() > 0
        unique_vals = np.unique(np.round(imp, 2))
        assert len(unique_vals) > 2


class TestFactory:
    def test_create_all(self):
        for mid in ["M0", "M1", "M5", "M6", "M7", "M8"]:
            m = create_method(mid, QP_CFG, CTU_CFG, BD_CFG)
            assert m.method_id == mid

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            create_method("M99", QP_CFG, CTU_CFG, BD_CFG)


class TestAllMethodsSameShape:
    """All methods must produce QP maps of identical shape for fair comparison."""

    def test_same_shape(self):
        objects = make_objects()
        shapes = []
        for mid in ["M0", "M1", "M5", "M6", "M7", "M8"]:
            m = create_method(mid, QP_CFG, CTU_CFG, BD_CFG)
            imp, qp = m.compute_qp_map(objects, FRAME_H, FRAME_W)
            shapes.append(qp.shape)

        assert all(s == shapes[0] for s in shapes), f"Shape mismatch: {shapes}"
