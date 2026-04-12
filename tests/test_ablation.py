"""Tests for ablation study components."""

import numpy as np
import pytest

from phase1.core.config import FieldConfig, CTUConfig, NormalizationConfig
from phase1.core.schemas import ObjectState
from phase1.field.importance_field import (
    compute_superposition_field,
    compute_gaussian_superposition_field,
    compute_importance_mass,
)
from phase1.analysis.comparison_runner import (
    ABLATION_VARIANTS,
    _select_objects,
    _compute_binary_field,
    _normalize_per_frame,
)

FRAME_H, FRAME_W = 1080, 1920
CTU_CFG = CTUConfig(ctu_size=128)
FIELD_CFG = FieldConfig(eps_k=1.0, beta=2.0, superposition="sum")


def make_objects():
    return [
        ObjectState(track_id=1, frame_idx=0, x_center=500, y_center=400,
                     width=100, height=200, confidence=0.9, class_priority=1.0, track_age=1.0),
        ObjectState(track_id=2, frame_idx=0, x_center=1500, y_center=600,
                     width=60, height=180, confidence=0.7, class_priority=0.8, track_age=0.5),
    ]


class TestSelectObjects:
    def test_all_mode(self):
        objs = make_objects()
        selected = _select_objects(objs, "all")
        assert len(selected) == 2

    def test_single_mode_picks_highest_mass(self):
        objs = make_objects()
        selected = _select_objects(objs, "single")
        assert len(selected) == 1
        masses = [compute_importance_mass(o) for o in objs]
        assert compute_importance_mass(selected[0]) == max(masses)

    def test_single_mode_empty(self):
        selected = _select_objects([], "single")
        assert len(selected) == 0


class TestBinaryField:
    def test_produces_binary_values(self):
        field, nr, nc = _compute_binary_field(make_objects(), FRAME_H, FRAME_W, 128)
        unique = np.unique(field)
        assert set(unique).issubset({0.0, 1.0})

    def test_has_roi_ctus(self):
        field, nr, nc = _compute_binary_field(make_objects(), FRAME_H, FRAME_W, 128)
        assert np.any(field == 1.0)
        assert np.any(field == 0.0)

    def test_empty_objects(self):
        field, nr, nc = _compute_binary_field([], FRAME_H, FRAME_W, 128)
        assert np.all(field == 0.0)


class TestPerFrameNormalize:
    def test_output_range(self):
        field = np.random.rand(9, 15) * 100
        normed = _normalize_per_frame(field)
        assert np.all(normed >= 0.0)
        assert np.all(normed <= 1.0)

    def test_zero_field(self):
        field = np.zeros((9, 15))
        normed = _normalize_per_frame(field)
        assert normed.shape == (9, 15)


class TestCauchyVsGaussianField:
    """Validate that Cauchy (IPF) and Gaussian kernels have expected properties."""

    def test_same_shape_output(self):
        objs = make_objects()
        cauchy, nr1, nc1 = compute_superposition_field(
            objs, FRAME_H, FRAME_W, FIELD_CFG, CTU_CFG
        )
        gauss, nr2, nc2 = compute_gaussian_superposition_field(
            objs, FRAME_H, FRAME_W, FIELD_CFG, CTU_CFG
        )
        assert cauchy.shape == gauss.shape

    def test_cauchy_wider_tail(self):
        """Cauchy 1/(d^2+1) has heavier tails than Gaussian exp(-d^2/2)."""
        objs = [make_objects()[0]]
        cauchy, _, _ = compute_superposition_field(
            objs, FRAME_H, FRAME_W, FIELD_CFG, CTU_CFG
        )
        gauss, _, _ = compute_gaussian_superposition_field(
            objs, FRAME_H, FRAME_W, FIELD_CFG, CTU_CFG
        )
        cauchy_n = cauchy / cauchy.max() if cauchy.max() > 0 else cauchy
        gauss_n = gauss / gauss.max() if gauss.max() > 0 else gauss

        far_cauchy = cauchy_n[0, -1]
        far_gauss = gauss_n[0, -1]
        assert far_cauchy > far_gauss

    def test_sum_superposition_increases_with_objects(self):
        """Sum superposition: adding objects should increase field in overlap regions."""
        obj1 = make_objects()[0]
        obj2 = ObjectState(
            track_id=3, frame_idx=0,
            x_center=obj1.x_center + 50, y_center=obj1.y_center + 50,
            width=100, height=200, confidence=0.9, class_priority=1.0, track_age=1.0,
        )
        field_one, _, _ = compute_superposition_field(
            [obj1], FRAME_H, FRAME_W, FIELD_CFG, CTU_CFG
        )
        field_two, _, _ = compute_superposition_field(
            [obj1, obj2], FRAME_H, FRAME_W, FIELD_CFG, CTU_CFG
        )
        assert field_two.max() > field_one.max()


class TestAblationVariantRegistry:
    def test_all_variants_defined(self):
        expected = {"A1", "A3", "A4", "A6", "M2", "M3"}
        assert expected == set(ABLATION_VARIANTS.keys())

    def test_each_variant_has_required_keys(self):
        required = {"name", "description", "use_ema", "use_bounded", "objects", "superposition", "kernel"}
        for vid, vdef in ABLATION_VARIANTS.items():
            assert required.issubset(vdef.keys()), f"{vid} missing keys: {required - set(vdef.keys())}"
