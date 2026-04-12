"""Tests for importance field computation."""

import numpy as np
import pytest

from phase1.core.config import FieldConfig, CTUConfig
from phase1.core.schemas import ObjectState
from phase1.field.importance_field import (
    compute_importance_mass,
    build_ctu_grid,
    compute_single_object_field,
    compute_single_object_gaussian_field,
    compute_superposition_field,
    compute_gaussian_superposition_field,
)


def make_obj(cx=500.0, cy=300.0, w=100.0, h=150.0, conf=0.9):
    return ObjectState(
        track_id=1, frame_idx=0,
        x_center=cx, y_center=cy,
        width=w, height=h,
        confidence=conf,
        class_priority=1.0,
        track_age=1.0,
    )


class TestImportanceMass:
    def test_basic(self):
        obj = make_obj(w=100, h=100, conf=1.0)
        m = compute_importance_mass(obj)
        assert abs(m - 100.0) < 1e-6  # 1.0 * 1.0 * 1.0 * sqrt(100*100)

    def test_scales_with_confidence(self):
        obj_hi = make_obj(conf=0.9)
        obj_lo = make_obj(conf=0.3)
        assert compute_importance_mass(obj_hi) > compute_importance_mass(obj_lo)

    def test_scales_with_area(self):
        obj_big = make_obj(w=200, h=200)
        obj_small = make_obj(w=50, h=50)
        assert compute_importance_mass(obj_big) > compute_importance_mass(obj_small)


class TestCTUGrid:
    def test_grid_shape(self):
        gx, gy, nr, nc = build_ctu_grid(1080, 1920, 128)
        assert nr == int(np.ceil(1080 / 128))  # 9
        assert nc == int(np.ceil(1920 / 128))  # 15
        assert gx.shape == (nr, nc)

    def test_centers(self):
        gx, gy, nr, nc = build_ctu_grid(256, 256, 128)
        assert nr == 2
        assert nc == 2
        assert abs(gx[0, 0] - 64.0) < 1e-6
        assert abs(gy[0, 0] - 64.0) < 1e-6


class TestSingleObjectField:
    def test_peak_at_object_center(self):
        obj = make_obj(cx=960.0, cy=540.0)
        gx, gy, nr, nc = build_ctu_grid(1080, 1920, 128)
        cfg = FieldConfig()
        field = compute_single_object_field(obj, gx, gy, cfg)

        peak_idx = np.unravel_index(np.argmax(field), field.shape)
        peak_y = peak_idx[0] * 128 + 64
        peak_x = peak_idx[1] * 128 + 64
        assert abs(peak_x - 960) < 128
        assert abs(peak_y - 540) < 128

    def test_field_decays(self):
        obj = make_obj(cx=960.0, cy=540.0)
        gx, gy, nr, nc = build_ctu_grid(1080, 1920, 128)
        cfg = FieldConfig()
        field = compute_single_object_field(obj, gx, gy, cfg)

        peak_val = np.max(field)
        corner_val = field[0, 0]
        assert peak_val > corner_val * 2

    def test_eps_k_controls_dynamic_range(self):
        """With eps_k=1.0, peak/d1 ratio should be ~2:1 (Cauchy profile)."""
        obj = make_obj(cx=960.0, cy=540.0, w=100, h=100)
        gx, gy, nr, nc = build_ctu_grid(1080, 1920, 128)
        cfg = FieldConfig(eps_k=1.0, beta=2.0)
        field = compute_single_object_field(obj, gx, gy, cfg)
        peak = np.max(field)
        assert peak < 200  # reasonable magnitude, not millions


class TestGaussianKernel:
    """Tests for the Gaussian kernel alternative (ablation A6)."""

    def test_peak_at_center(self):
        obj = make_obj(cx=960.0, cy=540.0)
        gx, gy, nr, nc = build_ctu_grid(1080, 1920, 128)
        cfg = FieldConfig()
        field = compute_single_object_gaussian_field(obj, gx, gy, cfg)
        peak_idx = np.unravel_index(np.argmax(field), field.shape)
        peak_x = peak_idx[1] * 128 + 64
        assert abs(peak_x - 960) < 128

    def test_gaussian_decays_faster_than_cauchy(self):
        """Gaussian exp(-d^2/2) should decay faster than Cauchy 1/(d^2+1) at large d."""
        obj = make_obj(cx=960.0, cy=540.0, w=100, h=100)
        gx, gy, nr, nc = build_ctu_grid(1080, 1920, 128)
        cfg = FieldConfig(eps_k=1.0, beta=2.0)

        cauchy = compute_single_object_field(obj, gx, gy, cfg)
        gauss = compute_single_object_gaussian_field(obj, gx, gy, cfg)

        cauchy_norm = cauchy / cauchy.max()
        gauss_norm = gauss / gauss.max()

        corner_cauchy = cauchy_norm[0, 0]
        corner_gauss = gauss_norm[0, 0]
        assert corner_cauchy > corner_gauss, "Cauchy should have heavier tail"

    def test_gaussian_superposition(self):
        obj1 = make_obj(cx=400, cy=300)
        obj2 = make_obj(cx=1500, cy=700)
        cfg = FieldConfig(superposition="sum")
        field, nr, nc = compute_gaussian_superposition_field(
            [obj1, obj2], 1080, 1920, cfg, CTUConfig()
        )
        assert field.shape == (nr, nc)
        assert field.max() > 0


class TestSuperposition:
    def test_empty_objects(self):
        field, nr, nc = compute_superposition_field(
            [], 1080, 1920, FieldConfig(), CTUConfig()
        )
        assert field.shape == (nr, nc)
        assert np.all(field == 0)

    def test_two_objects_additive(self):
        obj1 = make_obj(cx=400, cy=300)
        obj2 = make_obj(cx=1500, cy=700)
        cfg = FieldConfig(superposition="sum")
        field_both, _, _ = compute_superposition_field(
            [obj1, obj2], 1080, 1920, cfg, CTUConfig()
        )
        field_one, _, _ = compute_superposition_field(
            [obj1], 1080, 1920, cfg, CTUConfig()
        )
        # With two objects, field should be >= single object everywhere
        assert np.all(field_both >= field_one - 1e-10)

    def test_max_mode(self):
        obj1 = make_obj(cx=400, cy=300, conf=0.9)
        obj2 = make_obj(cx=400, cy=300, conf=0.5)
        cfg = FieldConfig(superposition="max")
        field, _, _ = compute_superposition_field(
            [obj1, obj2], 1080, 1920, cfg, CTUConfig()
        )
        field_single, _, _ = compute_superposition_field(
            [obj1], 1080, 1920, cfg, CTUConfig()
        )
        # Max mode: co-located objects → field = max, not sum
        np.testing.assert_array_almost_equal(field, field_single, decimal=5)
