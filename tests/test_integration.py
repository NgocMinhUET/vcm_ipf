"""Integration test: run field → normalize → QP → bounded dynamics end-to-end.

Does NOT require YOLO model or GPU — uses synthetic object states.
"""

import numpy as np
import pytest

from phase1.core.config import (
    FieldConfig, CTUConfig, NormalizationConfig,
    QPMappingConfig, BoundedDynamicsConfig,
)
from phase1.core.schemas import ObjectState
from phase1.field.importance_field import compute_superposition_field
from phase1.control.normalizer import TemporalNormalizer
from phase1.control.qp_mapper import map_field_to_qp
from phase1.control.bounded_dynamics import BoundedQPController


def make_scene(n_objects=3, frame_h=1080, frame_w=1920):
    """Create synthetic objects for testing."""
    np.random.seed(123)
    objects = []
    for i in range(n_objects):
        obj = ObjectState(
            track_id=i + 1,
            frame_idx=0,
            x_center=np.random.uniform(200, frame_w - 200),
            y_center=np.random.uniform(200, frame_h - 200),
            width=np.random.uniform(60, 200),
            height=np.random.uniform(80, 250),
            confidence=np.random.uniform(0.5, 0.99),
            class_priority=1.0,
            track_age=1.0,
        )
        objects.append(obj)
    return objects


class TestEndToEnd:
    def test_full_chain(self):
        """Field → Normalize → QP Map → Bounded QP, 3 frames."""
        frame_h, frame_w = 1080, 1920
        field_cfg = FieldConfig()
        ctu_cfg = CTUConfig(ctu_size=128)
        norm_cfg = NormalizationConfig()
        qp_cfg = QPMappingConfig(qp_base=32)
        bd_cfg = BoundedDynamicsConfig(eta=0.7, delta_slew=3)

        normalizer = TemporalNormalizer(norm_cfg)
        controller = BoundedQPController(bd_cfg)

        n_ctu_rows = int(np.ceil(frame_h / ctu_cfg.ctu_size))
        n_ctu_cols = int(np.ceil(frame_w / ctu_cfg.ctu_size))

        prev_qp = None

        for t in range(3):
            objects = make_scene(n_objects=3)
            for o in objects:
                o.frame_idx = t

            # Step 1: Field
            raw_field, nr, nc = compute_superposition_field(
                objects, frame_h, frame_w, field_cfg, ctu_cfg
            )
            assert raw_field.shape == (n_ctu_rows, n_ctu_cols)
            assert np.all(raw_field >= 0)

            # Step 2: Normalize
            norm_field = normalizer.normalize(raw_field)
            assert np.all(norm_field >= 0)
            assert np.all(norm_field <= 1)

            # Step 3: QP Map
            raw_qp = map_field_to_qp(norm_field, qp_cfg)
            assert raw_qp.shape == (n_ctu_rows, n_ctu_cols)

            # Step 4: Bounded Dynamics
            final_qp = controller.apply(raw_qp)
            assert final_qp.dtype == np.int32
            assert np.all(final_qp >= bd_cfg.qp_min)
            assert np.all(final_qp <= bd_cfg.qp_max)

            # Slew rate check (from frame 1 onward)
            if prev_qp is not None:
                delta = np.abs(final_qp.astype(float) - prev_qp.astype(float))
                assert np.all(delta <= bd_cfg.delta_slew + 0.5)  # +0.5 for rounding

            prev_qp = final_qp

    def test_empty_scene(self):
        """No objects → uniform QP at qp_base + delta_bg."""
        field_cfg = FieldConfig()
        ctu_cfg = CTUConfig(ctu_size=128)

        raw_field, nr, nc = compute_superposition_field(
            [], 1080, 1920, field_cfg, ctu_cfg
        )
        assert np.all(raw_field == 0)

        normalizer = TemporalNormalizer(NormalizationConfig())
        norm = normalizer.normalize(raw_field)

        qp = map_field_to_qp(norm, QPMappingConfig(qp_base=32))
        # All background → QP >= qp_base
        assert np.all(qp >= 32 - 0.01)
