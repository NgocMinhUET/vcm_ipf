"""Tests for core data schemas."""

import numpy as np
import pytest

from phase1.core.schemas import ObjectState, FrameResult, RunMetadata


def make_object(track_id=1, frame_idx=0, confidence=0.9):
    return ObjectState(
        track_id=track_id,
        frame_idx=frame_idx,
        x_center=500.0,
        y_center=300.0,
        width=100.0,
        height=150.0,
        confidence=confidence,
        class_id=0,
        class_name="person",
        class_priority=1.0,
        track_age=1.0,
    )


class TestObjectState:
    def test_to_dict_roundtrip(self):
        obj = make_object()
        d = obj.to_dict()
        obj2 = ObjectState.from_dict(d)
        assert obj.track_id == obj2.track_id
        assert obj.x_center == obj2.x_center

    def test_defaults(self):
        obj = ObjectState(
            track_id=1, frame_idx=0,
            x_center=100, y_center=200,
            width=50, height=80, confidence=0.8,
        )
        assert obj.class_priority == 1.0
        assert obj.track_age == 1.0


class TestFrameResult:
    def test_summary_dict_with_qp(self):
        fr = FrameResult(frame_idx=5, n_objects=3)
        fr.final_qp_map = np.array([[22, 30], [35, 40]], dtype=np.int32)
        s = fr.summary_dict()
        assert s["frame_idx"] == 5
        assert s["qp_min"] == 22
        assert s["qp_max"] == 40

    def test_summary_dict_no_qp(self):
        fr = FrameResult(frame_idx=0, n_objects=0)
        s = fr.summary_dict()
        assert "qp_min" not in s
