"""Tests for configuration loading and validation."""

import tempfile
from pathlib import Path

import pytest
import yaml

from phase1.core.config import IPFConfig, load_config, save_config


class TestIPFConfig:
    def test_defaults(self):
        cfg = IPFConfig()
        assert cfg.qp_mapping.qp_base == 32
        assert cfg.field.beta == 2.0
        assert cfg.bounded_dynamics.eta == 0.7

    def test_load_from_yaml(self, tmp_path):
        data = {
            "run_id": "test_run",
            "qp_mapping": {"qp_base": 28, "delta_roi": 12.0},
            "field": {"beta": 3.0},
        }
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml.dump(data))

        cfg = load_config(yaml_path)
        assert cfg.run_id == "test_run"
        assert cfg.qp_mapping.qp_base == 28
        assert cfg.qp_mapping.delta_roi == 12.0
        assert cfg.field.beta == 3.0
        # Defaults preserved for unspecified fields
        assert cfg.bounded_dynamics.delta_slew == 3.0

    def test_save_and_reload(self, tmp_path):
        cfg = IPFConfig(run_id="save_test")
        save_path = tmp_path / "saved.yaml"
        save_config(cfg, save_path)

        cfg2 = load_config(save_path)
        assert cfg2.run_id == "save_test"
        assert cfg2.qp_mapping.qp_base == cfg.qp_mapping.qp_base

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")
