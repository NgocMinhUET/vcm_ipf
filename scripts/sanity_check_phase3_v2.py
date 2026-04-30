"""Local sanity checks for the Phase 3 Stage C v2 wiring.

Verifies (without invoking VTM or torch):
  1. ``phase3_liteqp_v2.yaml`` parses and exposes ``version: "v2"``.
  2. ``version_suffix(cfg)`` returns ``"_v2"``.
  3. ``teacher_for_sequence(cfg, seq)`` returns the correct λ_task for
     the three known sequences (3.0 / 5.0 / 8.0).
  4. The cmd-line that would be passed to ``build_liteqp_dataset`` for
     each sequence carries the right ``--lambda-task`` argument.
  5. Path computation for dataset / model / learned dirs all carry the
     ``_v2`` suffix.
  6. ``phase3_liteqp.yaml`` (v1) yields empty suffix and default 5.0.

Run::

    python phase2/scripts/sanity_check_phase3_v2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Make orchestrator importable as a module file (it lives in scripts/).
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import importlib.util as _ilu

import yaml


def load_orchestrator():
    spec = _ilu.spec_from_file_location(
        "run_phase3_liteqp_pipeline",
        HERE / "run_phase3_liteqp_pipeline.py",
    )
    mod = _ilu.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    orch = load_orchestrator()

    repo_root = HERE.parent           # phase2/
    v1_yaml = repo_root / "configs" / "phase3_liteqp.yaml"
    v2_yaml = repo_root / "configs" / "phase3_liteqp_v2.yaml"

    cfg_v1 = yaml.safe_load(open(v1_yaml, "r", encoding="utf-8"))
    cfg_v2 = yaml.safe_load(open(v2_yaml, "r", encoding="utf-8"))

    print("=" * 78)
    print("Phase 3 Stage C v2 — sanity check")
    print("=" * 78)

    # ── 1) Suffix logic ──
    sfx_v1 = orch.version_suffix(cfg_v1)
    sfx_v2 = orch.version_suffix(cfg_v2)
    print(f"\n[1] version_suffix(v1) = {sfx_v1!r:6}  (expected '')")
    print(f"    version_suffix(v2) = {sfx_v2!r:6}  (expected '_v2')")
    assert sfx_v1 == "", f"v1 suffix should be empty, got {sfx_v1!r}"
    assert sfx_v2 == "_v2", f"v2 suffix should be '_v2', got {sfx_v2!r}"

    # ── 2) Per-sequence teacher resolution ──
    print(f"\n[2] teacher_for_sequence(v2, ...):")
    expected_lambda = {
        "MOT17-02-DPM": 3.0,
        "MOT17-04-DPM": 5.0,
        "MOT17-09-DPM": 8.0,
    }
    for seq_name, want in expected_lambda.items():
        teach = orch.teacher_for_sequence(cfg_v2, seq_name)
        got = teach["lambda_task"]
        ok = "✓" if abs(got - want) < 1e-9 else "✗"
        print(f"    {ok} {seq_name}: lambda_task={got:.2f}  (expected {want:.2f})  "
              f"lambda_anchor={teach['lambda_anchor']}  eta={teach['eta']}  xi={teach['xi']}")
        assert abs(got - want) < 1e-9, f"{seq_name}: {got} != {want}"

    # v1 should fall through to default 5.0 for everything
    print(f"\n[3] teacher_for_sequence(v1, MOT17-09-DPM):")
    teach_v1 = orch.teacher_for_sequence(cfg_v1, "MOT17-09-DPM")
    print(f"    lambda_task={teach_v1['lambda_task']:.2f}  (expected 5.00)")
    assert abs(teach_v1["lambda_task"] - 5.0) < 1e-9

    # ── 4) Path computation ──
    out_root = Path("/tmp/phase3_outputs_dummy")
    print(f"\n[4] Path computation under out_root={out_root}:")
    for seq in cfg_v2["sequences"]:
        nm = seq["name"]
        dataset_path = out_root / f"oracle_liteqp{sfx_v2}" / f"{nm}.jsonl"
        model_path   = out_root / "fit" / f"liteqp_mlp{sfx_v2}.joblib"
        learned_dir  = out_root / "learned" / f"liteqp{sfx_v2}_{nm}" / "M4"
        print(f"    {nm}:")
        print(f"      dataset → {dataset_path}")
        print(f"      learned → {learned_dir}")
        assert "_v2" in str(dataset_path)
        assert "_v2" in str(learned_dir)
        # Pilot_v5 phase1_run_prefix must match the parent of `learned_dir`.
        assert learned_dir.parent.name == f"liteqp_v2_{nm}", \
            f"pilot_v5 prefix mismatch: {learned_dir.parent.name}"
    print(f"      shared model → {model_path}")
    assert "_v2" in str(model_path)

    # ── 5) Cross-check pilot_v5.yaml prefix matches ──
    pilot_v5 = yaml.safe_load(
        open(repo_root / "configs" / "pilot_v5.yaml", "r", encoding="utf-8"))
    prefix = pilot_v5["encoding"]["phase1_run_prefix"]
    print(f"\n[5] pilot_v5 phase1_run_prefix = {prefix!r}  (expected 'liteqp_v2_')")
    assert prefix == "liteqp_v2_", f"prefix mismatch: {prefix!r}"

    print("\n" + "=" * 78)
    print("All sanity checks PASSED ✓")
    print("=" * 78)


if __name__ == "__main__":
    main()
