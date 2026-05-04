"""Local sanity check for Phase 3 Stage C v3 (Action 4 — auto-λ).

Verifies WITHOUT calling VTM, torch, or sklearn that:

    1.  ``auto_lambda.compute_lambdas`` returns the *expected* per-sequence
        λ_task values when fed a synthetic pilot summary that mirrors the
        real pilot_v1 M0 anchor (rates / mAPs hand-copied from the server
        ``experiment_summary.json``).
    2.  The orchestrator helper ``resolve_auto_lambda`` correctly:
        - reads the ``auto_lambda:`` block of ``phase3_liteqp_v3.yaml``,
        - calls ``compute_lambdas``,
        - WRITES a metadata file (``auto_lambda_v3.json``) under the
          requested ``output_root``,
        - INJECTS the computed λ into ``cfg["teacher_overrides"]``,
        - LEAVES other ``apply.q_aware_bound`` settings alone (Action 3 OFF
          by default).
    3.  ``apply_liteqp_model.q_aware_residual_bound`` matches the spec
        ``clip(2.0 − 0.04·(Q−32), 1.4, 2.2)`` at the four pilot QPs.
    4.  Path computation produces ``liteqp_v3_<seq>`` directories so
        ``pilot_v6.yaml``'s ``phase1_run_prefix: "liteqp_v3_"`` matches.

Run with::

    python phase2/scripts/sanity_check_phase3_v3.py
"""

from __future__ import annotations

import importlib.util as _ilu
import json
import os
import sys
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(HERE))


def load_orchestrator():
    spec = _ilu.spec_from_file_location(
        "run_phase3_liteqp_pipeline",
        HERE / "run_phase3_liteqp_pipeline.py",
    )
    mod = _ilu.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# Real pilot_v1 M0 anchor (copied from
# pilot_comparison_v1v3v4.json — used by head_to_head_analysis.py).
PILOT_V1_M0_ANCHOR = [
    # MOT17-02-DPM
    {"sequence": "MOT17-02-DPM", "method": "M0", "qp_base": 32,
     "bitrate_kbps": 891.4272, "mAP50": 0.6524},
    {"sequence": "MOT17-02-DPM", "method": "M0", "qp_base": 37,
     "bitrate_kbps": 352.7472, "mAP50": 0.6347},
    {"sequence": "MOT17-02-DPM", "method": "M0", "qp_base": 42,
     "bitrate_kbps": 154.7856, "mAP50": 0.4965},
    {"sequence": "MOT17-02-DPM", "method": "M0", "qp_base": 47,
     "bitrate_kbps":  68.1072, "mAP50": 0.4044},
    # MOT17-04-DPM
    {"sequence": "MOT17-04-DPM", "method": "M0", "qp_base": 32,
     "bitrate_kbps": 400.464,  "mAP50": 0.8461},
    {"sequence": "MOT17-04-DPM", "method": "M0", "qp_base": 37,
     "bitrate_kbps": 215.856,  "mAP50": 0.8208},
    {"sequence": "MOT17-04-DPM", "method": "M0", "qp_base": 42,
     "bitrate_kbps": 116.6112, "mAP50": 0.8022},
    {"sequence": "MOT17-04-DPM", "method": "M0", "qp_base": 47,
     "bitrate_kbps":  59.7648, "mAP50": 0.6170},
    # MOT17-09-DPM
    {"sequence": "MOT17-09-DPM", "method": "M0", "qp_base": 32,
     "bitrate_kbps": 702.888,  "mAP50": 0.7457},
    {"sequence": "MOT17-09-DPM", "method": "M0", "qp_base": 37,
     "bitrate_kbps": 380.9856, "mAP50": 0.7194},
    {"sequence": "MOT17-09-DPM", "method": "M0", "qp_base": 42,
     "bitrate_kbps": 210.1824, "mAP50": 0.5942},
    {"sequence": "MOT17-09-DPM", "method": "M0", "qp_base": 47,
     "bitrate_kbps": 111.5088, "mAP50": 0.5632},
]


# Expected auto-λ values (from `_quick_lambda_estimate.py`, base=5, α=0.5,
# clip [3.5, 7.0], median elasticity = 0.113).
EXPECTED_LAMBDA = {
    "MOT17-02-DPM": (7.00, True),     # raw=7.55 → CLIPPED to 7.00
    "MOT17-04-DPM": (4.58, False),
    "MOT17-09-DPM": (5.00, False),
}


def main() -> None:
    print("=" * 78)
    print("Phase 3 Stage C v3 — Action 4 (auto-λ) sanity check")
    print("=" * 78)

    # Write the synthetic pilot_v1 anchor to a temp file.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                       delete=False, encoding="utf-8") as f:
        json.dump({"results": PILOT_V1_M0_ANCHOR}, f, indent=2)
        anchor_path = Path(f.name)

    try:
        # ── 1) Direct check of compute_lambdas ────────────────────────────
        from phase2.phase3 import auto_lambda

        results = auto_lambda.compute_lambdas(
            pilot_summary_path=anchor_path,
            sequences=list(EXPECTED_LAMBDA.keys()),
            base_lambda=5.0, alpha=0.5,
            lambda_min=3.5, lambda_max=7.0,
            slope_mode="median",
        )
        print(f"\n[1] auto_lambda.compute_lambdas — bare call")
        for s, (want, want_clipped) in EXPECTED_LAMBDA.items():
            r = results[s]
            ok_lambda  = abs(r["lambda_task"] - want) < 0.02
            ok_clipped = bool(r["clipped"]) == want_clipped
            mark = "✓" if (ok_lambda and ok_clipped) else "✗"
            print(f"    {mark} {s}: λ={r['lambda_task']:.3f}  "
                  f"(want {want:.2f}, clipped={r['clipped']}, "
                  f"raw={r['raw_lambda']:.2f}, e={r['elasticity']:.3f})")
            assert ok_lambda, f"{s}: λ={r['lambda_task']} != {want}"
            assert ok_clipped, f"{s}: clipped={r['clipped']} != {want_clipped}"

        # ── 2) Orchestrator integration ───────────────────────────────────
        orch = load_orchestrator()

        v3_yaml = REPO_ROOT / "configs" / "phase3_liteqp_v3.yaml"
        cfg = yaml.safe_load(open(v3_yaml, "r", encoding="utf-8"))
        # Repoint the source to our temp anchor (v3.yaml points at the
        # production server path, which doesn't exist locally).
        cfg["auto_lambda"]["source"] = str(anchor_path)

        sfx = orch.version_suffix(cfg)
        print(f"\n[2] orchestrator integration — version_suffix = {sfx!r}")
        assert sfx == "_v3", f"want '_v3', got {sfx!r}"

        # Run the resolver against a temp out_root.
        with tempfile.TemporaryDirectory() as out_dir:
            out_root = Path(out_dir)
            print(f"    resolve_auto_lambda → temp out_root={out_root}")
            payload = orch.resolve_auto_lambda(cfg, out_root)
            assert payload, "auto_lambda payload empty (expected non-empty)"

            # Verify metadata file written to <out_root>/auto_lambda_v3.json
            meta_path = out_root / "auto_lambda_v3.json"
            assert meta_path.exists(), f"missing {meta_path}"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            assert "lambda_per_seq" in meta
            assert "config" in meta and "median_elasticity" in meta
            print(f"    ✓ wrote auto_lambda_v3.json "
                  f"(median_elasticity = {meta['median_elasticity']:.3f})")

            # Verify cfg.teacher_overrides was populated.
            ov = cfg.get("teacher_overrides", {})
            for s, (want, _) in EXPECTED_LAMBDA.items():
                got = ov.get(s, {}).get("lambda_task")
                assert got is not None, f"{s} missing from teacher_overrides"
                assert abs(got - want) < 0.02, f"{s}: {got} != {want}"
            print(f"    ✓ cfg.teacher_overrides populated for all 3 sequences")

            # Verify teacher_for_sequence resolves correctly afterwards.
            for s, (want, _) in EXPECTED_LAMBDA.items():
                teach = orch.teacher_for_sequence(cfg, s)
                assert abs(teach["lambda_task"] - want) < 0.02
                # Other keys should still be the defaults (0.6 / 0.06 / 0.15)
                assert abs(teach["lambda_anchor"] - 0.6) < 1e-9
                assert abs(teach["eta"] - 0.06) < 1e-9
                assert abs(teach["xi"] - 0.15) < 1e-9
            print(f"    ✓ teacher_for_sequence respects auto-λ AND "
                  f"keeps other teacher keys at defaults")

        # ── 3) Q-aware residual_bound spec ────────────────────────────────
        from phase2.phase3.apply_liteqp_model import q_aware_residual_bound
        spec = {27: 2.20, 32: 2.00, 37: 1.80, 42: 1.60}
        print(f"\n[3] q_aware_residual_bound — default schedule")
        for q, want in spec.items():
            got = q_aware_residual_bound(q, base=2.0, slope=0.04,
                                          lo=1.4, hi=2.2)
            mark = "✓" if abs(got - want) < 1e-9 else "✗"
            print(f"    {mark} QP={q} → ±{got:.2f}  (want ±{want:.2f})")
            assert abs(got - want) < 1e-9

        # Q-aware should be DEFAULT-OFF in the v3 yaml (clean attribution).
        apply_cfg = cfg.get("apply", {})
        qaw_enabled = (apply_cfg.get("q_aware_bound") or {}).get("enabled", False)
        print(f"    ✓ apply.q_aware_bound.enabled = {qaw_enabled} "
              f"(must be False in v3 for clean attribution)")
        assert qaw_enabled is False, "Action 3 must be DISABLED in v3 yaml"

        # ── 4) Pilot_v6 prefix matches ────────────────────────────────────
        pilot_v6 = yaml.safe_load(
            open(REPO_ROOT / "configs" / "pilot_v6.yaml", "r", encoding="utf-8"))
        prefix = pilot_v6["encoding"]["phase1_run_prefix"]
        print(f"\n[4] pilot_v6 phase1_run_prefix = {prefix!r}  "
              f"(orchestrator writes to liteqp_v3_<seq>/)")
        assert prefix == "liteqp_v3_", f"want 'liteqp_v3_', got {prefix!r}"

        print("\n" + "=" * 78)
        print("All sanity checks PASSED ✓")
        print("=" * 78)

    finally:
        os.unlink(anchor_path)


if __name__ == "__main__":
    main()
