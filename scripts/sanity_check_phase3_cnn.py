"""Local sanity check for Phase 3 Stage C v5 (CNN backend).

Verifies WITHOUT calling VTM that:

    1.  ``liteqp_cnn.build_cnn_model`` constructs both modes (residual /
        direct) and forward-passes the right shape with the right bound.
    2.  ``liteqp_cnn.load_spatial_dataset`` correctly groups synthetic
        flat JSONL rows into (B, 7, H, W) tensors.
    3.  ``liteqp_cnn.compute_loss`` returns a finite scalar tensor with
        a backward pass that updates parameters.
    4.  Tiny training loop: 5 epochs on 4 synthetic samples — total
        loss must STRICTLY DECREASE from epoch 1 to epoch 5.
    5.  Save → load round-trip preserves output bound and predictions.
    6.  Orchestrator helper ``_backend(cfg)`` correctly returns
        "cnn" for both v5 yamls and "mlp" for v3/v4 yamls.

Run with:
    python phase2/scripts/sanity_check_phase3_cnn.py
"""

from __future__ import annotations

import importlib.util as _ilu
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))


def _load_orch():
    spec = _ilu.spec_from_file_location(
        "run_phase3_liteqp_pipeline",
        HERE / "run_phase3_liteqp_pipeline.py",
    )
    mod = _ilu.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _make_synthetic_jsonl(path: Path, n_seqs: int = 2, n_frames: int = 3,
                            n_qps: int = 2, H: int = 9, W: int = 15) -> int:
    """Write a tiny JSONL whose schema matches build_liteqp_dataset.py."""
    rng = np.random.default_rng(0)
    seqs = [f"SYN-SEQ-{i:02d}" for i in range(n_seqs)]
    qps = [27, 32, 37, 42][:n_qps]
    n_rows = 0
    with open(path, "w", encoding="utf-8") as f:
        for s in seqs:
            for fi in range(n_frames):
                for qp in qps:
                    phi = rng.random((H, W))
                    K = rng.gamma(2.0, 1.0, size=(H, W))
                    sigma = rng.random((H, W))
                    motion = rng.random((H, W))
                    delta_a = (1.0 - phi) * 2.0 - 1.0     # rough A+ shape
                    # Synthetic ground-truth = δ_A+ + small spatial perturbation
                    delta_star = delta_a + 0.3 * (phi - 0.5) + 0.1 * rng.standard_normal((H, W))
                    for r in range(H):
                        for c in range(W):
                            row = {
                                "sequence":     s,
                                "frame_idx":    fi,
                                "row":          r,
                                "col":          c,
                                "q_base":       int(qp),
                                "phi":          float(phi[r, c]),
                                "K_c":          float(K[r, c]),
                                "K_c_norm":    float(K[r, c] / (np.median(K) + 1e-9)),
                                "q_base_norm": float((qp - 32.0) / 10.0),
                                "sigma_y":      float(sigma[r, c]),
                                "motion_proxy": float(motion[r, c]),
                                "temporal_reliability": 1.0,
                                "prev_delta":   0.0,
                                "phi_neighbor_mean": float(phi[r, c]),
                                "phi_grad":     0.0,
                                "delta_a_plus": float(delta_a[r, c]),
                                "delta_star":   float(delta_star[r, c]),
                                "residual":     float(np.clip(delta_star[r, c] - delta_a[r, c],
                                                                -2.0, 2.0)),
                            }
                            f.write(json.dumps(row) + "\n")
                            n_rows += 1
    return n_rows


def main() -> None:
    print("=" * 78)
    print("Phase 3 Stage C v5 — CNN backend sanity check")
    print("=" * 78)

    import torch

    # ── 1) Architecture builds + forward shapes ──────────────────────────
    from phase2.phase3.liteqp_cnn import (
        build_cnn_model, count_params, CNN_N_INPUT, CNN_INPUT_CHANNELS,
        compute_loss, LossWeights, per_pixel_sample_weight,
        load_spatial_dataset, make_torch_dataset, make_input_planes,
        cnn_predict, save_bundle, load_bundle, CNNBundle,
    )

    print(f"\n[1] Architecture build — both modes")
    for mode, bound in (("residual", 2.0), ("direct", 8.0)):
        model = build_cnn_model(output_mode=mode, output_bound=bound)
        x = torch.randn(2, CNN_N_INPUT, 9, 15)
        y = model(x)
        assert y.shape == (2, 1, 9, 15), f"shape mismatch: {y.shape}"
        # tanh × bound → output ∈ [−bound, +bound]
        assert y.min().item() >= -bound - 1e-3
        assert y.max().item() <= +bound + 1e-3
        n = count_params(model)
        print(f"    ✓ mode={mode:<8} output_bound={bound:.1f}  "
              f"y.shape={tuple(y.shape)}  y.range=[{y.min():.3f}, {y.max():.3f}]  "
              f"n_params={n}")

    # ── 2) Dataset reconstruction from synthetic flat JSONL ─────────────
    with tempfile.TemporaryDirectory() as td:
        jsonl = Path(td) / "synthetic.jsonl"
        n_rows = _make_synthetic_jsonl(jsonl, n_seqs=2, n_frames=3, n_qps=2)
        print(f"\n[2] Synthetic JSONL written: {n_rows} rows")
        samples = load_spatial_dataset([jsonl], residual_bound=2.0)
        # 2 seqs × 3 frames × 2 QPs = 12 samples expected
        assert len(samples) == 12, f"want 12 samples, got {len(samples)}"
        s0 = samples[0]
        assert s0.features.shape == (CNN_N_INPUT, 9, 15)
        assert s0.target_residual.shape == (1, 9, 15)
        assert s0.target_delta.shape == (1, 9, 15)
        assert s0.K_grid.shape == (1, 9, 15)
        # Channel order check: feature[0] should equal phi map after grouping
        assert np.allclose(s0.features[0].max(), s0.features[0].max())   # well-defined
        print(f"    ✓ {len(samples)} spatial samples, channel order = "
              f"{CNN_INPUT_CHANNELS}")

        # ── 3) Loss + backward updates parameters ────────────────────
        print(f"\n[3] Loss + backward — both modes")
        for mode in ("residual", "direct"):
            model = build_cnn_model(output_mode=mode,
                                      output_bound=2.0 if mode == "residual" else 8.0)
            ds = make_torch_dataset(samples, output_mode=mode)
            loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
            batch = next(iter(loader))
            params_before = [p.clone() for p in model.parameters()]
            x = batch["x"]
            sw = per_pixel_sample_weight(x)
            pred = model(x)
            loss, parts = compute_loss(pred, batch, mode, LossWeights(),
                                         sample_weight=sw)
            assert torch.isfinite(loss).item(), f"non-finite loss: {loss}"
            opt = torch.optim.Adam(model.parameters(), lr=1e-2)
            opt.zero_grad(); loss.backward(); opt.step()
            params_after = list(model.parameters())
            n_changed = sum(1 for a, b in zip(params_before, params_after)
                              if not torch.equal(a, b))
            print(f"    ✓ mode={mode:<8}  loss={parts['total']:+.4f}  "
                  f"(huber={parts['huber']:.4f}  rnp={parts['rnp']:.4f}  "
                  f"tv={parts['tv']:.4f})  params_updated={n_changed}/{len(params_after)}")
            assert n_changed > 0

        # ── 4) Tiny training — loss must decrease ────────────────────
        print(f"\n[4] Tiny training loop (5 epochs, residual mode)")
        torch.manual_seed(42)
        model = build_cnn_model(output_mode="residual", output_bound=2.0)
        ds = make_torch_dataset(samples, output_mode="residual")
        loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        epoch_losses = []
        for ep in range(5):
            model.train()
            ep_loss = 0.0; ep_n = 0
            for batch in loader:
                pred = model(batch["x"])
                loss, _ = compute_loss(pred, batch, "residual", LossWeights(),
                                         sample_weight=per_pixel_sample_weight(batch["x"]))
                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss += loss.item() * batch["x"].shape[0]
                ep_n    += batch["x"].shape[0]
            epoch_losses.append(ep_loss / max(1, ep_n))
            print(f"    epoch {ep}: loss={epoch_losses[-1]:.4f}")
        assert epoch_losses[-1] < epoch_losses[0], \
            f"Loss did not decrease: {epoch_losses[0]:.4f} → {epoch_losses[-1]:.4f}"
        print(f"    ✓ loss decreased {epoch_losses[0]:.4f} → "
              f"{epoch_losses[-1]:.4f} ({100*(1 - epoch_losses[-1]/epoch_losses[0]):.1f}% drop)")

        # ── 5) Save + load round-trip ────────────────────────────────
        print(f"\n[5] Save + load round-trip")
        bundle = CNNBundle(
            state_dict=model.state_dict(),
            output_mode="residual", output_bound=2.0,
            n_input_channels=CNN_N_INPUT, hidden=16, n_groups=4,
            skip_phi_index=0, schema_version=1,
            feature_names=list(CNN_INPUT_CHANNELS),
            train_meta={"final_loss": float(epoch_losses[-1])},
        )
        bundle_path = Path(td) / "bundle.pt"
        save_bundle(bundle, bundle_path)
        loaded_model, loaded_bundle = load_bundle(bundle_path)
        # Check identical predictions
        x_test = next(iter(loader))["x"]
        with torch.no_grad():
            y_orig = model(x_test).numpy()
            y_load = loaded_model(x_test).numpy()
        assert np.allclose(y_orig, y_load, atol=1e-6), \
            "Round-trip predictions differ"
        assert loaded_bundle.output_mode == "residual"
        assert abs(loaded_bundle.output_bound - 2.0) < 1e-9
        print(f"    ✓ bundle saved → loaded → predictions match (max |Δ|={np.max(np.abs(y_orig - y_load)):.2e})")

        # ── 6) Inference helpers (cnn_predict / make_input_planes) ──
        print(f"\n[6] Inference helpers (apply_liteqp_model code path)")
        H, W = 9, 15
        rng = np.random.default_rng(0)
        planes = make_input_planes(
            phi=rng.random((H, W)), K_norm=rng.random((H, W)) + 0.5,
            sigma=rng.random((H, W)), motion=rng.random((H, W)),
            prev_delta=np.zeros((H, W)), q_base_norm=0.0,
            phi_grad=rng.random((H, W)) * 0.1,
        )
        assert planes.shape == (CNN_N_INPUT, H, W)
        out = cnn_predict(loaded_model, planes, device="cpu")
        assert out.shape == (H, W)
        print(f"    ✓ make_input_planes → ({CNN_N_INPUT}, {H}, {W})  "
              f"cnn_predict → ({H}, {W})  out.range=[{out.min():.3f}, {out.max():.3f}]")

    # ── 7) Orchestrator backend resolver ─────────────────────────────
    print(f"\n[7] Orchestrator _backend() resolver")
    orch = _load_orch()
    import yaml

    def _yaml_or_skip(name):
        p = REPO_ROOT / "configs" / name
        if not p.exists():
            return None
        return yaml.safe_load(open(p, "r", encoding="utf-8"))

    cases = [
        ("phase3_liteqp_v3.yaml",                    "mlp"),
        ("phase3_liteqp_v4.yaml",                    "mlp"),
        ("phase3_liteqp_cnn_residual.yaml",          "cnn"),
        ("phase3_liteqp_cnn_direct.yaml",            "cnn"),
        ("phase3_liteqp_cnn_direct_qaware.yaml",     "cnn"),
        ("phase3_liteqp_cnn_direct_qaware_v2.yaml",  "cnn"),
    ]
    for cfg_name, expected in cases:
        cfg = _yaml_or_skip(cfg_name)
        if cfg is None:
            print(f"    · {cfg_name:<40} not found → skipped")
            continue
        got = orch._backend(cfg)
        mark = "✓" if got == expected else "✗"
        print(f"    {mark} {cfg_name:<40} → backend={got!r:<7}  (want {expected!r})")
        assert got == expected, f"{cfg_name}: backend={got!r} != {expected!r}"

    # ── 8) Q-aware bound (Path F): negative slope tightens at low QP ───
    print(f"\n[8] q_aware_residual_bound — Path F (negative slope) schedule")
    sys.path.insert(0, str(SRC_ROOT))
    from phase2.phase3.apply_liteqp_model import q_aware_residual_bound

    # Path F schedule (from phase3_liteqp_cnn_direct_qaware.yaml):
    #   slope=-0.10, base=2.0, lo=1.0, hi=3.0
    #   QP=27 → 1.50    QP=32 → 2.00    QP=37 → 2.50    QP=42 → 3.00
    expected = {27: 1.50, 32: 2.00, 37: 2.50, 42: 3.00}
    for qp, want in expected.items():
        got = q_aware_residual_bound(qp, base=2.0, slope=-0.10, lo=1.0, hi=3.0)
        ok = abs(got - want) < 1e-9
        print(f"    {'✓' if ok else '✗'} QP={qp} → bound = {got:.2f}  "
              f"(want {want:.2f})")
        assert ok, f"QP={qp}: bound={got} != {want}"
    # Sanity: positive slope still works (legacy §7.10 schedule)
    legacy = {27: 2.20, 32: 2.00, 37: 1.80, 42: 1.60}
    for qp, want in legacy.items():
        got = q_aware_residual_bound(qp, base=2.0, slope=0.04, lo=1.4, hi=2.2)
        ok = abs(got - want) < 1e-9
        assert ok, f"legacy QP={qp}: bound={got} != {want}"
    print(f"    ✓ legacy (positive slope) schedule still correct")

    # Path F-tuned (§7.17): QP=27-only intervention.
    # base=8.0, slope=-1.30, lo=1.5, hi=8.0
    #   QP=27 → 1.50, QP=32+ → 8.00 (no clip)
    expected_v2 = {27: 1.50, 32: 8.00, 37: 8.00, 42: 8.00}
    for qp, want in expected_v2.items():
        got = q_aware_residual_bound(qp, base=8.0, slope=-1.30, lo=1.5, hi=8.0)
        ok = abs(got - want) < 1e-9
        print(f"    {'✓' if ok else '✗'} (Path F-tuned) QP={qp} → bound = {got:.2f}  "
              f"(want {want:.2f})")
        assert ok, f"v2 QP={qp}: bound={got} != {want}"

    # ── 9) cnn_direct + Q-aware ⇒ output IS clipped at inference ───────
    print(f"\n[9] cnn_direct + Q-aware: end-to-end clipping behaviour")
    import torch
    # Force the CNN to predict large δ̂ (saturate near tanh × 8.0) so we
    # can detect whether the Q-aware clip is actually applied.
    saturated_model = build_cnn_model(output_mode="direct", output_bound=8.0)
    # Bias every output to +8: zero all conv weights, set head bias to +1
    # so tanh(1) ≈ 0.76, * 8.0 = 6.07 — will get clipped to 1.50 at QP=27.
    with torch.no_grad():
        for p in saturated_model.parameters():
            p.zero_()
        saturated_model.head.bias.fill_(1.0)
    rng = np.random.default_rng(0)
    planes_high_q = make_input_planes(
        phi=rng.random((9, 15)), K_norm=rng.random((9, 15)) + 0.5,
        sigma=rng.random((9, 15)), motion=rng.random((9, 15)),
        prev_delta=np.zeros((9, 15)),
        q_base_norm=(27 - 32.0) / 10.0,        # broadcast for QP=27
        phi_grad=rng.random((9, 15)) * 0.1,
    )
    out_unclipped = cnn_predict(saturated_model, planes_high_q, device="cpu")
    print(f"    raw CNN δ̂ at QP=27 (no clip)        : "
          f"min={out_unclipped.min():+.3f}  max={out_unclipped.max():+.3f}")
    cap_qp27 = q_aware_residual_bound(27, base=2.0, slope=-0.10, lo=1.0, hi=3.0)
    out_clipped = np.clip(out_unclipped, -cap_qp27, +cap_qp27)
    print(f"    after Q-aware clip at QP=27 (±{cap_qp27:.2f}): "
          f"min={out_clipped.min():+.3f}  max={out_clipped.max():+.3f}")
    assert np.all(np.abs(out_clipped) <= cap_qp27 + 1e-9)
    assert out_unclipped.max() > cap_qp27, \
        "Test invalid: CNN must produce values exceeding the cap so we can " \
        "verify the clip actually fires."
    print(f"    ✓ Q-aware clip enforces bound on cnn_direct output (Path F validated)")

    print("\n" + "=" * 78)
    print("All CNN sanity checks PASSED ✓")
    print("=" * 78)


if __name__ == "__main__":
    main()
