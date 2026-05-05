"""LiteQP-CNN — small spatial CNN for per-CTU dQP allocation.

Why CNN, not MLP (PROJECT_STATE §7.15):
    The per-CTU MLP cannot see neighbors. It treats CTU_{r,c} as i.i.d.,
    forcing TV-regularisation post-hoc. A small CNN with 3×3 convs sees
    receptive fields up to 7×7 around each CTU after three layers — i.e.
    the spatial inductive bias replaces the heuristic TV penalty *and*
    eliminates the need for sequence-level λ tuning, because the
    optimal CTU-level Lagrangian λ_c = −(∂R/∂δ)/(∂D/∂δ) is itself a
    spatial quantity that depends on local (φ, K, σ, motion, neighbors).

Two output modes (controlled by ``output_mode``):
    "residual"   r̂ = CNN(features); final δ = δ_A+(φ,K,Q) + r̂
                 → conservative variant; CNN learns the gap to the
                   analytic A+ prior (current LiteQP framework).
    "direct"     δ̂ = CNN(features); skip the A+ prior entirely
                 → aggressive variant; tests whether the analytic prior
                   is even necessary.

Loss (always Lagrangian; per-sequence λ is *baked into* δ_oracle so
the trainer never sees λ — it only sees a target). λ for oracle
generation is FIXED at 5.0 (pilot_v4 validation sweet spot, see
PROJECT_STATE §7.13/§7.15). All decisions about λ are now hyper-
parameter selection, not per-sequence cherry-picking.

Trainable parameters:
    7 × 16 × 9   (Conv1, 3×3, no bias on GN block) + GN params
  + 16 × 16 × 9
  + 16 × 16 × 9
  + 16 × 1
  ≈ 5,632 weights — **2× the MLP's 2,700 params, still "lightweight"**.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Torch is imported lazily so ``import liteqp_cnn`` does not fail in
# environments that do not have PyTorch (e.g. the CI sanity-check on
# Windows). The lazy ``_torch()`` import is cached.
_TORCH = None

def _torch():
    global _TORCH
    if _TORCH is None:
        import torch  # noqa: F401
        _TORCH = torch
    return _TORCH


logger = logging.getLogger("phase2.phase3.liteqp_cnn")


# ---------------------------------------------------------------------------
# Spatial feature schema (what the CNN sees per CTU)
# ---------------------------------------------------------------------------

# IMPORTANT: order matters. The same order is used in
# ``apply_liteqp_model.cnn_features_for_frame`` so the trained model can
# be applied to live data. Adding a new channel requires bumping the
# bundle's ``schema_version``.

CNN_INPUT_CHANNELS = [
    "phi",                # CTU-level saliency (already percentile-normalised)
    "K_c_norm",           # rate sensitivity, normalised by frame median
    "sigma_y",            # luma std-dev (normalised by frame's 95th pct)
    "motion_proxy",       # temporal abs-diff (normalised similarly)
    "prev_delta_norm",    # prev_delta / 8 (clipped to [-1, 1])
    "q_base_norm",        # (q_base − 32) / 10, broadcast to a full plane
    "phi_grad",           # |∇φ| — kept for warm-start / interpretability
]
CNN_N_INPUT = len(CNN_INPUT_CHANNELS)   # = 7


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

def build_cnn_model(*, n_input_channels: int = CNN_N_INPUT,
                     hidden: int = 16,
                     output_mode: str = "residual",
                     output_bound: float = 2.0,
                     n_groups: int = 4,
                     skip_phi_index: Optional[int] = 0):
    """Construct a fresh ``LiteQPCNN`` instance (PyTorch nn.Module).

    Parameters
    ----------
    n_input_channels : int
        Default 7 = len(CNN_INPUT_CHANNELS).
    hidden : int
        Width of the convolutional embeddings (default 16).
    output_mode : str
        "residual" (predict r̂; final δ = δ_A+ + r̂) or
        "direct"   (predict δ̂ directly, no A+ prior).
    output_bound : float
        Final activation = ``output_bound · tanh(...)``. Hard-bounds the
        CNN output to ``±output_bound``. For residual mode this should
        match ``residual_bound`` (default 2.0). For direct mode it
        should be the symmetric δ envelope (e.g. 8.0 — the hard
        VVC limit).
    n_groups : int
        GroupNorm groups (default 4 → channels/groups = 4 ⇒ no group is
        empty when ``hidden=16``). GroupNorm is preferred over BatchNorm
        because spatial training batches will be small.
    skip_phi_index : int or None
        If given, an additive skip connection from input channel
        ``skip_phi_index`` (default = 0 = φ) is added to the last hidden
        block. This biases the model toward saliency-driven decisions
        and is the architectural counterpart of the analytic A+ prior.
    """
    torch = _torch()
    nn = torch.nn

    if output_mode not in ("residual", "direct"):
        raise ValueError(f"output_mode must be 'residual' or 'direct'; "
                         f"got {output_mode!r}")

    class LiteQPCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.output_mode = output_mode
            self.output_bound = float(output_bound)
            self.skip_phi_index = skip_phi_index
            # Stage 1
            self.conv1 = nn.Conv2d(n_input_channels, hidden, 3, padding=1)
            self.gn1   = nn.GroupNorm(n_groups, hidden)
            # Stage 2
            self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
            self.gn2   = nn.GroupNorm(n_groups, hidden)
            # Stage 3
            self.conv3 = nn.Conv2d(hidden, hidden, 3, padding=1)
            self.gn3   = nn.GroupNorm(n_groups, hidden)
            # Optional skip: project input φ (1 channel) to ``hidden``.
            if skip_phi_index is not None:
                self.phi_lift = nn.Conv2d(1, hidden, 1, padding=0)
            # Head
            self.head = nn.Conv2d(hidden, 1, 1, padding=0)
            # Initialise the head to small values so the network starts
            # close to identity (0 residual / δ_A+ wins the prediction).
            nn.init.zeros_(self.head.bias)
            nn.init.normal_(self.head.weight, std=1e-3)

        def forward(self, x):
            # x: (B, n_input_channels, H, W)
            relu = torch.nn.functional.relu
            h = relu(self.gn1(self.conv1(x)))
            h = relu(self.gn2(self.conv2(h)))
            h3 = self.conv3(h)
            if self.skip_phi_index is not None:
                phi = x[:, self.skip_phi_index:self.skip_phi_index + 1]
                h3 = h3 + self.phi_lift(phi)
            h = relu(self.gn3(h3))
            y = self.head(h)
            return self.output_bound * torch.tanh(y)

    return LiteQPCNN()


def count_params(model) -> int:
    """Total trainable parameter count."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# ---------------------------------------------------------------------------
# Spatial dataset — reconstructs (B, C, H, W) tensors from existing JSONL
# ---------------------------------------------------------------------------

@dataclass
class SpatialSample:
    """One (sequence, frame, q_base) → 9×15 spatial sample."""
    sequence: str
    frame_idx: int
    q_base:   int
    n_rows:   int
    n_cols:   int
    features: np.ndarray          # (CNN_N_INPUT, H, W) float32
    target_residual: np.ndarray   # (1, H, W) float32  (clipped to ±bound)
    target_delta:    np.ndarray   # (1, H, W) float32
    delta_a_plus:    np.ndarray   # (1, H, W) float32  (for residual mode)
    K_grid:          np.ndarray   # (1, H, W) float64  (rate weights, NOT input)


def _row_to_feature_vector(r: dict) -> List[float]:
    """Same order as CNN_INPUT_CHANNELS."""
    return [
        float(r.get("phi", 0.0)),
        float(r.get("K_c_norm", 1.0)),
        float(r.get("sigma_y", 0.0)),
        float(r.get("motion_proxy", 0.0)),
        float(r.get("prev_delta", 0.0)) / 8.0,
        float(r.get("q_base_norm", 0.0)),
        float(r.get("phi_grad", 0.0)),
    ]


def load_spatial_dataset(jsonl_paths: Sequence[Path],
                          residual_bound: float = 2.0
                          ) -> List[SpatialSample]:
    """Group flat JSONL rows into ``SpatialSample`` objects.

    Returns one sample per unique (sequence, frame_idx, q_base). Each
    sample has a 9×15 spatial map of features and targets.
    """
    by_key: Dict[Tuple[str, int, int], List[dict]] = {}
    for path in jsonl_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = (row["sequence"], int(row["frame_idx"]), int(row["q_base"]))
                by_key.setdefault(key, []).append(row)

    samples: List[SpatialSample] = []
    for (seq, fi, qp), rows in by_key.items():
        n_rows = max(int(r["row"]) for r in rows) + 1
        n_cols = max(int(r["col"]) for r in rows) + 1
        feats = np.zeros((CNN_N_INPUT, n_rows, n_cols), dtype=np.float32)
        target_residual = np.zeros((1, n_rows, n_cols), dtype=np.float32)
        target_delta    = np.zeros((1, n_rows, n_cols), dtype=np.float32)
        delta_a_plus    = np.zeros((1, n_rows, n_cols), dtype=np.float32)
        K_grid          = np.ones((1, n_rows, n_cols),  dtype=np.float64)
        for r in rows:
            ri, ci = int(r["row"]), int(r["col"])
            fv = _row_to_feature_vector(r)
            for k in range(CNN_N_INPUT):
                feats[k, ri, ci] = fv[k]
            res = float(r.get("residual", float(r["delta_star"]) -
                                          float(r["delta_a_plus"])))
            res = float(np.clip(res, -residual_bound, +residual_bound))
            target_residual[0, ri, ci] = res
            target_delta[0, ri, ci]    = float(r["delta_star"])
            delta_a_plus[0, ri, ci]    = float(r["delta_a_plus"])
            K_grid[0, ri, ci]          = float(r.get("K_c", 1.0))
        samples.append(SpatialSample(
            sequence=seq, frame_idx=fi, q_base=qp,
            n_rows=n_rows, n_cols=n_cols,
            features=feats,
            target_residual=target_residual,
            target_delta=target_delta,
            delta_a_plus=delta_a_plus,
            K_grid=K_grid,
        ))
    samples.sort(key=lambda s: (s.sequence, s.q_base, s.frame_idx))
    return samples


def make_torch_dataset(samples: List[SpatialSample], output_mode: str):
    """Wrap ``SpatialSample`` list as a PyTorch ``Dataset``."""
    torch = _torch()

    class _SpatialDS(torch.utils.data.Dataset):
        def __init__(self):
            self.samples = samples
            self.output_mode = output_mode
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            s = self.samples[idx]
            x  = torch.from_numpy(s.features).float()
            ka = torch.from_numpy(s.delta_a_plus).float()
            kK = torch.from_numpy(s.K_grid).float()
            if output_mode == "residual":
                y = torch.from_numpy(s.target_residual).float()
            else:
                y = torch.from_numpy(s.target_delta).float()
            return {"x": x, "y": y, "delta_a_plus": ka, "K_grid": kK,
                    "sequence": s.sequence, "q_base": int(s.q_base),
                    "frame_idx": int(s.frame_idx)}

    return _SpatialDS()


# ---------------------------------------------------------------------------
# Loss components
# ---------------------------------------------------------------------------

@dataclass
class LossWeights:
    """Hyperparameters for the multi-term Lagrangian-style loss."""
    huber_delta: float = 0.5    # robustness threshold (units: dQP steps)
    alpha_rnp:   float = 0.10   # rate-neutral penalty
    alpha_tv:    float = 0.005  # total-variation smoothness
    alpha_bound: float = 0.10   # only used in direct mode
    delta_max:   float = 8.0    # hard δ ceiling for direct mode penalty


def _huber(diff, delta: float):
    torch = _torch()
    abs_diff = torch.abs(diff)
    quad = 0.5 * (diff ** 2)
    lin = delta * (abs_diff - 0.5 * delta)
    return torch.where(abs_diff <= delta, quad, lin)


def _tv(x):
    torch = _torch()
    dx = torch.abs(x[..., :, 1:] - x[..., :, :-1])
    dy = torch.abs(x[..., 1:, :] - x[..., :-1, :])
    return dx.mean() + dy.mean()


def _rate_neutral_squared(x, K):
    """Squared, K-weighted rate-neutral residual.

    For a rate-neutral map we want ``Σ_c K_c · x_c = 0`` per frame. We
    penalise the per-sample squared residual normalised by ``Σ K`` so
    high-rate frames don't dominate.
    """
    torch = _torch()
    # x, K: (B, 1, H, W)
    Kx = (K * x).sum(dim=(2, 3))  # (B, 1)
    Kt = K.sum(dim=(2, 3)).clamp_min(1e-6)
    ratio = Kx / Kt
    return (ratio ** 2).mean()


def compute_loss(pred, batch, output_mode: str,
                  weights: LossWeights, sample_weight=None):
    """Main loss for both modes. ``pred`` is the CNN output (residual or δ̂)."""
    torch = _torch()
    y = batch["y"]
    K = batch["K_grid"]

    # 1) Huber on the chosen target.
    diff = pred - y
    huber = _huber(diff, weights.huber_delta)
    if sample_weight is not None:
        # Per-pixel weight, e.g. 0.2 + φ + 0.1 · K̃ (same as MLP, broadcast over (B, 1, H, W))
        huber = huber * sample_weight
    huber_loss = huber.mean()

    # 2) Rate-neutral.
    if output_mode == "residual":
        rnp_arg = pred                     # residual itself should be K-balanced
    else:
        # For direct δ̂, the *deviation* from δ_A+ is what should be K-balanced.
        rnp_arg = pred - batch["delta_a_plus"]
    rnp_loss = _rate_neutral_squared(rnp_arg, K)

    # 3) TV smoothness.
    tv_loss = _tv(pred)

    # 4) Bound penalty (direct mode only).
    bound_loss = torch.tensor(0.0, device=pred.device)
    if output_mode == "direct":
        excess = torch.relu(torch.abs(pred) - weights.delta_max)
        bound_loss = (excess ** 2).mean()

    total = (huber_loss
             + weights.alpha_rnp * rnp_loss
             + weights.alpha_tv * tv_loss
             + weights.alpha_bound * bound_loss)
    return total, {
        "huber":  float(huber_loss.detach().cpu()),
        "rnp":    float(rnp_loss.detach().cpu()),
        "tv":     float(tv_loss.detach().cpu()),
        "bound":  float(bound_loss.detach().cpu()),
        "total":  float(total.detach().cpu()),
    }


def per_pixel_sample_weight(features: "_torch().Tensor"):
    """Same heuristic as the MLP trainer: ``w = 0.2 + φ + 0.1 · K̃``.

    ``features`` shape: (B, CNN_N_INPUT, H, W). Returns shape (B, 1, H, W).
    """
    phi = features[:, 0:1]
    Kn  = features[:, 1:2]
    return 0.2 + phi + 0.1 * Kn


# ---------------------------------------------------------------------------
# Inference helpers (used by apply_liteqp_model.py)
# ---------------------------------------------------------------------------

def make_input_planes(*, phi, K_norm, sigma, motion, prev_delta,
                       q_base_norm: float, phi_grad,
                       n_input_channels: int = CNN_N_INPUT) -> np.ndarray:
    """Stack 2-D feature grids into a (n_input_channels, H, W) tensor.

    Same channel order as ``CNN_INPUT_CHANNELS`` (and ``_row_to_feature_vector``).
    """
    H, W = phi.shape
    out = np.zeros((n_input_channels, H, W), dtype=np.float32)
    out[0] = phi
    out[1] = K_norm
    out[2] = sigma
    out[3] = motion
    out[4] = np.clip(prev_delta / 8.0, -1.0, 1.0)
    out[5] = float(q_base_norm)               # broadcast scalar
    out[6] = phi_grad
    return out


def cnn_predict(model, planes: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Run a single-frame forward pass and return a (H, W) output map."""
    torch = _torch()
    x = torch.from_numpy(planes).float().unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        y = model(x)            # (1, 1, H, W)
    return y.squeeze(0).squeeze(0).cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

@dataclass
class CNNBundle:
    """Everything needed to re-run the model on new data."""
    state_dict: dict           = field(default_factory=dict)
    output_mode: str           = "residual"
    output_bound: float        = 2.0
    n_input_channels: int      = CNN_N_INPUT
    hidden: int                = 16
    n_groups: int              = 4
    skip_phi_index: Optional[int] = 0
    schema_version: int        = 1
    feature_names: List[str]   = field(default_factory=lambda: list(CNN_INPUT_CHANNELS))
    train_meta: Dict[str, float] = field(default_factory=dict)


def save_bundle(bundle: CNNBundle, path: Path) -> None:
    """Save a ``CNNBundle`` (pytorch state + scalar config) to disk."""
    torch = _torch()
    payload = {
        "state_dict":      bundle.state_dict,
        "output_mode":     bundle.output_mode,
        "output_bound":    bundle.output_bound,
        "n_input_channels": bundle.n_input_channels,
        "hidden":          bundle.hidden,
        "n_groups":        bundle.n_groups,
        "skip_phi_index":  bundle.skip_phi_index,
        "schema_version":  bundle.schema_version,
        "feature_names":   bundle.feature_names,
        "train_meta":      bundle.train_meta,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    logger.info("Saved CNN bundle to %s", path)


def load_bundle(path: Path):
    """Load a ``CNNBundle`` and return ``(model_loaded_into_eval_mode, bundle)``."""
    torch = _torch()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    bundle = CNNBundle(
        state_dict=payload["state_dict"],
        output_mode=payload["output_mode"],
        output_bound=float(payload["output_bound"]),
        n_input_channels=int(payload["n_input_channels"]),
        hidden=int(payload["hidden"]),
        n_groups=int(payload["n_groups"]),
        skip_phi_index=payload.get("skip_phi_index", 0),
        schema_version=int(payload.get("schema_version", 1)),
        feature_names=list(payload.get("feature_names", CNN_INPUT_CHANNELS)),
        train_meta=dict(payload.get("train_meta", {})),
    )
    model = build_cnn_model(
        n_input_channels=bundle.n_input_channels,
        hidden=bundle.hidden,
        output_mode=bundle.output_mode,
        output_bound=bundle.output_bound,
        n_groups=bundle.n_groups,
        skip_phi_index=bundle.skip_phi_index,
    )
    model.load_state_dict(bundle.state_dict)
    model.eval()
    return model, bundle
