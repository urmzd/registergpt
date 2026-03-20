"""
BrainWaveGPT v6 — Oscillatory dynamics as computation.

Not a transformer. No softmax attention. No embedding. No output projection.
Cross-frequency coupling between oscillatory bands replaces uniform layer processing.

Architecture:
  1. One-hot encoding → register state (vocab-dimensional)
  2. N oscillatory cycles, each:
     a. Temporal mixing (cross-position):
        - Slow memory (theta): long-range via high decay, projects through low-freq basis
        - Fast memory (gamma): local via low decay, projects through high-freq basis
        - Theta-gamma coupling: slow retrieval gates fast retrieval
     b. Band processing (within-position):
        - Low band (delta/theta): broad context via low-freq basis
        - Mid band (alpha/beta): mid-scale features via mid-freq basis
        - High band (gamma): fine detail via high-freq basis
        - Alpha coupling: low band output gates high band output
  3. Register state → softcap → cross-entropy loss

Mathematical basis:
  Fourier basis Φ ∈ R^{V×2K} partitioned into bands Φ_L, Φ_M, Φ_H.
  Each band projects vocab-space → channel-space via learned coefficients.
  Cross-frequency coupling: g = σ(h_L @ W + b), h_H ← g ⊙ h_H.
  See v6_brain_wave/DESIGN.md for full derivation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Fourier basis (shared with v3)
# ---------------------------------------------------------------------------

def make_fourier_basis(dim: int, n_basis: int) -> Tensor:
    """Fourier basis functions over vocabulary indices.

    φ_k(v) = [cos(2πkv/V), sin(2πkv/V)] for k=1..n_basis.
    Low k = smooth (broad semantic groups). High k = sharp (fine lexical).
    """
    pos = torch.arange(dim, dtype=torch.float32) / dim
    basis = torch.zeros(dim, 2 * n_basis)
    for k in range(n_basis):
        freq = k + 1
        basis[:, 2 * k] = torch.cos(2 * math.pi * freq * pos)
        basis[:, 2 * k + 1] = torch.sin(2 * math.pi * freq * pos)
    return basis


# ---------------------------------------------------------------------------
# Band projection: vocab-space → channel-space through a frequency band
# ---------------------------------------------------------------------------

class BandProjection(nn.Module):
    """Project vocab-space to channel-space via a frequency band's basis.

    R_b = softmax(Φ_b @ α_b^T, dim=0) if soft, else Φ_b @ α_b^T.
    """

    def __init__(self, n_band_basis: int, n_channels: int, soft: bool = False):
        super().__init__()
        self.soft = soft
        self.coeffs = nn.Parameter(torch.randn(n_channels, 2 * n_band_basis) * 0.02)

    def forward(self, band_basis: Tensor) -> Tensor:
        w = band_basis @ self.coeffs.T  # (V, C)
        if self.soft:
            w = torch.softmax(w, dim=0)
        return w


# ---------------------------------------------------------------------------
# Band register op: within-position transform for one frequency band
# ---------------------------------------------------------------------------

class BandRegisterOp(nn.Module):
    """Within-position register transform restricted to one frequency band.

    Analysis:  z_b = x @ softmax(Φ_b @ α_b^T)      (read from registers)
    Transform: h_b = σ(z_b @ A_b + d_b)              (channel mix)
    Synthesis: Δx = h_b @ (Φ_b @ β_b^T)^T * scale   (write to registers)
    """

    def __init__(self, n_band_basis: int, n_channels: int, activation: str = "gelu"):
        super().__init__()
        self.activation = activation
        s = 0.02
        self.read_coeffs = nn.Parameter(torch.randn(n_channels, 2 * n_band_basis) * s)
        self.write_coeffs = nn.Parameter(torch.randn(n_channels, 2 * n_band_basis) * s)
        self.channel_mix = nn.Parameter(torch.randn(n_channels, n_channels) * s)
        self.bias = nn.Parameter(torch.zeros(n_channels))
        self.out_scale = nn.Parameter(torch.tensor(0.1))

    def transform(self, x: Tensor, band_basis: Tensor) -> Tensor:
        """Read from registers and transform in channel space. Returns (B, T, C)."""
        read_w = torch.softmax(band_basis @ self.read_coeffs.T, dim=0).to(x.dtype)
        z = x @ read_w
        h = z @ self.channel_mix.to(x.dtype) + self.bias.to(x.dtype)
        if self.activation == "relu2":
            h = F.relu(h).square()
        elif self.activation == "swish":
            h = F.silu(h)
        else:
            h = F.gelu(h)
        return h

    def write_back(self, h: Tensor, band_basis: Tensor) -> Tensor:
        """Write channel representation back to vocab space. Returns (B, T, V)."""
        write_w = (band_basis @ self.write_coeffs.T).to(h.dtype)
        return h @ write_w.T * self.out_scale.to(h.dtype)


# ---------------------------------------------------------------------------
# Band memory: cross-position mixing through one frequency band
# ---------------------------------------------------------------------------

class BandMemory(nn.Module):
    """Causal decay-weighted associative memory restricted to one frequency band.

    Q, K, V projected through Φ_b. Decay rate λ_b controls temporal range.
    High λ → slow decay → long-range (theta). Low λ → fast decay → local (gamma).
    """

    def __init__(self, n_band_basis: int, n_channels: int, decay_init: float = 3.0):
        super().__init__()
        self.n_channels = n_channels
        self.query_proj = BandProjection(n_band_basis, n_channels, soft=True)
        self.key_proj = BandProjection(n_band_basis, n_channels, soft=True)
        self.value_proj = BandProjection(n_band_basis, n_channels, soft=True)
        self.output_proj = BandProjection(n_band_basis, n_channels, soft=False)
        self.decay_logit = nn.Parameter(torch.tensor(decay_init))
        self.out_scale = nn.Parameter(torch.tensor(0.1))

    def retrieve(self, x: Tensor, band_basis: Tensor) -> Tensor:
        """Causal decay-weighted retrieval in channel space. Returns (B, T, C)."""
        B, T, V = x.shape
        dtype = x.dtype

        q_w = self.query_proj(band_basis).to(dtype)
        k_w = self.key_proj(band_basis).to(dtype)
        v_w = self.value_proj(band_basis).to(dtype)

        queries = x @ q_w   # (B, T, C)
        keys = x @ k_w      # (B, T, C)
        values = x @ v_w    # (B, T, C)

        scores = torch.bmm(queries, keys.transpose(1, 2))  # (B, T, T)

        decay = torch.sigmoid(self.decay_logit)
        pos = torch.arange(T, device=x.device)
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)
        causal_mask = (diff > 0)
        decay_weights = (decay ** (diff.float() - 1).clamp(min=0)) * causal_mask
        scores = scores * decay_weights.to(dtype).unsqueeze(0)

        return torch.bmm(scores, values)  # (B, T, C)

    def project(self, retrieved: Tensor, band_basis: Tensor) -> Tensor:
        """Project channel-space retrieval back to vocab space. Returns (B, T, V)."""
        o_w = self.output_proj(band_basis).to(retrieved.dtype)
        return retrieved @ o_w.T * self.out_scale.to(retrieved.dtype)


# ---------------------------------------------------------------------------
# Oscillatory cycle: the core unit (replaces RegisterStep)
# ---------------------------------------------------------------------------

class OscillatoryCycle(nn.Module):
    """One oscillatory cycle: all frequency bands process, then couple.

    Phase 1 — Temporal mixing:
      slow_mem(Φ_L) retrieves long-range context (theta)
      fast_mem(Φ_H) retrieves local detail (gamma)
      theta-gamma coupling: g_θ = σ(r_slow @ W_θ), r_fast ← g_θ ⊙ r_fast

    Phase 2 — Band processing:
      low_op(Φ_L):  broad context
      mid_op(Φ_M):  mid-scale features
      high_op(Φ_H): fine detail
      alpha coupling: g_α = σ(h_low @ W_α), h_high ← g_α ⊙ h_high
    """

    def __init__(self, n_basis_low: int, n_basis_mid: int, n_basis_high: int,
                 n_channels: int, activation: str = "gelu",
                 slow_decay_init: float = 4.0, fast_decay_init: float = 2.0):
        super().__init__()
        s = 0.02

        # Temporal: two memories at different scales
        self.slow_mem = BandMemory(n_basis_low, n_channels, slow_decay_init)
        self.fast_mem = BandMemory(n_basis_high, n_channels, fast_decay_init)

        # Band processing
        self.low_op = BandRegisterOp(n_basis_low, n_channels, activation)
        self.mid_op = BandRegisterOp(n_basis_mid, n_channels, activation)
        self.high_op = BandRegisterOp(n_basis_high, n_channels, activation)

        # Cross-frequency coupling: theta-gamma (temporal)
        self.tg_gate_w = nn.Parameter(torch.randn(n_channels, n_channels) * s)
        self.tg_gate_b = nn.Parameter(torch.zeros(n_channels))

        # Cross-frequency coupling: alpha (band processing)
        self.alpha_gate_w = nn.Parameter(torch.randn(n_channels, n_channels) * s)
        self.alpha_gate_b = nn.Parameter(torch.zeros(n_channels))

        # Per-component scales
        self.mem_slow_scale = nn.Parameter(torch.ones(1))
        self.mem_fast_scale = nn.Parameter(torch.ones(1))
        self.op_low_scale = nn.Parameter(torch.ones(1))
        self.op_mid_scale = nn.Parameter(torch.ones(1))
        self.op_high_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor, basis_low: Tensor,
                basis_mid: Tensor, basis_high: Tensor) -> Tensor:
        D = x.size(-1)
        dtype = x.dtype

        # === Phase 1: Temporal mixing (cross-position) ===
        xn = F.rms_norm(x, (D,))

        # Slow memory: long-range context (theta)
        r_slow = self.slow_mem.retrieve(xn, basis_low)    # (B, T, C)

        # Fast memory: local detail (gamma)
        r_fast = self.fast_mem.retrieve(xn, basis_high)   # (B, T, C)

        # Theta-gamma coupling: slow retrieval gates fast retrieval
        tg_gate = torch.sigmoid(
            r_slow @ self.tg_gate_w.to(dtype) + self.tg_gate_b.to(dtype)
        )
        r_fast = tg_gate * r_fast

        # Project back to vocab space
        x = x + (self.slow_mem.project(r_slow, basis_low) * self.mem_slow_scale.to(dtype) +
                  self.fast_mem.project(r_fast, basis_high) * self.mem_fast_scale.to(dtype))

        # === Phase 2: Band processing (within-position) ===
        xn = F.rms_norm(x, (D,))

        # Each band transforms in its own frequency range
        h_low = self.low_op.transform(xn, basis_low)      # (B, T, C)
        h_mid = self.mid_op.transform(xn, basis_mid)      # (B, T, C)
        h_high = self.high_op.transform(xn, basis_high)   # (B, T, C)

        # Alpha coupling: low band gates high band (suppression of irrelevant detail)
        alpha_gate = torch.sigmoid(
            h_low @ self.alpha_gate_w.to(dtype) + self.alpha_gate_b.to(dtype)
        )
        h_high = alpha_gate * h_high

        # Write all bands back to vocab space
        x = x + (self.low_op.write_back(h_low, basis_low) * self.op_low_scale.to(dtype) +
                  self.mid_op.write_back(h_mid, basis_mid) * self.op_mid_scale.to(dtype) +
                  self.high_op.write_back(h_high, basis_high) * self.op_high_scale.to(dtype))

        return x


# ---------------------------------------------------------------------------
# BrainWaveGPT: the full model
# ---------------------------------------------------------------------------

class BrainWaveGPT(nn.Module):
    """Oscillatory language model with cross-frequency coupling.

    Not a transformer. Cross-position mixing via band-specific associative
    memories with different decay rates (theta/gamma). Within-position
    processing via band-specific register ops with alpha gating.

    Input:  one-hot(token) → register state in R^V
    Output: register state IS the prediction (no output projection)
    """

    def __init__(self, vocab_size: int = 1024, num_cycles: int = 8,
                 n_fourier_basis: int = 16, n_channels: int = 128,
                 logit_softcap: float = 30.0, activation: str = "gelu",
                 slow_decay_init: float = 4.0, fast_decay_init: float = 2.0,
                 band_split: tuple = (4, 4, 8)):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_cycles = num_cycles
        self.logit_softcap = logit_softcap

        n_low, n_mid, n_high = band_split
        assert n_low + n_mid + n_high == n_fourier_basis, \
            f"Band split {band_split} doesn't sum to {n_fourier_basis}"

        self.n_low = n_low
        self.n_mid = n_mid
        self.n_high = n_high

        self.cycles = nn.ModuleList([
            OscillatoryCycle(n_low, n_mid, n_high, n_channels, activation,
                             slow_decay_init, fast_decay_init)
            for _ in range(num_cycles)
        ])

        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("fourier_basis",
                             make_fourier_basis(vocab_size, n_fourier_basis))

    def _split_basis(self):
        """Partition Fourier basis into low/mid/high frequency bands."""
        basis = self.fourier_basis
        lo = 2 * self.n_low
        mi = lo + 2 * self.n_mid
        return basis[:, :lo], basis[:, lo:mi], basis[:, mi:]

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        V = self.vocab_size
        x = F.one_hot(input_ids, V).to(dtype=torch.bfloat16)
        x = F.rms_norm(x, (V,))

        basis_low, basis_mid, basis_high = self._split_basis()

        for cycle in self.cycles:
            x = cycle(x, basis_low, basis_mid, basis_high)

        x = F.rms_norm(x, (V,))
        logits = x * self.logit_scale.to(x.dtype)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)

        return F.cross_entropy(logits.float().reshape(-1, V),
                               target_ids.reshape(-1), reduction="mean")
