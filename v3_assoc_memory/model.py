"""
RegisterGPT v3 — Register machine with associative memory.

Replaces attention with a causal decay-weighted associative memory.
Mathematically: output_t = sum_{s<t} decay^(t-s-1) * (q_t · k_s) * v_s
This is linear attention with exponential decay — content-based, causal, parallel.

  Input:  one-hot("cat") → R["cat"] = 1.0, everything else 0.0
  State:  always a distribution over words
  Output: register state IS the prediction — no output projection

Architecture:
  1. One-hot encoding over vocabulary (no learned embedding)
  2. N unique steps, each:
     a. Associative memory query  (causal decay-weighted matmul, parallel)
     b. Fourier register op       (within-position: combine word activations)
  3. Register state → softcap → cross-entropy loss

No attention. No embedding. No output projection.
All 1970s math: dot products, outer products, weighted averages.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from core.base import AgiModel, CommonSettings


# ---------------------------------------------------------------------------
# Fourier basis
# ---------------------------------------------------------------------------

def make_fourier_basis(dim: int, n_basis: int) -> Tensor:
    """Fourier basis functions over vocabulary indices."""
    pos = torch.arange(dim, dtype=torch.float32) / dim
    basis = torch.zeros(dim, 2 * n_basis)
    for k in range(n_basis):
        freq = k + 1
        basis[:, 2 * k] = torch.cos(2 * math.pi * freq * pos)
        basis[:, 2 * k + 1] = torch.sin(2 * math.pi * freq * pos)
    return basis


# ---------------------------------------------------------------------------
# Fourier projection
# ---------------------------------------------------------------------------

class FourierProjection(nn.Module):
    """Project from vocab-space to channel space via Fourier basis coefficients."""

    def __init__(self, n_basis: int, n_channels: int, soft: bool = False):
        super().__init__()
        self.soft = soft
        self.coeffs = nn.Parameter(torch.randn(n_channels, 2 * n_basis) * 0.02)

    def forward(self, basis: Tensor) -> Tensor:
        w = basis @ self.coeffs.T  # (V, C)
        if self.soft:
            w = torch.softmax(w, dim=0)
        return w


# ---------------------------------------------------------------------------
# Associative memory (parallel, no sequential loop)
# ---------------------------------------------------------------------------

class AssociativeMemoryStep(nn.Module):
    """Cross-position mixing via causal decay-weighted associative memory.

    Fully parallel — uses batched matmuls, no Python loops.
    output_t = sum_{s<t} decay^(t-s-1) * (q_t · k_s) * v_s
    """

    def __init__(self, n_basis: int, n_channels: int, decay_init: float = 3.0):
        super().__init__()
        self.n_channels = n_channels
        self.query_proj = FourierProjection(n_basis, n_channels, soft=False)
        self.key_proj = FourierProjection(n_basis, n_channels, soft=False)
        self.value_proj = FourierProjection(n_basis, n_channels, soft=False)
        self.output_proj = FourierProjection(n_basis, n_channels, soft=False)
        # decay_init=3.0 → sigmoid(3.0) ≈ 0.95
        self.decay_logit = nn.Parameter(torch.tensor(decay_init))
        self.out_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor, basis: Tensor) -> Tensor:
        B, T, V = x.shape
        dtype = x.dtype

        # Project vocab → channels via Fourier
        q_w = self.query_proj(basis).to(dtype)   # (V, C)
        k_w = self.key_proj(basis).to(dtype)     # (V, C)
        v_w = self.value_proj(basis).to(dtype)   # (V, C)
        o_w = self.output_proj(basis).to(dtype)  # (V, C)

        queries = x @ q_w       # (B, T, C)
        keys = x @ k_w          # (B, T, C)
        values = x @ v_w        # (B, T, C)

        # Content similarity in channel space
        scores = torch.bmm(queries, keys.transpose(1, 2))  # (B, T, T)

        # Causal decay mask: decay^(t-s-1) for s < t, 0 otherwise
        decay = torch.sigmoid(self.decay_logit)
        pos = torch.arange(T, device=x.device)
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # (T, T), diff[t,s] = t - s
        causal_mask = (diff > 0)
        decay_weights = (decay ** (diff.float() - 1).clamp(min=0)) * causal_mask
        scores = scores * decay_weights.to(dtype).unsqueeze(0)  # (B, T, T)

        # Retrieve: weighted sum of values
        retrieved = torch.bmm(scores, values)  # (B, T, C)

        # Map back to vocab space
        return retrieved @ o_w.T * self.out_scale.to(dtype)


# ---------------------------------------------------------------------------
# Fourier register operation (within-position transform)
# ---------------------------------------------------------------------------

class FourierRegisterOp(nn.Module):
    """Within-position register transform via Fourier basis."""

    def __init__(self, n_basis: int, n_channels: int, activation: str = "gelu"):
        super().__init__()
        self.activation = activation
        s = 0.02
        self.read_coeffs = nn.Parameter(torch.randn(n_channels, 2 * n_basis) * s)
        self.write_coeffs = nn.Parameter(torch.randn(n_channels, 2 * n_basis) * s)
        self.channel_mix = nn.Parameter(torch.randn(n_channels, n_channels) * s)
        self.bias = nn.Parameter(torch.zeros(n_channels))
        self.out_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor, basis: Tensor) -> Tensor:
        read_w = torch.softmax(basis @ self.read_coeffs.T, dim=0)
        values = x @ read_w.to(x.dtype)
        values = values @ self.channel_mix.to(x.dtype) + self.bias.to(x.dtype)
        if self.activation == "relu2":
            values = F.relu(values).square()
        elif self.activation == "swish":
            values = F.silu(values)
        else:
            values = F.gelu(values)
        write_w = (basis @ self.write_coeffs.T).to(x.dtype)
        return values @ write_w.T * self.out_scale.to(x.dtype)


# ---------------------------------------------------------------------------
# Register step
# ---------------------------------------------------------------------------

class RegisterStep(nn.Module):
    """One LGP instruction: memory query + register transform."""

    def __init__(self, n_basis: int, n_channels: int, activation: str = "gelu",
                 decay_init: float = 3.0):
        super().__init__()
        self.memory = AssociativeMemoryStep(n_basis, n_channels, decay_init)
        self.register_op = FourierRegisterOp(n_basis, n_channels, activation)
        self.mem_scale = nn.Parameter(torch.ones(1))
        self.op_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor, basis: Tensor) -> Tensor:
        D = x.size(-1)
        x = x + self.mem_scale.to(x.dtype) * self.memory(
            F.rms_norm(x, (D,)), basis)
        x = x + self.op_scale.to(x.dtype) * self.register_op(
            F.rms_norm(x, (D,)), basis)
        return x


# ---------------------------------------------------------------------------
# RegisterGPT v3
# ---------------------------------------------------------------------------

class RegisterGPT(AgiModel):
    """Register machine with associative memory.

    No embedding. No output projection. No attention.
    Cross-position via decay-weighted associative memory (parallel).
    Within-position via Fourier register ops.
    """

    version = "v3_assoc"
    architecture = "Associative memory"
    cross_position = "Decay-weighted associative memory"
    within_position = "Fourier register ops"

    Settings = CommonSettings

    def __init__(self, vocab_size: int = 1024, num_steps: int = 8,
                 n_fourier_basis: int = 16, n_channels: int = 128,
                 logit_softcap: float = 30.0, activation: str = "gelu",
                 decay_init: float = 3.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_steps = num_steps
        self.logit_softcap = logit_softcap

        self.steps = nn.ModuleList([
            RegisterStep(n_fourier_basis, n_channels, activation, decay_init)
            for _ in range(num_steps)
        ])

        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("fourier_basis",
                             make_fourier_basis(vocab_size, n_fourier_basis))

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        V = self.vocab_size
        x = F.one_hot(input_ids, V).to(dtype=torch.bfloat16)
        x = F.rms_norm(x, (V,))

        for step in self.steps:
            x = step(x, self.fourier_basis)

        x = F.rms_norm(x, (V,))
        logits = x * self.logit_scale.to(x.dtype)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)

        return F.cross_entropy(logits.float().reshape(-1, V),
                               target_ids.reshape(-1), reduction="mean")
