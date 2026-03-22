"""
RegisterGPT v4 — Parameter-Golf-Optimized Register Machine.

Targets ~101K params (69% reduction from v3's 329K) via five optimizations:

1. Factor channel_mix -> diagonal + low-rank (16K -> 2.2K per step)
2. Share Q/K projections across all steps (65K -> 8K total)
3. Multi-head decay (H=4) for multi-timescale retrieval (+3 params/step)
4. Step reuse: 5 unique steps x 2 invocations = 10 depth (save ~38%)
5. Q/K normalization for stability (0 extra params)

Architecture:
  Input:  one-hot("cat") -> R["cat"] = 1.0
  State:  always a distribution over words
  Output: register state IS the prediction

  Shared (paid once):
    fourier_basis: (V, 2*n_basis) buffer
    shared_query_proj: (C, 2*n_basis) = 4096
    shared_key_proj:   (C, 2*n_basis) = 4096
    logit_scale: 1
    Total: 8,193

  Per unique step (x5):
    value_proj + output_proj: 2 * 4096 = 8,192
    decay_logits: (H,) = 4
    out_scale: 1
    read_coeffs + write_coeffs: 2 * 4096 = 8,192
    diag: 128, mix_down: 1024, mix_up: 1024, bias: 128
    op out_scale: 1
    mem_scale + op_scale: 2
    Step total: 18,696

  Per invocation override (x10):
    decay, out, mem, op overrides: 7 each
    Total: 70

  Grand total: 8,193 + 5*18,696 + 10*7 = ~101.7K
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from core.base import AgiModel, CommonSettings


# ---------------------------------------------------------------------------
# Fourier basis (shared with v3)
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
# Fourier projection (reused from v3)
# ---------------------------------------------------------------------------

class FourierProjection(nn.Module):
    """Project from vocab-space to channel space via Fourier basis coefficients.

    Identical to v3. Each projection is (n_channels, 2*n_basis) = 4096 params.
    """

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
# Multi-head associative memory (replaces AssociativeMemoryStep)
# ---------------------------------------------------------------------------

class MultiHeadAssociativeMemory(nn.Module):
    """Cross-position mixing via causal decay-weighted associative memory.

    Key differences from v3's AssociativeMemoryStep:
    - Q and K projections are shared (passed in, not owned) -> saves 2*4096 per step
    - H=4 heads with independent decay rates -> multi-timescale retrieval
    - Q/K are L2-normalized -> cosine similarity, better stability in bf16/int8
    - V and O projections remain per-step (unique retrieval/output per depth)

    Reasoning: Q/K define "what is similar" — the Fourier basis constrains these
    to smooth functions of vocab index, so distinct steps learn correlated Q/K.
    The per-head decay provides temporal specialization (fast bigrams vs slow
    agreement). V/O remain unique so each step retrieves different content.
    """

    def __init__(self, n_basis: int, n_channels: int, n_heads: int = 4,
                 decay_init: float = 3.0):
        super().__init__()
        self.n_channels = n_channels
        self.n_heads = n_heads
        self.head_dim = n_channels // n_heads
        assert n_channels % n_heads == 0

        # Only V and O are owned; Q and K are shared across steps
        self.value_proj = FourierProjection(n_basis, n_channels, soft=False)
        self.output_proj = FourierProjection(n_basis, n_channels, soft=False)

        # Per-head decay rates: different timescales
        self.decay_logits = nn.Parameter(
            torch.linspace(1.0, 5.0, n_heads)  # ~sigmoid -> 0.73 to 0.99
        )
        self.out_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor, basis: Tensor,
                shared_q_w: Tensor, shared_k_w: Tensor,
                decay_override: Tensor | None = None) -> Tensor:
        B, T, V = x.shape
        H, D = self.n_heads, self.head_dim
        dtype = x.dtype

        v_w = self.value_proj(basis).to(dtype)   # (V, C)
        o_w = self.output_proj(basis).to(dtype)   # (V, C)

        # Project and normalize Q/K (cosine similarity)
        queries = F.normalize(x @ shared_q_w, dim=-1)  # (B, T, C)
        keys = F.normalize(x @ shared_k_w, dim=-1)     # (B, T, C)
        values = x @ v_w                                # (B, T, C)

        # Reshape to multi-head: (B, H, T, D)
        queries = queries.view(B, T, H, D).permute(0, 2, 1, 3)
        keys = keys.view(B, T, H, D).permute(0, 2, 1, 3)
        values = values.view(B, T, H, D).permute(0, 2, 1, 3)

        # Per-head scores: (B, H, T, T)
        scores = torch.matmul(queries, keys.transpose(-1, -2))

        # Causal decay mask with per-head decay rates
        effective_decay = self.decay_logits
        if decay_override is not None:
            effective_decay = effective_decay + decay_override
        decay = torch.sigmoid(effective_decay)  # (H,)
        pos = torch.arange(T, device=x.device)
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # (T, T)
        causal_mask = (diff > 0)

        # decay_weights: (H, T, T)
        decay_weights = (
            decay.view(H, 1, 1) ** (diff.float() - 1).clamp(min=0).unsqueeze(0)
        ) * causal_mask.unsqueeze(0)

        scores = scores * decay_weights.to(dtype).unsqueeze(0)  # (B, H, T, T)

        # Retrieve and reshape back
        retrieved = torch.matmul(scores, values)  # (B, H, T, D)
        retrieved = retrieved.permute(0, 2, 1, 3).reshape(B, T, -1)  # (B, T, C)

        return retrieved @ o_w.T * self.out_scale.to(dtype)


# ---------------------------------------------------------------------------
# Efficient register op (replaces FourierRegisterOp)
# ---------------------------------------------------------------------------

class EfficientRegisterOp(nn.Module):
    """Within-position register transform with factored channel_mix.

    Key difference from v3's FourierRegisterOp:
    - channel_mix (128x128 = 16,384) replaced with diagonal + low-rank:
      diag (128) + mix_down (128x8) + mix_up (8x128) = 2,176 params
    - Saves 14,208 params per step (87% reduction in this component)

    Reasoning: The channel_mix follows a Fourier softmax read that compresses
    V=1024 -> C=128. The effective rank of the input is bounded by the Fourier
    basis (2*n_basis=32 dimensions). A rank-r cross-channel interaction plus
    per-channel scaling captures the dominant structure.

    transform(v) = v * diag + (v @ mix_down) @ mix_up + bias
    """

    def __init__(self, n_basis: int, n_channels: int, transform_rank: int = 8,
                 activation: str = "gelu"):
        super().__init__()
        self.activation = activation
        s = 0.02
        self.read_coeffs = nn.Parameter(torch.randn(n_channels, 2 * n_basis) * s)
        self.write_coeffs = nn.Parameter(torch.randn(n_channels, 2 * n_basis) * s)

        # Factored channel_mix: diagonal + low-rank
        self.diag = nn.Parameter(torch.ones(n_channels) * s)
        self.mix_down = nn.Parameter(torch.randn(n_channels, transform_rank) * s)
        self.mix_up = nn.Parameter(torch.randn(transform_rank, n_channels) * s)
        self.bias = nn.Parameter(torch.zeros(n_channels))

        self.out_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor, basis: Tensor) -> Tensor:
        dtype = x.dtype
        read_w = torch.softmax(basis @ self.read_coeffs.T, dim=0).to(dtype)
        values = x @ read_w

        # Factored transform: diagonal + low-rank
        values = (
            values * self.diag.to(dtype)
            + (values @ self.mix_down.to(dtype)) @ self.mix_up.to(dtype)
            + self.bias.to(dtype)
        )

        if self.activation == "relu2":
            values = F.relu(values).square()
        elif self.activation == "swish":
            values = F.silu(values)
        else:
            values = F.gelu(values)

        write_w = (basis @ self.write_coeffs.T).to(dtype)
        return values @ write_w.T * self.out_scale.to(dtype)


# ---------------------------------------------------------------------------
# Register step (uses new components)
# ---------------------------------------------------------------------------

class RegisterStep(nn.Module):
    """One LGP instruction: memory query + register transform.

    Same structure as v3, but uses MultiHeadAssociativeMemory and
    EfficientRegisterOp for parameter savings.
    """

    def __init__(self, n_basis: int, n_channels: int, n_heads: int = 4,
                 transform_rank: int = 8, activation: str = "gelu",
                 decay_init: float = 3.0):
        super().__init__()
        self.memory = MultiHeadAssociativeMemory(
            n_basis, n_channels, n_heads, decay_init
        )
        self.register_op = EfficientRegisterOp(
            n_basis, n_channels, transform_rank, activation
        )
        self.mem_scale = nn.Parameter(torch.ones(1))
        self.op_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor, basis: Tensor,
                shared_q_w: Tensor, shared_k_w: Tensor,
                decay_override: Tensor | None = None) -> Tensor:
        D = x.size(-1)
        x = x + self.mem_scale.to(x.dtype) * self.memory(
            F.rms_norm(x, (D,)), basis, shared_q_w, shared_k_w, decay_override)
        x = x + self.op_scale.to(x.dtype) * self.register_op(
            F.rms_norm(x, (D,)), basis)
        return x


# ---------------------------------------------------------------------------
# RegisterGPT v4
# ---------------------------------------------------------------------------

class RegisterGPTv4(AgiModel):
    """Parameter-golf-optimized register machine.

    Manages shared Q/K projections, step reuse with per-invocation overrides.

    Step reuse: unique_steps definitions are each invoked invocations_per_step
    times, yielding total_depth = unique_steps * invocations_per_step. Each
    invocation gets scalar overrides (decay, out_scale, mem_scale, op_scale)
    so the same weights behave differently at different depths. This is
    analogous to universal transformers with per-layer scalar modulation.
    """

    version = "v4_golf"
    architecture = "Parameter-optimized"
    cross_position = "Multi-head associative memory (shared Q/K)"
    within_position = "Factored register ops"

    class Settings(CommonSettings):
        unique_steps: int = 5
        invocations_per_step: int = 2
        n_heads: int = 4
        transform_rank: int = 8

    def __init__(self, vocab_size: int = 1024, unique_steps: int = 5,
                 invocations_per_step: int = 2, n_fourier_basis: int = 16,
                 n_channels: int = 128, n_heads: int = 4,
                 transform_rank: int = 8, logit_softcap: float = 30.0,
                 activation: str = "gelu", decay_init: float = 3.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.unique_steps = unique_steps
        self.invocations_per_step = invocations_per_step
        self.total_depth = unique_steps * invocations_per_step
        self.logit_softcap = logit_softcap

        # Shared Q/K projections (paid once across all steps)
        self.shared_query_proj = FourierProjection(n_fourier_basis, n_channels)
        self.shared_key_proj = FourierProjection(n_fourier_basis, n_channels)

        # Unique step definitions
        self.steps = nn.ModuleList([
            RegisterStep(n_fourier_basis, n_channels, n_heads,
                         transform_rank, activation, decay_init)
            for _ in range(unique_steps)
        ])

        # Per-invocation scalar overrides: cheap specialization
        # Shape: (total_depth,) for each scalar type
        n_inv = self.total_depth
        self.inv_decay_overrides = nn.ParameterList([
            nn.Parameter(torch.zeros(n_heads)) for _ in range(n_inv)
        ])
        self.inv_out_scale_override = nn.Parameter(torch.ones(n_inv))
        self.inv_mem_scale_override = nn.Parameter(torch.ones(n_inv))
        self.inv_op_scale_override = nn.Parameter(torch.ones(n_inv))

        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("fourier_basis",
                             make_fourier_basis(vocab_size, n_fourier_basis))

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        V = self.vocab_size
        x = F.one_hot(input_ids, V).to(dtype=torch.bfloat16)
        x = F.rms_norm(x, (V,))
        basis = self.fourier_basis
        dtype = x.dtype

        # Compute shared Q/K weight matrices once
        shared_q_w = self.shared_query_proj(basis).to(dtype)  # (V, C)
        shared_k_w = self.shared_key_proj(basis).to(dtype)    # (V, C)

        # Run steps with reuse and per-invocation overrides
        inv_idx = 0
        for step_idx, step in enumerate(self.steps):
            for inv in range(self.invocations_per_step):
                decay_override = self.inv_decay_overrides[inv_idx]
                mem_override = self.inv_mem_scale_override[inv_idx].to(dtype)
                op_override = self.inv_op_scale_override[inv_idx].to(dtype)
                out_override = self.inv_out_scale_override[inv_idx].to(dtype)

                D = x.size(-1)
                mem_out = step.memory(
                    F.rms_norm(x, (D,)), basis, shared_q_w, shared_k_w,
                    decay_override
                )
                x = x + mem_override * step.mem_scale.to(dtype) * mem_out

                op_out = step.register_op(F.rms_norm(x, (D,)), basis)
                x = x + op_override * step.op_scale.to(dtype) * op_out

                x = x * out_override

                inv_idx += 1

        x = F.rms_norm(x, (V,))
        logits = x * self.logit_scale.to(dtype)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)

        return F.cross_entropy(logits.float().reshape(-1, V),
                               target_ids.reshape(-1), reduction="mean")
