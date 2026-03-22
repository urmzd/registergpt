"""
GaussRegisterGPT — Register machine with FFT-based operations.

Applies Gauss's mathematics to RegisterGPT:
  1. FFT (Gauss, 1805): O(V log V) frequency extraction replaces stored basis matrix
  2. Complex coefficients (Gaussian integers): amplitude + phase via real/imag pairs
  3. Cyclic convolution (Disquisitiones, 1801): register ops as frequency-domain transforms
  4. Band-limited synthesis: IFFT output is naturally smooth — built-in regularizer

Architecture:
  1. One-hot encoding over vocabulary (no learned embedding)
  2. N steps, each:
     a. FFT-based associative memory  (cross-position: decay-weighted, parallel)
     b. FFT register op               (within-position: FFT → channels → IFFT)
  3. Register state → softcap → cross-entropy loss

vs RegisterGPT v3:
  - No fourier_basis buffer — FFT computes frequencies on the fly
  - Uses n_freq frequencies (default 64) instead of n_basis=16
  - O(V log V) projections instead of O(V * n_basis) basis matmuls
  - Full spectrum available, not just first 16 harmonics

No attention. No embedding. No output projection. No stored basis.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from core.base import AgiModel, CommonSettings


# ---------------------------------------------------------------------------
# FFT-based projection: vocab → channels (Gauss, 1805)
# ---------------------------------------------------------------------------

class GaussProjection(nn.Module):
    """Project from vocab-space to channel-space via FFT.

    Replaces FourierProjection. Instead of storing a (V, 2*n_basis) basis matrix
    and computing basis @ coeffs.T, we:
      1. FFT the input to extract frequency components — O(V log V)
      2. Take the first n_freq harmonics (skip DC)
      3. Concatenate real/imag parts (Gauss's complex representation)
      4. Apply a learned linear transform to get channels
    """

    def __init__(self, n_freq: int, n_channels: int):
        super().__init__()
        self.n_freq = n_freq
        self.weight = nn.Parameter(torch.randn(n_channels, 2 * n_freq) * 0.02)

    def forward(self, x: Tensor) -> Tensor:
        # Gauss's FFT: all frequencies in O(V log V)
        X = torch.fft.rfft(x.float(), dim=-1)     # (..., V//2+1) complex
        X = X[..., 1:self.n_freq + 1]              # skip DC, take n_freq harmonics
        X_ri = torch.cat([X.real, X.imag], dim=-1) # (..., 2*n_freq) real
        return (X_ri @ self.weight.T).to(x.dtype)


# ---------------------------------------------------------------------------
# FFT-based synthesis: channels → vocab (inverse Gauss)
# ---------------------------------------------------------------------------

class GaussSynthesis(nn.Module):
    """Project from channel-space back to vocab-space via IFFT.

    Inverse of GaussProjection:
      1. Learned linear maps channels to frequency coefficients
      2. Pack into complex spectrum (real + i*imag)
      3. Zero-pad to full spectrum (band-limited: only n_freq harmonics)
      4. IFFT back to vocabulary space

    The band-limiting means output is smooth over vocabulary indices —
    a natural regularizer from Gauss's frequency-domain perspective.
    """

    def __init__(self, n_freq: int, n_channels: int, vocab_size: int):
        super().__init__()
        self.n_freq = n_freq
        self.vocab_size = vocab_size
        self.weight = nn.Parameter(torch.randn(n_channels, 2 * n_freq) * 0.02)

    def forward(self, h: Tensor) -> Tensor:
        n = self.n_freq
        V = self.vocab_size

        # Channels → frequency coefficients
        Y_ri = h.float() @ self.weight                          # (..., 2*n_freq)
        Y = torch.complex(Y_ri[..., :n], Y_ri[..., n:])        # (..., n_freq) complex

        # Place into full spectrum (DC=0, harmonics 1..n_freq, rest=0)
        shape = list(h.shape[:-1]) + [V // 2 + 1]
        full = torch.zeros(shape, dtype=Y.dtype, device=h.device)
        full[..., 1:n + 1] = Y

        # Gauss's inverse FFT: frequency → vocab space
        return torch.fft.irfft(full, n=V, dim=-1).to(h.dtype)


# ---------------------------------------------------------------------------
# FFT-based associative memory (cross-position mixing)
# ---------------------------------------------------------------------------

class GaussMemoryStep(nn.Module):
    """Cross-position mixing via causal decay-weighted associative memory.

    Same mechanism as RegisterGPT v3 (outer-product memory with exponential decay),
    but uses GaussProjection/GaussSynthesis for the vocab↔channel mappings.

    output_t = sum_{s<t} decay^(t-s-1) * (q_t · k_s) * v_s
    """

    def __init__(self, n_freq: int, n_channels: int, vocab_size: int,
                 decay_init: float = 3.0):
        super().__init__()
        self.n_channels = n_channels
        self.q_proj = GaussProjection(n_freq, n_channels)
        self.k_proj = GaussProjection(n_freq, n_channels)
        self.v_proj = GaussProjection(n_freq, n_channels)
        self.o_proj = GaussSynthesis(n_freq, n_channels, vocab_size)
        # decay_init=3.0 → sigmoid(3.0) ≈ 0.95
        self.decay_logit = nn.Parameter(torch.tensor(decay_init))
        self.out_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor) -> Tensor:
        B, T, V = x.shape
        dtype = x.dtype

        queries = self.q_proj(x)    # (B, T, C)
        keys = self.k_proj(x)       # (B, T, C)
        values = self.v_proj(x)     # (B, T, C)

        # Content similarity in channel space
        scores = torch.bmm(queries, keys.transpose(1, 2))  # (B, T, T)

        # Causal decay mask
        decay = torch.sigmoid(self.decay_logit)
        pos = torch.arange(T, device=x.device)
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)          # (T, T)
        causal = (diff > 0)
        weights = (decay ** (diff.float() - 1).clamp(min=0)) * causal
        scores = scores * weights.to(dtype).unsqueeze(0)

        # Retrieve and synthesize back to vocab space via IFFT
        retrieved = torch.bmm(scores, values)                # (B, T, C)
        return self.o_proj(retrieved) * self.out_scale.to(dtype)


# ---------------------------------------------------------------------------
# FFT register operation (within-position cyclic convolution)
# ---------------------------------------------------------------------------

class GaussRegisterOp(nn.Module):
    """Within-position register transform via FFT → channels → IFFT.

    Gauss's cyclic convolution theorem (Disquisitiones, 1801):
    convolution over the cyclic group Z/VZ equals pointwise multiply
    in frequency domain.

    Pipeline:
      1. FFT(x) → extract frequency components
      2. Linear project to channel space (learned frequency-domain read)
      3. Channel mixing + nonlinearity
      4. Linear project back to frequency coefficients (learned write)
      5. IFFT → smooth output in vocab space
    """

    def __init__(self, n_freq: int, n_channels: int, vocab_size: int,
                 activation: str = "gelu"):
        super().__init__()
        self.activation = activation
        self.n_freq = n_freq
        self.vocab_size = vocab_size
        s = 0.02
        # Frequency domain → channels ("read" in Gauss's frequency space)
        self.freq_to_ch = nn.Parameter(torch.randn(n_channels, 2 * n_freq) * s)
        # Channel mixing
        self.channel_mix = nn.Parameter(torch.randn(n_channels, n_channels) * s)
        self.bias = nn.Parameter(torch.zeros(n_channels))
        # Channels → frequency domain ("write" in Gauss's frequency space)
        self.ch_to_freq = nn.Parameter(torch.randn(2 * n_freq, n_channels) * s)
        self.out_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor) -> Tensor:
        B, T, V = x.shape
        dtype = x.dtype
        n = self.n_freq

        # === FFT: vocab → frequency domain ===
        X = torch.fft.rfft(x.float(), dim=-1)          # (B, T, V//2+1) complex
        X = X[..., 1:n + 1]                             # skip DC, take n_freq
        X_ri = torch.cat([X.real, X.imag], dim=-1)      # (B, T, 2*n_freq)

        # === Channel bottleneck with nonlinearity ===
        h = X_ri @ self.freq_to_ch.T                    # (B, T, C)
        h = h @ self.channel_mix + self.bias             # (B, T, C)
        if self.activation == "relu2":
            h = F.relu(h).square()
        elif self.activation == "swish":
            h = F.silu(h)
        else:
            h = F.gelu(h)

        # === IFFT: channels → frequency → vocab ===
        Y_ri = h @ self.ch_to_freq.T                     # (B, T, 2*n_freq)
        Y = torch.complex(Y_ri[..., :n], Y_ri[..., n:]) # (B, T, n_freq) complex

        full = torch.zeros(B, T, V // 2 + 1, dtype=Y.dtype, device=x.device)
        full[..., 1:n + 1] = Y
        output = torch.fft.irfft(full, n=V, dim=-1)     # (B, T, V)

        return output.to(dtype) * self.out_scale.to(dtype)


# ---------------------------------------------------------------------------
# Register step
# ---------------------------------------------------------------------------

class GaussRegisterStep(nn.Module):
    """One computation step: FFT memory + FFT register op."""

    def __init__(self, n_freq: int, n_channels: int, vocab_size: int,
                 activation: str = "gelu", decay_init: float = 3.0):
        super().__init__()
        self.memory = GaussMemoryStep(n_freq, n_channels, vocab_size, decay_init)
        self.register_op = GaussRegisterOp(n_freq, n_channels, vocab_size, activation)
        self.mem_scale = nn.Parameter(torch.ones(1))
        self.op_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor) -> Tensor:
        D = x.size(-1)
        x = x + self.mem_scale.to(x.dtype) * self.memory(
            F.rms_norm(x, (D,)))
        x = x + self.op_scale.to(x.dtype) * self.register_op(
            F.rms_norm(x, (D,)))
        return x


# ---------------------------------------------------------------------------
# GaussRegisterGPT
# ---------------------------------------------------------------------------

class GaussRegisterGPT(AgiModel):
    """Register machine with FFT-based operations.

    Gauss's mathematical framework applied to RegisterGPT:
    - FFT/IFFT for vocab↔frequency transforms (1805)
    - Complex coefficients for amplitude/phase encoding
    - Cyclic convolution structure from Disquisitiones (1801)
    - Band-limited synthesis as smoothness regularizer

    No embedding. No output projection. No attention. No stored Fourier basis.
    """

    version = "v5_gauss"
    architecture = "Gaussian FFT"
    cross_position = "FFT-based associative memory"
    within_position = "FFT register ops"

    class Settings(CommonSettings):
        pass

    @classmethod
    def build_kwargs(cls, args) -> dict:
        kw = super().build_kwargs(args)
        kw['n_freq'] = kw.pop('n_fourier_basis', args.n_fourier_basis)
        return kw

    def __init__(self, vocab_size: int = 1024, num_steps: int = 8,
                 n_freq: int = 64, n_channels: int = 128,
                 logit_softcap: float = 30.0, activation: str = "gelu",
                 decay_init: float = 3.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_steps = num_steps
        self.logit_softcap = logit_softcap

        self.steps = nn.ModuleList([
            GaussRegisterStep(n_freq, n_channels, vocab_size, activation,
                              decay_init)
            for _ in range(num_steps)
        ])

        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        V = self.vocab_size
        x = F.one_hot(input_ids, V).to(dtype=torch.bfloat16)
        x = F.rms_norm(x, (V,))

        for step in self.steps:
            x = step(x)

        x = F.rms_norm(x, (V,))
        logits = x * self.logit_scale.to(x.dtype)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)

        return F.cross_entropy(logits.float().reshape(-1, V),
                               target_ids.reshape(-1), reduction="mean")
