"""
RegisterGPT — A language model where each register is a word.

Instead of mapping tokens into an opaque embedding space, computation stays
in vocabulary space the entire time.

  Input:  one-hot("cat") → R["cat"] = 1.0, everything else 0.0
  State:  always a distribution over words — transparent by definition
  Output: register state IS the prediction — no output projection needed

Every intermediate step is readable: after any instruction, you can see
which words are active and how strongly. Interpretability by construction.

Architecture:
  1. One-hot encoding over vocabulary (no learned embedding)
  2. Repeat N times:
     a. Shared self-attention  (cross-position: what words co-occur?)
     b. Fourier register op    (within-position: combine word activations)
  3. Register state → softcap → cross-entropy loss

The Fourier register ops use a compact basis over vocabulary indices.
Low frequencies group words broadly (nouns vs verbs).
High frequencies distinguish specific words (cat vs dog).
Each operation costs ~585 parameters — 3,400x cheaper than a dense layer.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from core.base import AgiModel, CommonSettings


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class CastedLinear(nn.Linear):
    """Linear layer that stores weights in fp32 but computes in the input dtype."""

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)


class Rotary(nn.Module):
    """RoPE positional encoding with cached sin/cos tables."""

    def __init__(self, dim: int, base: float = 10_000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache: tuple[int, Tensor | None, Tensor | None] = (0, None, None)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if self._cache[0] != seq_len or self._cache[1] is None:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cache = (seq_len, freqs.cos()[None, None], freqs.sin()[None, None])
        return self._cache[1].to(dtype), self._cache[2].to(dtype)


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    h = x.size(-1) // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], dim=-1)


class SharedAttention(nn.Module):
    """Causal self-attention with GQA and RoPE.

    Shared across all recurrent steps — weights paid once, used N times.
    In vocabulary space this asks: "given which words are active here,
    what words are active at other positions that I should attend to?"
    """

    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 rope_base: float, qk_gain_init: float):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads

        kv_dim = num_kv_heads * self.head_dim
        self.q_proj = CastedLinear(dim, dim, bias=False)
        self.k_proj = CastedLinear(dim, kv_dim, bias=False)
        self.v_proj = CastedLinear(dim, kv_dim, bias=False)
        self.out_proj = CastedLinear(dim, dim, bias=False)
        self.out_proj._zero_init = True

        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        nh, nkv, hd = self.num_heads, self.num_kv_heads, self.head_dim

        q = self.q_proj(x).reshape(B, T, nh, hd).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, nkv, hd).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, nkv, hd).transpose(1, 2)

        q, k = F.rms_norm(q, (hd,)), F.rms_norm(k, (hd,))
        cos, sin = self.rotary(T, x.device, q.dtype)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        q = q * self.q_gain.to(q.dtype)[None, :, None, None]

        if nkv != nh:
            rep = nh // nkv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(y.transpose(1, 2).contiguous().reshape(B, T, D))


# ---------------------------------------------------------------------------
# Fourier register operations
# ---------------------------------------------------------------------------

def make_fourier_basis(dim: int, n_basis: int) -> Tensor:
    """Fourier basis functions over vocabulary indices.

    Frequency k creates patterns that group every dim/k words together.
    Over a vocabulary: low frequencies ≈ broad categories,
    high frequencies ≈ individual word distinctions.
    """
    pos = torch.arange(dim, dtype=torch.float32) / dim
    basis = torch.zeros(dim, 2 * n_basis)
    for k in range(n_basis):
        freq = k + 1
        basis[:, 2 * k] = torch.cos(2 * math.pi * freq * pos)
        basis[:, 2 * k + 1] = torch.sin(2 * math.pi * freq * pos)
    return basis


class FourierRegisterOp(nn.Module):
    """One instruction operating on the vocabulary register bank.

    Inspired by LGP: instead of hard-indexing registers (pick R[2], apply op,
    store in R[0]), we use continuous Fourier-parameterized weighting over
    all registers simultaneously.

    Read:  weighted sum of word activations  (Fourier coefficients → softmax)
    Mix:   small channel transform + nonlinearity
    Write: distribute result back to words   (Fourier coefficients)

    ~585 parameters per operation.
    """

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
        # Read: which words to gather, weighted by Fourier pattern
        read_w = torch.softmax(basis @ self.read_coeffs.T, dim=0)  # (V, C)
        values = x @ read_w.to(x.dtype)                            # (B, T, C)

        # Mix: combine channels + nonlinearity
        values = values @ self.channel_mix.to(x.dtype) + self.bias.to(x.dtype)
        if self.activation == "relu2":
            values = F.relu(values).square()
        elif self.activation == "swish":
            values = F.silu(values)
        else:
            values = F.gelu(values)

        # Write: scatter result back to vocabulary registers
        write_w = (basis @ self.write_coeffs.T).to(x.dtype)        # (V, C)
        return values @ write_w.T * self.out_scale.to(x.dtype)


# ---------------------------------------------------------------------------
# RegisterGPT
# ---------------------------------------------------------------------------

class RegisterGPT(AgiModel):
    """Language model where registers ARE words.

    No embedding matrix — input is one-hot.
    No output projection — register state is the prediction.
    Hidden dimension = vocabulary size.
    Every intermediate state is a named distribution over words.
    """

    version = "v1_attention"
    architecture = "Shared attention"
    cross_position = "GQA + RoPE (shared weights)"
    within_position = "Fourier register ops"

    class Settings(CommonSettings):
        num_heads: int = 8
        num_kv_heads: int = 4
        rope_base: float = 10000.0
        qk_gain_init: float = 1.5

    def __init__(self, vocab_size: int = 1024, num_heads: int = 8,
                 num_kv_heads: int = 4, num_steps: int = 24,
                 n_fourier_basis: int = 16, n_channels: int = 32,
                 logit_softcap: float = 30.0, rope_base: float = 10_000.0,
                 qk_gain_init: float = 1.5, activation: str = "gelu"):
        super().__init__()
        dim = vocab_size
        self.vocab_size = vocab_size
        self.num_steps = num_steps
        self.logit_softcap = logit_softcap

        # Shared attention — paid once, reused every step
        self.attn = SharedAttention(dim, num_heads, num_kv_heads,
                                    rope_base, qk_gain_init)

        # Unique register ops — each step gets its own tiny instruction
        self.ops = nn.ModuleList([
            FourierRegisterOp(n_fourier_basis, n_channels, activation)
            for _ in range(num_steps)
        ])

        # Per-step residual gating
        self.attn_scales = nn.Parameter(torch.ones(num_steps, dim))
        self.op_scales = nn.Parameter(torch.ones(num_steps, dim))

        # Learned scale to map register magnitudes → logit range
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

        # Fourier basis over vocabulary (deterministic, not learned)
        self.register_buffer("fourier_basis", make_fourier_basis(dim, n_fourier_basis))

        # Zero-init output projections so residual stream starts clean
        for m in self.modules():
            if isinstance(m, nn.Linear) and getattr(m, "_zero_init", False):
                nn.init.zeros_(m.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        V = self.vocab_size

        # Token → one-hot register state: R["cat"] = 1.0
        x = F.one_hot(input_ids, V).to(dtype=torch.bfloat16)
        x = F.rms_norm(x, (V,))

        # Execute program: shared attention + unique register ops
        for i in range(self.num_steps):
            x = x + self.attn_scales[i].to(x.dtype) * self.attn(F.rms_norm(x, (V,)))
            x = x + self.op_scales[i].to(x.dtype) * self.ops[i](F.rms_norm(x, (V,)), self.fourier_basis)

        # Register state IS the prediction
        x = F.rms_norm(x, (V,))
        logits = x * self.logit_scale.to(x.dtype)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)

        return F.cross_entropy(logits.float().reshape(-1, V), target_ids.reshape(-1), reduction="mean")
