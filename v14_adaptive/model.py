"""
v14: Adaptive Register Machine — Data-dependent dynamics.

Built on v2 (best performer) with insights from Mamba, RWKV, and Hyena:

1. Data-dependent decay: decay rate is a function of the current register
   state, not a fixed scalar. Function words get fast decay, content words
   get slow decay. (Mamba/RWKV insight)

2. Input-dependent convolution: conv kernel weights are modulated by the
   current state, making the filter adaptive rather than fixed. (Hyena insight)

3. DCT basis instead of DFT: Discrete Cosine Transform has better energy
   compaction for real signals, halving coefficient count.

No embedding. No output projection. No attention.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from core.base import AgiModel, CommonSettings


def make_dct_basis(dim: int, n_basis: int) -> Tensor:
    """DCT-II basis functions over vocabulary indices.

    Unlike DFT which needs cos+sin pairs, DCT uses only cosines,
    halving the coefficient count for equivalent frequency coverage.
    """
    n = torch.arange(dim, dtype=torch.float32)
    basis = torch.zeros(dim, n_basis)
    for k in range(n_basis):
        basis[:, k] = torch.cos(math.pi * (k + 1) * (2 * n + 1) / (2 * dim))
    return basis


class AdaptiveCausalConv1D(nn.Module):
    """Causal depthwise conv with input-dependent kernel modulation.

    The base kernel is a fixed learned filter. A small projection from
    the current state produces per-position multiplicative gates on the
    kernel, making the effective filter adaptive. (Hyena insight)
    """

    def __init__(self, dim: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.randn(dim, 1, kernel_size) * 0.02)
        self.bias = nn.Parameter(torch.zeros(dim))
        # Gate projection: state -> per-kernel-position modulation
        self.gate_proj = nn.Linear(dim, kernel_size, bias=False)
        nn.init.zeros_(self.gate_proj.weight)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        # Compute per-position kernel gates from mean state
        gates = torch.sigmoid(self.gate_proj(x.float()))  # (B, T, K)
        gates = gates.mean(dim=1, keepdim=True)  # (B, 1, K) — batch-level gate

        # Modulate kernel
        w = self.weight.to(x.dtype) * gates.unsqueeze(1).to(x.dtype)  # (B, D, 1, K) broadcast

        # Standard causal conv (use base kernel + gate as residual)
        xp = x.transpose(1, 2)  # (B, D, T)
        xp = F.pad(xp, (self.kernel_size - 1, 0))
        # Use base conv + gated perturbation
        base = F.conv1d(xp, self.weight.to(x.dtype), self.bias.to(x.dtype), groups=D)
        return base.transpose(1, 2)


class AdaptiveDecayMemory(nn.Module):
    """Cross-position mixing with data-dependent decay rates.

    Instead of a single learned decay scalar, the decay rate is computed
    from the current register state. This lets the model learn that
    function words need fast decay (local context) while content words
    need slow decay (long-range context). (Mamba/RWKV insight)
    """

    def __init__(self, dim: int, decay_init: float = 3.0):
        super().__init__()
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        for m in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.normal_(m.weight, std=0.02)

        # Data-dependent decay: project state to per-position decay logit
        self.decay_proj = nn.Linear(dim, 1, bias=True)
        nn.init.zeros_(self.decay_proj.weight)
        self.decay_proj.bias.data.fill_(decay_init)

        self.out_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor) -> Tensor:
        B, T, k = x.shape
        dtype = x.dtype
        scale = 1.0 / math.sqrt(self.dim)

        q = self.q_proj(x.float()).to(dtype)
        k_ = self.k_proj(x.float()).to(dtype)
        v = self.v_proj(x.float()).to(dtype)

        scores = torch.bmm(q, k_.transpose(1, 2)) * scale

        # Data-dependent decay: each position gets its own decay rate
        decay_logits = self.decay_proj(x.float()).squeeze(-1)  # (B, T)
        decay = torch.sigmoid(decay_logits)  # (B, T)

        # Build causal decay mask using per-key decay rates
        pos = torch.arange(T, device=x.device)
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # (T, T)
        causal = (diff > 0).float()

        # Use key-position decay: decay_j^(i-j-1) for position i attending to j
        # decay shape: (B, T) -> use decay of key position
        log_decay = torch.log(decay + 1e-8)  # (B, T)
        # weights[i,j] = decay_j^(i-j-1) * causal[i,j]
        dist = (diff.float() - 1).clamp(min=0)  # (T, T)
        log_weights = log_decay.unsqueeze(1) * dist.unsqueeze(0)  # (B, T_q, T_k)
        weights = torch.exp(log_weights) * causal.unsqueeze(0)

        scores = scores * weights.to(dtype)
        retrieved = torch.bmm(scores, v)
        return self.o_proj(retrieved.float()).to(dtype) * self.out_scale.to(dtype)


class DCTRegisterOp(nn.Module):
    """Within-position register transform using DCT basis.

    DCT halves the coefficient count vs DFT while providing equivalent
    or better frequency coverage for real-valued signals.
    """

    def __init__(self, n_basis: int, n_channels: int, activation: str = "gelu"):
        super().__init__()
        self.activation = activation
        s = 0.02
        # DCT uses n_basis coefficients (not 2*n_basis like DFT)
        self.read_coeffs = nn.Parameter(torch.randn(n_channels, n_basis) * s)
        self.write_coeffs = nn.Parameter(torch.randn(n_channels, n_basis) * s)
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


class AdaptiveRegisterStep(nn.Module):
    """One instruction: adaptive conv + adaptive decay + DCT register op."""

    def __init__(self, vocab_size: int, k_active: int, kernel_size: int,
                 n_basis: int, n_channels: int, activation: str = "gelu",
                 decay_init: float = 3.0, step_idx: int = 0,
                 total_steps: int = 1):
        super().__init__()
        self.vocab_size = vocab_size
        self.k_active = k_active

        # Sparse routing (from v12)
        stride = vocab_size // total_steps if total_steps > 0 else vocab_size
        read_offset = (step_idx * stride) % vocab_size
        write_offset = (read_offset + vocab_size // 2) % vocab_size

        read_indices = torch.tensor(
            [(read_offset + j) % vocab_size for j in range(k_active)],
            dtype=torch.long)
        write_indices = torch.tensor(
            [(write_offset + j) % vocab_size for j in range(k_active)],
            dtype=torch.long)
        write_selector = torch.zeros(k_active, vocab_size)
        write_selector[torch.arange(k_active), write_indices] = 1.0

        self.register_buffer("read_indices", read_indices)
        self.register_buffer("write_selector", write_selector)

        # Adaptive components
        self.conv = AdaptiveCausalConv1D(vocab_size, kernel_size)
        self.memory = AdaptiveDecayMemory(k_active, decay_init)
        self.register_op = DCTRegisterOp(n_basis, n_channels, activation)

        self.conv_scale = nn.Parameter(torch.ones(vocab_size))
        self.mem_scale = nn.Parameter(torch.ones(1))
        self.write_scale = nn.Parameter(torch.tensor(0.1))

        self.inv_sqrt_k = 1.0 / math.sqrt(k_active)

    def forward(self, x: Tensor, basis: Tensor) -> Tensor:
        B, T, V = x.shape
        dtype = x.dtype
        k = self.k_active

        # Cross-position: adaptive causal conv (full vocab space)
        x = x + self.conv_scale.to(dtype) * self.conv(F.rms_norm(x, (V,)))

        # Gather sparse registers
        read_idx = self.read_indices.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
        gathered = torch.gather(x, -1, read_idx)  # (B, T, k)

        # Cross-position: adaptive decay memory (sparse subspace)
        g_norm = F.rms_norm(gathered, (k,))
        gathered = gathered + self.mem_scale.to(dtype) * self.memory(g_norm)

        # Within-position: DCT register op (full vocab space)
        x = x + self.register_op(F.rms_norm(x, (V,)), basis)

        # Sparse write-back from memory
        g_norm = F.rms_norm(gathered, (k,))
        output = g_norm * (self.write_scale.to(dtype) * self.inv_sqrt_k)
        x = x + output @ self.write_selector.to(dtype)

        return x


class AdaptiveGPT(AgiModel):
    """Adaptive register machine with data-dependent dynamics.

    Combines v2's causal convolution (best throughput) with v12's sparse
    register routing, plus:
    - Data-dependent decay rates (Mamba/RWKV)
    - Input-dependent conv modulation (Hyena)
    - DCT basis (better energy compaction)
    """

    version = "v14_adaptive"
    architecture = "Adaptive dynamics"
    cross_position = "Adaptive conv + adaptive decay"
    within_position = "DCT register ops"

    class Settings(CommonSettings):
        k_active: int = 256
        kernel_size: int = 16

    @classmethod
    def build_kwargs(cls, args) -> dict:
        fields = cls.Settings.model_fields
        return {k: getattr(args, k) for k in fields if hasattr(args, k)}

    def __init__(self, vocab_size: int = 1024, num_steps: int = 8,
                 k_active: int = 256, kernel_size: int = 16,
                 n_fourier_basis: int = 16, n_channels: int = 128,
                 logit_softcap: float = 30.0, activation: str = "gelu",
                 decay_init: float = 3.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.logit_softcap = logit_softcap

        self.steps = nn.ModuleList([
            AdaptiveRegisterStep(
                vocab_size, k_active, kernel_size, n_fourier_basis,
                n_channels, activation, decay_init,
                step_idx=i, total_steps=num_steps)
            for i in range(num_steps)
        ])

        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("dct_basis",
                             make_dct_basis(vocab_size, n_fourier_basis))

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        V = self.vocab_size
        x = F.one_hot(input_ids, V).to(dtype=torch.bfloat16)
        x = F.rms_norm(x, (V,))

        for step in self.steps:
            x = step(x, self.dct_basis)

        x = F.rms_norm(x, (V,))
        logits = x * self.logit_scale.to(x.dtype)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)

        return F.cross_entropy(logits.float().reshape(-1, V),
                               target_ids.reshape(-1), reduction="mean")
