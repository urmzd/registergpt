"""
v11: BrainWave — Five oscillatory primitives, composed simply.

Each brain wave type is a single, distinct operation:
  Delta: heavy-decay running average (broad context sync)
  Theta: causal decay memory (long-range retrieval)
  Alpha: learned gate (inhibition / noise suppression)
  Beta:  dense transform (active processing / thinking)
  Gamma: low-decay memory (sharp local binding)

No Fourier. No band-splitting. No coupling matrices.
Cross-frequency coupling emerges from sequential composition.

No attention. No embedding. No output projection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic_settings import BaseSettings
from torch import Tensor

from core.base import AgiModel


# ---------------------------------------------------------------------------
# Delta: broad context (exponential moving average)
# ---------------------------------------------------------------------------

class Delta(nn.Module):
    """Heavy-decay running average. Smooths register state over long spans.

    Like delta waves in deep sleep: slow, global synchronization.
    Implemented as causal EMA with learned per-channel decay rates.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.decay_logits = nn.Parameter(torch.full((dim,), 4.0))  # high init = slow decay
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        decay = torch.sigmoid(self.decay_logits)  # (D,) per-channel decay

        # Parallel scan via cumulative weighted sum
        # For each position t: ema[t] = sum_{s<t} decay^(t-s-1) * x[s]
        # Efficient: use log-space cumsum trick
        log_decay = torch.log(decay.clamp(min=1e-6))  # (D,)
        # Weight each position: w[t] = decay^(T-1-t)
        steps = torch.arange(T, device=x.device, dtype=x.dtype)
        # weights[t, d] = exp((T-1-t) * log_decay[d])
        weights = torch.exp(steps.flip(0).unsqueeze(-1) * log_decay.unsqueeze(0))  # (T, D)
        # Weighted x, cumsum, then unweight
        wx = x * weights.unsqueeze(0)  # (B, T, D)
        cum = torch.cumsum(wx, dim=1)
        # Shift right (causal: position t sees only s < t)
        cum = F.pad(cum[:, :-1], (0, 0, 1, 0))
        # Unweight
        result = cum / weights.unsqueeze(0).clamp(min=1e-8)
        return result * self.scale


# ---------------------------------------------------------------------------
# CausalMemory: shared Q/K/V/O projections, used by both Theta and Gamma
# ---------------------------------------------------------------------------

class CausalMemory(nn.Module):
    """Shared causal decay memory projections.

    Q/K/V/O are shared — Theta and Gamma differ only in decay rate.
    This mirrors the brain: same synaptic pathways, different oscillatory timing.
    """

    def __init__(self, vocab_size: int, state_dim: int):
        super().__init__()
        self.q = nn.Linear(vocab_size, state_dim, bias=False)
        self.k = nn.Linear(vocab_size, state_dim, bias=False)
        self.v = nn.Linear(vocab_size, state_dim, bias=False)
        self.o = nn.Linear(state_dim, vocab_size, bias=False)
        for m in [self.q, self.k, self.v, self.o]:
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: Tensor, decay_logit: Tensor, scale: Tensor) -> Tensor:
        B, T, V = x.shape
        dtype = x.dtype

        q = self.q(x.float()).to(dtype)
        k = self.k(x.float()).to(dtype)
        v = self.v(x.float()).to(dtype)

        scores = torch.bmm(q, k.transpose(1, 2))

        decay = torch.sigmoid(decay_logit)
        pos = torch.arange(T, device=x.device)
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)
        causal = (diff > 0)
        weights = (decay ** (diff.float() - 1).clamp(min=0)) * causal
        scores = scores * weights.to(dtype).unsqueeze(0)

        retrieved = torch.bmm(scores, v)
        return self.o(retrieved.float()).to(dtype) * scale


# ---------------------------------------------------------------------------
# Alpha: inhibition gate
# ---------------------------------------------------------------------------

class Alpha(nn.Module):
    """Learned sigmoid gate over register dimensions.

    Like alpha waves: cortical inhibition, suppressing irrelevant activity.
    Compresses state, computes gate, applies element-wise.
    Bias initialized so sigmoid ≈ 1 (no suppression at init).
    """

    def __init__(self, vocab_size: int, gate_dim: int):
        super().__init__()
        self.down = nn.Linear(vocab_size, gate_dim, bias=False)
        self.up = nn.Linear(gate_dim, vocab_size)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.normal_(self.up.weight, std=0.01)
        nn.init.constant_(self.up.bias, 2.0)  # sigmoid(2) ≈ 0.88, near-identity

    def forward(self, x: Tensor) -> Tensor:
        gate = torch.sigmoid(self.up(self.down(x.float())))
        return x * gate.to(x.dtype)


# ---------------------------------------------------------------------------
# Beta: active processing (dense transform)
# ---------------------------------------------------------------------------

class Beta(nn.Module):
    """Within-position dense transform. The 'thinking' operation.

    Like beta waves during active cognition.
    Down-project, activate, up-project. Simple MLP bottleneck.
    """

    def __init__(self, vocab_size: int, inner_dim: int):
        super().__init__()
        self.down = nn.Linear(vocab_size, inner_dim, bias=False)
        self.up = nn.Linear(inner_dim, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(inner_dim))
        self.scale = nn.Parameter(torch.tensor(0.1))
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.normal_(self.up.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        h = F.gelu(self.down(x.float()) + self.bias)
        return self.up(h).to(x.dtype) * self.scale


# ---------------------------------------------------------------------------
# BrainWave step: delta → theta → alpha → beta → gamma
# ---------------------------------------------------------------------------

class BrainWaveStep(nn.Module):
    """One oscillatory cycle: all five waves in sequence.

    Theta and Gamma share Q/K/V/O projections (same synaptic pathways,
    different oscillatory timing). Each step owns only its decay rates
    and scales — the heavy projection weights are shared.

    The ordering mirrors the brain's processing hierarchy:
      1. Delta: establish broad context
      2. Theta: retrieve relevant memories (high decay = long range)
      3. Alpha: suppress irrelevant activations
      4. Beta: active computation on what remains
      5. Gamma: bind local features (low decay = short range)
    """

    def __init__(self, vocab_size: int, memory: CausalMemory, inner_dim: int,
                 gate_dim: int):
        super().__init__()
        self.delta = Delta(vocab_size)
        self.memory = memory  # shared, not owned
        self.theta_decay = nn.Parameter(torch.tensor(4.0))  # high = long range
        self.theta_scale = nn.Parameter(torch.tensor(0.1))
        self.gamma_decay = nn.Parameter(torch.tensor(1.0))  # low = short range
        self.gamma_scale = nn.Parameter(torch.tensor(0.1))
        self.alpha = Alpha(vocab_size, gate_dim)
        self.beta = Beta(vocab_size, inner_dim)

    def forward(self, x: Tensor) -> Tensor:
        V = x.size(-1)
        x = x + self.delta(F.rms_norm(x, (V,)))
        x = x + self.memory(F.rms_norm(x, (V,)), self.theta_decay, self.theta_scale)
        x = self.alpha(x)
        x = x + self.beta(F.rms_norm(x, (V,)))
        x = x + self.memory(F.rms_norm(x, (V,)), self.gamma_decay, self.gamma_scale)
        return x


# ---------------------------------------------------------------------------
# BrainWaveGPT v11
# ---------------------------------------------------------------------------

class BrainWaveGPT(AgiModel):
    """Language model as composed brain oscillations.

    Five operations per step, each a direct analogue of a brain wave:
    delta (sync), theta (memory), alpha (gate), beta (think), gamma (bind).

    No Fourier. No attention. No embedding. No output projection.
    """

    version = "v11_brainwave"
    architecture = "BrainWave v2"
    cross_position = "EMA + causal decay"
    within_position = "Oscillatory primitives"

    class Settings(BaseSettings):
        vocab_size: int = 1024
        num_steps: int = 8
        state_dim: int = 64
        inner_dim: int = 128
        gate_dim: int = 64
        logit_softcap: float = 30.0

    def __init__(self, vocab_size: int = 1024, num_steps: int = 8,
                 state_dim: int = 64, inner_dim: int = 128,
                 gate_dim: int = 64, logit_softcap: float = 30.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_steps = num_steps
        self.logit_softcap = logit_softcap

        # Shared memory projections (Theta/Gamma share Q/K/V/O across all steps)
        self.memory = CausalMemory(vocab_size, state_dim)

        self.steps = nn.ModuleList([
            BrainWaveStep(vocab_size, self.memory, inner_dim, gate_dim)
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
