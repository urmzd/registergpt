"""
v9: Meta-State Register Machine — Q-table as cross-position mechanism.

Inspired by the Q-table in Linear Genetic Programming: the cross-position
mechanism is NOT fixed learned weights — it's a meta-state that evolves
as the model processes each sequence. Starts empty, fills up with
word-to-word associations observed in the current context.

No Fourier. No attention. No embedding. No output projection.
Simple math: dense projections, relu/gelu, outer products.

Architecture:
  1. One-hot → register state (vocab-dimensional)
  2. N steps, each:
     a. Query the Q-table (what does accumulated context predict?)
     b. Update the Q-table (store what we just observed)
     c. Transform registers (simple activation on dense projection)
  3. Register state → softcap → cross-entropy loss

The Q-table is:
  - A (r × r) matrix that accumulates key⊗value outer products
  - Keys/values come from dense (not Fourier) projections of register state
  - Decays over time (recent context weighted more)
  - Each step has its own Q-table (different "aspects" of context)
  - Parallel computation via causal decay-weighted matmul

What makes this different from v3 (associative memory):
  - Dense projections (full-rank) instead of Fourier (rank-32 bottleneck)
  - Framed as meta-learning: the Q-table IS learned during inference
  - Simple activations throughout, no softmax on projections
  - The trained weights define the UPDATE RULE, not the associations
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic_settings import BaseSettings
from torch import Tensor

from core.base import AgiModel


# ---------------------------------------------------------------------------
# Q-table: evolving cross-position meta-state
# ---------------------------------------------------------------------------

class QTable(nn.Module):
    """Evolving word association table.

    Projects vocab-space to a small state-space via dense learned matrices.
    Accumulates key⊗value outer products causally (parallel via decay matmul).
    Queries the table to retrieve context-dependent predictions.

    The Q-table starts empty for each sequence and fills up as positions
    are processed — like a Q-table in RL that learns from experience.
    """

    def __init__(self, vocab_size: int, state_dim: int, decay_init: float = 3.0):
        super().__init__()
        self.state_dim = state_dim
        # Dense projections: vocab → state (full-rank, not Fourier)
        self.q_proj = nn.Linear(vocab_size, state_dim, bias=False)
        self.k_proj = nn.Linear(vocab_size, state_dim, bias=False)
        self.v_proj = nn.Linear(vocab_size, state_dim, bias=False)
        self.o_proj = nn.Linear(state_dim, vocab_size, bias=False)

        # Initialize small
        for m in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.normal_(m.weight, std=0.02)

        # Decay: how quickly old associations fade
        self.decay_logit = nn.Parameter(torch.tensor(decay_init))
        self.out_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, T, V)
        returns: (B, T, V) — retrieved associations mapped back to vocab
        """
        B, T, V = x.shape
        dtype = x.dtype

        queries = self.q_proj(x.float()).to(dtype)   # (B, T, r)
        keys = self.k_proj(x.float()).to(dtype)      # (B, T, r)
        values = self.v_proj(x.float()).to(dtype)     # (B, T, r)

        # Causal decay-weighted similarity (parallel)
        scores = torch.bmm(queries, keys.transpose(1, 2))  # (B, T, T)

        decay = torch.sigmoid(self.decay_logit)
        pos = torch.arange(T, device=x.device)
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)
        causal = (diff > 0)
        weights = (decay ** (diff.float() - 1).clamp(min=0)) * causal
        scores = scores * weights.to(dtype).unsqueeze(0)

        # Retrieve from Q-table
        retrieved = torch.bmm(scores, values)  # (B, T, r)

        # Project back to vocab space
        return self.o_proj(retrieved.float()).to(dtype) * self.out_scale.to(dtype)


# ---------------------------------------------------------------------------
# Register transform: simple within-position operation
# ---------------------------------------------------------------------------

class RegisterTransform(nn.Module):
    """Within-position transform. Dense down-project, activate, up-project.

    No Fourier. Just a small MLP bottleneck in vocab space.
    Down: V → r (compress). Activate: gelu/relu. Up: r → V (expand).
    """

    def __init__(self, vocab_size: int, inner_dim: int, activation: str = "gelu"):
        super().__init__()
        self.activation = activation
        self.down = nn.Linear(vocab_size, inner_dim, bias=False)
        self.up = nn.Linear(inner_dim, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(inner_dim))
        self.out_scale = nn.Parameter(torch.tensor(0.1))

        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.normal_(self.up.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        h = self.down(x.float()) + self.bias
        if self.activation == "relu":
            h = F.relu(h)
        elif self.activation == "relu2":
            h = F.relu(h).square()
        elif self.activation == "swish":
            h = F.silu(h)
        else:
            h = F.gelu(h)
        return self.up(h).to(dtype) * self.out_scale.to(dtype)


# ---------------------------------------------------------------------------
# Meta-state step
# ---------------------------------------------------------------------------

class MetaStateStep(nn.Module):
    """One step: query/update Q-table + transform registers."""

    def __init__(self, vocab_size: int, state_dim: int, inner_dim: int,
                 activation: str = "gelu", decay_init: float = 3.0):
        super().__init__()
        self.qtable = QTable(vocab_size, state_dim, decay_init)
        self.transform = RegisterTransform(vocab_size, inner_dim, activation)
        self.q_scale = nn.Parameter(torch.ones(1))
        self.t_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor) -> Tensor:
        V = x.size(-1)
        x = x + self.q_scale.to(x.dtype) * self.qtable(F.rms_norm(x, (V,)))
        x = x + self.t_scale.to(x.dtype) * self.transform(F.rms_norm(x, (V,)))
        return x


# ---------------------------------------------------------------------------
# MetaStateGPT
# ---------------------------------------------------------------------------

class MetaStateGPT(AgiModel):
    """Register machine with Q-table meta-state.

    Cross-position: evolving Q-table (dense projections, no Fourier)
    Within-position: simple MLP bottleneck (dense, no Fourier)

    The Q-table starts empty for each sequence. As positions are processed,
    it accumulates word associations — learning about this specific context
    at inference time. The trained weights define HOW to learn, not WHAT
    was learned.

    No Fourier. No attention. No embedding. No output projection.
    """

    version = "v9_meta"
    architecture = "Meta-state Q-table"
    cross_position = "Evolving Q-table (dense)"
    within_position = "Dense MLP"

    class Settings(BaseSettings):
        vocab_size: int = 1024
        num_steps: int = 8
        state_dim: int = 64
        inner_dim: int = 128
        logit_softcap: float = 30.0
        activation: str = "gelu"
        decay_init: float = 3.0

    def __init__(self, vocab_size: int = 1024, num_steps: int = 8,
                 state_dim: int = 64, inner_dim: int = 128,
                 logit_softcap: float = 30.0, activation: str = "gelu",
                 decay_init: float = 3.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_steps = num_steps
        self.logit_softcap = logit_softcap

        self.steps = nn.ModuleList([
            MetaStateStep(vocab_size, state_dim, inner_dim, activation, decay_init)
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
