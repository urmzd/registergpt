"""
v13: Sparse Register Machine with Factored Embedding.

Same sparse read/write routing as v12, but replaces one-hot initialization
with a learned factored embedding: token_id → d-dim → V-dim register space.

This gives the model a learned similarity structure so that related tokens
start with similar register activations, instead of being orthogonal.

Architecture:
  1. token_id → Embedding(V, embed_dim) → Linear(embed_dim, V) → register state
  2. N steps, each (identical to v12):
     a. READ:  gather k registers via pre-computed index buffer
     b. CROSS: causal decay memory in k-dim
     c. OP:    MLP transforms gathered → output (k-dim)
     d. WRITE: matmul with pre-computed selector matrix → V-dim delta
  3. Register state → softcap → cross-entropy loss

The embedding adds ~embed_dim * V * 2 parameters but solves the fundamental
problem of one-hot orthogonality: the model no longer needs to spend capacity
learning that similar tokens should have similar representations.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic_settings import BaseSettings
from torch import Tensor
from torch.utils.checkpoint import checkpoint as grad_checkpoint_fn

from core.base import AgiModel


class CausalDecayMemory(nn.Module):
    """Cross-position mixing via causal decay in the sparse subspace."""

    def __init__(self, dim: int, decay_init: float = 3.0):
        super().__init__()
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        for m in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.normal_(m.weight, std=0.02)

        self.decay_logit = nn.Parameter(torch.tensor(decay_init))
        self.out_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor) -> Tensor:
        B, T, k = x.shape
        dtype = x.dtype
        scale = 1.0 / math.sqrt(self.dim)

        q = self.q_proj(x.float()).to(dtype)
        k_ = self.k_proj(x.float()).to(dtype)
        v = self.v_proj(x.float()).to(dtype)

        scores = torch.bmm(q, k_.transpose(1, 2)) * scale

        decay = torch.sigmoid(self.decay_logit)
        pos = torch.arange(T, device=x.device)
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)
        causal = (diff > 0)
        weights = (decay ** (diff.float() - 1).clamp(min=0)) * causal
        scores = scores * weights.to(dtype).unsqueeze(0)

        retrieved = torch.bmm(scores, v)
        return self.o_proj(retrieved.float()).to(dtype) * self.out_scale.to(dtype)


class SparseRegisterStep(nn.Module):
    """One LGP instruction: READ src → transform → WRITE tgt."""

    def __init__(self, vocab_size: int, k_active: int, inner_mul: int = 2,
                 activation: str = "gelu", decay_init: float = 3.0,
                 step_idx: int = 0, total_steps: int = 1):
        super().__init__()
        self.vocab_size = vocab_size
        self.k_active = k_active

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
        self.register_buffer("write_indices", write_indices)
        self.register_buffer("write_selector", write_selector)

        self.memory = CausalDecayMemory(k_active, decay_init)

        inner_dim = k_active * inner_mul
        self.down = nn.Linear(k_active, inner_dim, bias=False)
        self.up = nn.Linear(inner_dim, k_active, bias=False)
        self.mlp_bias = nn.Parameter(torch.zeros(inner_dim))
        self.activation = activation

        init_std = 0.1
        nn.init.normal_(self.down.weight, std=init_std)
        nn.init.normal_(self.up.weight, std=init_std)

        self.inv_sqrt_k = 1.0 / math.sqrt(k_active)
        self.mem_scale = nn.Parameter(torch.ones(1))
        self.write_scale = nn.Parameter(torch.tensor(0.1))

    def _mlp(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        h = self.down(x.float()) + self.mlp_bias
        if self.activation == "relu":
            h = F.relu(h)
        elif self.activation == "relu2":
            h = F.relu(h).square()
        elif self.activation == "swish":
            h = F.silu(h)
        else:
            h = F.gelu(h)
        return self.up(h).to(dtype)

    def forward(self, x: Tensor) -> Tensor:
        B, T, V = x.shape
        dtype = x.dtype
        k = self.k_active

        read_idx = self.read_indices.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
        gathered = torch.gather(x, -1, read_idx)

        g_norm = F.rms_norm(gathered, (k,))
        gathered = gathered + self.mem_scale.to(dtype) * self.memory(g_norm)

        g_norm = F.rms_norm(gathered, (k,))
        output = self._mlp(g_norm)

        output = output * (self.write_scale.to(dtype) * self.inv_sqrt_k)

        return output @ self.write_selector.to(dtype)


def _compute_waves(steps: nn.ModuleList) -> list[list[int]]:
    """Group steps into parallel waves based on register set conflicts."""
    n = len(steps)
    read_sets = [set(steps[i].read_indices.tolist()) for i in range(n)]
    write_sets = [set(steps[i].write_indices.tolist()) for i in range(n)]

    def conflicts(i: int, j: int) -> bool:
        return bool(
            write_sets[i] & read_sets[j]
            or write_sets[j] & read_sets[i]
            or write_sets[i] & write_sets[j]
        )

    waves: list[list[int]] = []
    assigned = set()
    for i in range(n):
        if i in assigned:
            continue
        wave = [i]
        assigned.add(i)
        for j in range(i + 1, n):
            if j in assigned:
                continue
            if not any(conflicts(j, w) for w in wave):
                wave.append(j)
                assigned.add(j)
        waves.append(wave)
    return waves


class SparseEmbedGPT(AgiModel):
    """Sparse register machine with learned factored embedding.

    Identical to v12 SparseRegisterGPT except the one-hot init is replaced
    with: Embedding(V, embed_dim) → Linear(embed_dim, V).

    This gives tokens a learned similarity structure in register space.
    """

    version = "v13_embed"
    architecture = "Sparse embed"
    cross_position = "Causal decay (k-subspace)"
    within_position = "Factored embedding"

    class Settings(BaseSettings):
        vocab_size: int = 1024
        num_steps: int = 12
        embed_dim: int = 128
        k_active: int = 256
        inner_mul: int = 2
        logit_softcap: float = 30.0
        activation: str = "gelu"
        decay_init: float = 3.0
        parallel_waves: bool = True
        grad_checkpoint: bool = False

    def __init__(self, vocab_size: int = 1024, num_steps: int = 12,
                 embed_dim: int = 128, k_active: int = 256, inner_mul: int = 2,
                 logit_softcap: float = 30.0, activation: str = "gelu",
                 decay_init: float = 3.0, parallel_waves: bool = True,
                 grad_checkpoint: bool = False):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_steps = num_steps
        self.logit_softcap = logit_softcap
        self.use_parallel_waves = parallel_waves
        self.use_grad_checkpoint = grad_checkpoint

        # Factored embedding: token_id → embed_dim → vocab_size
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.embed_up = nn.Linear(embed_dim, vocab_size, bias=False)
        nn.init.normal_(self.tok_embed.weight, std=0.02)
        nn.init.normal_(self.embed_up.weight, std=0.02)

        self.steps = nn.ModuleList([
            SparseRegisterStep(vocab_size, k_active, inner_mul,
                               activation, decay_init,
                               step_idx=i, total_steps=num_steps)
            for i in range(num_steps)
        ])

        self.logit_scale = nn.Parameter(torch.tensor(1.0))

        self.waves = _compute_waves(self.steps)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        V = self.vocab_size

        # Factored embedding instead of one-hot
        x = self.tok_embed(input_ids)           # (B, T, embed_dim)
        x = self.embed_up(x)                     # (B, T, V)
        x = x.to(dtype=torch.bfloat16)
        x = F.rms_norm(x, (V,))

        if self.use_parallel_waves:
            for wave in self.waves:
                if self.use_grad_checkpoint:
                    deltas = [grad_checkpoint_fn(self.steps[i], x,
                              use_reentrant=False) for i in wave]
                else:
                    deltas = [self.steps[i](x) for i in wave]
                x = x + sum(deltas)
        else:
            for step in self.steps:
                if self.use_grad_checkpoint:
                    delta = grad_checkpoint_fn(step, x, use_reentrant=False)
                else:
                    delta = step(x)
                x = x + delta

        x = F.rms_norm(x, (V,))
        logits = x * self.logit_scale.to(x.dtype)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)

        return F.cross_entropy(logits.float().reshape(-1, V),
                               target_ids.reshape(-1), reduction="mean")
