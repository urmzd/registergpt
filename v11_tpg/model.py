"""
v11: Neural TPG — Hard routing, multi-timescale memory, adaptive depth.

Three additions to v9 (meta-state register machine):

1. Winner-take-all routing: Gumbel-softmax hard routing on activation
   selection. During forward pass, one op fires. During backward pass,
   gradients flow through all via straight-through relaxation. Produces
   conditional behavior instead of average behavior.

2. Multi-timescale memory: 3 Q-tables per step with different decay rates
   (fast ~0.5, medium ~0.95, slow ~0.99). Captures bigram, paragraph, and
   document-level patterns simultaneously. Policy decides which temporal
   scale to read from (indexed read, differentiable).

3. Adaptive depth: Early-exit via halting probability (PonderNet-style).
   A tiny classifier inspects register state after each step and decides
   "is this token resolved?" Continuous relaxation during training, hard
   threshold at inference. Most common tokens exit early.

No Fourier. No attention. No embedding. No output projection.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic_settings import BaseSettings
from torch import Tensor

from core.base import AgiModel


# ---------------------------------------------------------------------------
# Gumbel-softmax hard routing
# ---------------------------------------------------------------------------

def gumbel_hard_route(logits: Tensor, tau: float = 1.0, hard: bool = True) -> Tensor:
    """Gumbel-softmax with straight-through hard routing.

    Returns one-hot during forward (hard=True), but gradients flow
    through the soft relaxation.
    """
    if hard:
        gumbels = -torch.empty_like(logits).exponential_().log()
        y_soft = F.softmax((logits + gumbels) / tau, dim=-1)
        index = y_soft.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
        return y_hard - y_soft.detach() + y_soft  # straight-through
    else:
        gumbels = -torch.empty_like(logits).exponential_().log()
        return F.softmax((logits + gumbels) / tau, dim=-1)


# ---------------------------------------------------------------------------
# Op bank with hard routing
# ---------------------------------------------------------------------------

class HardOpBank(nn.Module):
    """Bank of primitive operations with winner-take-all selection."""

    def __init__(self, n_ops: int, dim: int):
        super().__init__()
        self.n_ops = n_ops
        self.op_transforms = nn.ModuleList([
            nn.Linear(dim, dim, bias=False) for _ in range(n_ops)
        ])
        self.activations = [
            F.gelu,
            F.relu,
            lambda x: F.relu(x).square(),
            F.silu,
            torch.tanh,
            torch.sigmoid,
            lambda x: x,
            lambda x: -x,
        ]
        for m in self.op_transforms:
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: Tensor, op_logits: Tensor, tau: float = 1.0) -> Tensor:
        """Apply hard-routed operation.

        x: (B, T, d)
        op_logits: (B, T, n_ops) raw logits from policy
        tau: Gumbel temperature (anneal during training)
        """
        hard = not self.training or True  # always hard route
        route = gumbel_hard_route(op_logits, tau=tau, hard=True)

        result = torch.zeros_like(x)
        for i, (transform, act) in enumerate(zip(self.op_transforms, self.activations)):
            h = act(transform(x.float())).to(x.dtype)
            result = result + route[..., i:i+1] * h
        return result


# ---------------------------------------------------------------------------
# Multi-timescale Q-table
# ---------------------------------------------------------------------------

class MultiScaleQTable(nn.Module):
    """Three Q-tables with different temporal decay rates.

    Fast (gamma ~0.5): bigram/trigram patterns
    Medium (gamma ~0.95): paragraph-level topic
    Slow (gamma ~0.99): document-level context

    A learned gate decides which timescale to read from at each position.
    """

    def __init__(self, vocab_size: int, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.n_scales = 3

        # Per-scale projections (separate k/v, shared q for efficiency)
        self.q_proj = nn.Linear(vocab_size, state_dim, bias=False)
        self.k_projs = nn.ModuleList([
            nn.Linear(vocab_size, state_dim, bias=False) for _ in range(3)
        ])
        self.v_projs = nn.ModuleList([
            nn.Linear(vocab_size, state_dim, bias=False) for _ in range(3)
        ])
        self.o_proj = nn.Linear(state_dim, vocab_size, bias=False)

        for m in [self.q_proj, self.o_proj] + list(self.k_projs) + list(self.v_projs):
            nn.init.normal_(m.weight, std=0.02)

        # Fixed decay inits: fast, medium, slow
        # sigmoid(x) -> decay rate. sigmoid(0)=0.5, sigmoid(3)=0.95, sigmoid(5)=0.99
        self.decay_logits = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0)),   # fast: gamma ~0.5
            nn.Parameter(torch.tensor(3.0)),   # medium: gamma ~0.95
            nn.Parameter(torch.tensor(5.0)),   # slow: gamma ~0.99
        ])

        # Scale selection gate: which timescale to read from
        self.scale_gate = nn.Linear(vocab_size, 3, bias=False)
        nn.init.normal_(self.scale_gate.weight, std=0.02)

        self.out_scale = nn.Parameter(torch.tensor(0.1))

    def _causal_retrieval(self, queries: Tensor, keys: Tensor, values: Tensor,
                          decay_logit: Tensor) -> Tensor:
        B, T, _ = queries.shape
        dtype = queries.dtype

        scores = torch.bmm(queries, keys.transpose(1, 2))

        decay = torch.sigmoid(decay_logit)
        pos = torch.arange(T, device=queries.device)
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)
        causal = (diff > 0)
        weights = (decay ** (diff.float() - 1).clamp(min=0)) * causal
        scores = scores * weights.to(dtype).unsqueeze(0)

        return torch.bmm(scores, values)

    def forward(self, x: Tensor) -> Tensor:
        B, T, V = x.shape
        dtype = x.dtype

        queries = self.q_proj(x.float()).to(dtype)

        # Scale selection (which timescale matters here?)
        scale_w = F.softmax(self.scale_gate(x.float()).to(dtype), dim=-1)  # (B, T, 3)

        # Retrieve from each timescale and blend via gate
        blended = torch.zeros(B, T, self.state_dim, device=x.device, dtype=dtype)
        for i in range(3):
            keys = self.k_projs[i](x.float()).to(dtype)
            values = self.v_projs[i](x.float()).to(dtype)
            retrieved = self._causal_retrieval(queries, keys, values, self.decay_logits[i])
            blended = blended + scale_w[..., i:i+1] * retrieved

        return self.o_proj(blended.float()).to(dtype) * self.out_scale.to(dtype)


# ---------------------------------------------------------------------------
# Register transform with hard-routed activation
# ---------------------------------------------------------------------------

class HardRouteTransform(nn.Module):
    """Within-position transform with hard activation routing.

    Instead of always using gelu, the model selects which activation
    to apply based on the input state. Different tokens get different
    nonlinearities.
    """

    def __init__(self, vocab_size: int, inner_dim: int, n_acts: int = 4):
        super().__init__()
        self.n_acts = n_acts
        self.down = nn.Linear(vocab_size, inner_dim, bias=False)
        self.up = nn.Linear(inner_dim, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(inner_dim))
        self.out_scale = nn.Parameter(torch.tensor(0.1))

        # Activation selector
        self.act_selector = nn.Linear(vocab_size, n_acts, bias=False)
        nn.init.normal_(self.act_selector.weight, std=0.02)

        self.activations = [F.gelu, F.relu, F.silu, torch.tanh]

        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.normal_(self.up.weight, std=0.02)

    def forward(self, x: Tensor, tau: float = 1.0) -> Tensor:
        dtype = x.dtype
        h = self.down(x.float()) + self.bias

        # Hard-route activation selection
        act_logits = self.act_selector(x.float()).to(dtype)  # (B, T, n_acts)
        route = gumbel_hard_route(act_logits, tau=tau, hard=True)

        result = torch.zeros_like(h).to(dtype)
        for i, act in enumerate(self.activations):
            activated = act(h).to(dtype)
            result = result + route[..., i:i+1] * activated

        return self.up(result.float()).to(dtype) * self.out_scale.to(dtype)


# ---------------------------------------------------------------------------
# Halting mechanism (adaptive depth)
# ---------------------------------------------------------------------------

class HaltingUnit(nn.Module):
    """Decides whether to stop processing at this step.

    PonderNet-style: outputs a halting probability. During training,
    we use the geometric distribution formulation. During inference,
    we hard-threshold.
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        self.halt_proj = nn.Linear(vocab_size, 1, bias=True)
        nn.init.normal_(self.halt_proj.weight, std=0.01)
        nn.init.constant_(self.halt_proj.bias, -2.0)  # bias toward NOT halting early

    def forward(self, x: Tensor) -> Tensor:
        """Returns halting probability per position. (B, T, 1)"""
        return torch.sigmoid(self.halt_proj(x.float())).to(x.dtype)


# ---------------------------------------------------------------------------
# TPG Step: Q-table + hard-routed transform
# ---------------------------------------------------------------------------

class TPGStep(nn.Module):
    """One step: multi-scale Q-table + hard-routed transform."""

    def __init__(self, vocab_size: int, state_dim: int, inner_dim: int):
        super().__init__()
        self.qtable = MultiScaleQTable(vocab_size, state_dim)
        self.transform = HardRouteTransform(vocab_size, inner_dim)
        self.q_scale = nn.Parameter(torch.ones(1))
        self.t_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor, tau: float = 1.0) -> Tensor:
        V = x.size(-1)
        x = x + self.q_scale.to(x.dtype) * self.qtable(F.rms_norm(x, (V,)))
        x = x + self.t_scale.to(x.dtype) * self.transform(F.rms_norm(x, (V,)), tau=tau)
        return x


# ---------------------------------------------------------------------------
# TPGGPT: the full model
# ---------------------------------------------------------------------------

class TPGGPT(AgiModel):
    """Neural TPG for language modeling.

    - Hard Gumbel routing on activation selection (conditional behavior)
    - 3 decay rates per step (fast/medium/slow temporal memory)
    - Adaptive depth via halting probability (PonderNet)

    No Fourier. No attention. No embedding. No output projection.
    """

    version = "v11_tpg"
    architecture = "Neural TPG"
    cross_position = "Multi-scale Q-table (3 decays)"
    within_position = "Hard Gumbel routing"

    class Settings(BaseSettings):
        vocab_size: int = 1024
        num_steps: int = 8
        state_dim: int = 64
        inner_dim: int = 128
        logit_softcap: float = 30.0
        tau: float = 1.0
        halt_threshold: float = 0.5
        ponder_lambda: float = 0.01

    @classmethod
    def build_kwargs(cls, args) -> dict:
        kw = super().build_kwargs(args)
        if hasattr(args, 'gumbel_tau') and 'tau' not in kw:
            kw['tau'] = args.gumbel_tau
        return kw

    def __init__(self, vocab_size: int = 1024, num_steps: int = 8,
                 state_dim: int = 64, inner_dim: int = 128,
                 logit_softcap: float = 30.0, tau: float = 1.0,
                 halt_threshold: float = 0.5,
                 ponder_lambda: float = 0.01):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_steps = num_steps
        self.logit_softcap = logit_softcap
        self.tau = tau
        self.halt_threshold = halt_threshold
        self.ponder_lambda = ponder_lambda

        self.steps = nn.ModuleList([
            TPGStep(vocab_size, state_dim, inner_dim)
            for _ in range(num_steps)
        ])

        self.halting = HaltingUnit(vocab_size)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        V = self.vocab_size
        x = F.one_hot(input_ids, V).to(dtype=torch.bfloat16)
        x = F.rms_norm(x, (V,))

        B, T, _ = x.shape
        device = x.device
        dtype = x.dtype

        if self.training:
            # PonderNet: accumulate outputs weighted by halting probability
            # p_n = halt_prob_n * prod(1 - halt_prob_k, k<n)
            accum = torch.zeros_like(x)
            remainder = torch.ones(B, T, 1, device=device, dtype=dtype)
            ponder_cost = torch.zeros(B, T, 1, device=device, dtype=dtype)

            for i, step in enumerate(self.steps):
                x = step(x, tau=self.tau)
                halt_prob = self.halting(x)  # (B, T, 1)

                if i < self.num_steps - 1:
                    p_n = remainder * halt_prob
                    remainder = remainder * (1.0 - halt_prob)
                else:
                    # Last step gets all remaining probability
                    p_n = remainder

                accum = accum + p_n * x
                ponder_cost = ponder_cost + remainder

            x = accum
        else:
            # Inference: hard early exit
            halted = torch.zeros(B, T, 1, device=device, dtype=torch.bool)

            for i, step in enumerate(self.steps):
                # Only process non-halted positions
                x_new = step(x, tau=self.tau)
                # Blend: halted positions keep old state, others update
                x = torch.where(halted, x, x_new)

                if i < self.num_steps - 1:
                    halt_prob = self.halting(x)
                    newly_halted = halt_prob > self.halt_threshold
                    halted = halted | newly_halted

        x = F.rms_norm(x, (V,))
        logits = x * self.logit_scale.to(x.dtype)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)

        loss = F.cross_entropy(logits.float().reshape(-1, V),
                               target_ids.reshape(-1), reduction="mean")

        # Add ponder regularization during training
        if self.training:
            loss = loss + self.ponder_lambda * ponder_cost.mean()

        return loss
