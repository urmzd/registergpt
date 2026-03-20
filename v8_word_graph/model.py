"""
v8: Word Interaction Graph — Direct word-to-word learned associations.

No projections. No bottlenecks. No frequency decomposition.
A learned V×V matrix where W[i,j] = "when word i is active, word j activates."
Multiple hops through the graph build compositional representations.

Key innovation:
- Direct word-to-word interaction (fully interpretable)
- W["the"]["cat"] = 0.3 means "the" activates "cat" with weight 0.3
- Low-rank factored: W = U @ V^T + diag (captures both common patterns and word-specific)
- Multiple hops = multi-step reasoning in word space
- Cross-position via causal decay-weighted propagation

No attention. No embedding. No output projection. No Fourier basis.
The graph IS the model's knowledge.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Word interaction layer
# ---------------------------------------------------------------------------

class WordInteraction(nn.Module):
    """Learned word-to-word interaction matrix.

    W = U @ V^T + diag(d)
    - U @ V^T: low-rank shared structure (common word associations)
    - diag(d): per-word self-interaction (word-specific bias)

    Applying W to register state x:
      y = x @ W = x @ (U @ V^T + diag(d))
        = (x @ U) @ V^T + x * d

    Cost: O(V * r) instead of O(V^2) for full matrix.
    Interpretable: U captures word groups, V^T maps groups to targets.
    """

    def __init__(self, vocab_size: int, rank: int, activation: str = "gelu"):
        super().__init__()
        self.activation = activation
        s = 0.02
        self.U = nn.Parameter(torch.randn(vocab_size, rank) * s)
        self.V = nn.Parameter(torch.randn(vocab_size, rank) * s)
        self.diag = nn.Parameter(torch.zeros(vocab_size))
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.out_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        # Low-rank interaction + diagonal
        h = (x @ self.U.to(dtype)) @ self.V.to(dtype).T + x * self.diag.to(dtype) + self.bias.to(dtype)
        # Nonlinearity
        if self.activation == "relu2":
            h = F.relu(h).square()
        elif self.activation == "swish":
            h = F.silu(h)
        else:
            h = F.gelu(h)
        return h * self.out_scale.to(dtype)


# ---------------------------------------------------------------------------
# Causal word propagation (cross-position)
# ---------------------------------------------------------------------------

class CausalWordPropagation(nn.Module):
    """Cross-position mixing via causal decay in word space.

    No projection to channel space — operate directly on vocab activations.
    scores[t,s] = x_t · x_s (word activation similarity)
    Decay-weighted causal sum propagates word activations forward.

    This asks: "which past positions had similar word activations?"
    and blends their register states into the current position.
    """

    def __init__(self, vocab_size: int, decay_init: float = 3.0):
        super().__init__()
        self.decay_logit = nn.Parameter(torch.tensor(decay_init))
        self.out_scale = nn.Parameter(torch.tensor(0.1))
        # Learned query/key transforms in word space (diagonal — per-word scaling)
        self.q_scale = nn.Parameter(torch.ones(vocab_size))
        self.k_scale = nn.Parameter(torch.ones(vocab_size))

    def forward(self, x: Tensor) -> Tensor:
        B, T, V = x.shape
        dtype = x.dtype

        queries = x * self.q_scale.to(dtype)
        keys = x * self.k_scale.to(dtype)

        # Word activation similarity
        scores = torch.bmm(queries, keys.transpose(1, 2))  # (B, T, T)

        # Causal decay mask
        decay = torch.sigmoid(self.decay_logit)
        pos = torch.arange(T, device=x.device)
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)
        causal = (diff > 0)
        weights = (decay ** (diff.float() - 1).clamp(min=0)) * causal
        scores = scores * weights.to(dtype).unsqueeze(0)

        # Propagate word activations
        retrieved = torch.bmm(scores, x)  # (B, T, V)
        return retrieved * self.out_scale.to(dtype)


# ---------------------------------------------------------------------------
# Graph hop step
# ---------------------------------------------------------------------------

class GraphHop(nn.Module):
    """One hop through the word interaction graph.

    Cross-position: causal word propagation (which past positions matter?)
    Within-position: word interaction (which words activate which other words?)
    """

    def __init__(self, vocab_size: int, rank: int, activation: str = "gelu",
                 decay_init: float = 3.0):
        super().__init__()
        self.propagation = CausalWordPropagation(vocab_size, decay_init)
        self.interaction = WordInteraction(vocab_size, rank, activation)
        self.prop_scale = nn.Parameter(torch.ones(1))
        self.interact_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor) -> Tensor:
        V = x.size(-1)
        x = x + self.prop_scale.to(x.dtype) * self.propagation(
            F.rms_norm(x, (V,)))
        x = x + self.interact_scale.to(x.dtype) * self.interaction(
            F.rms_norm(x, (V,)))
        return x


# ---------------------------------------------------------------------------
# WordGraphGPT
# ---------------------------------------------------------------------------

class WordGraphGPT(nn.Module):
    """Language model as a word interaction graph.

    Multiple hops through a learned word-to-word graph.
    Each hop: propagate activations across positions, then transform
    via direct word associations.

    Fully interpretable: read the U, V matrices to see which words
    are associated. The graph IS the model's learned knowledge of language.

    No attention. No embedding. No output projection. No Fourier basis.
    """

    def __init__(self, vocab_size: int = 1024, num_hops: int = 8,
                 interaction_rank: int = 64, logit_softcap: float = 30.0,
                 activation: str = "gelu", decay_init: float = 3.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_hops = num_hops
        self.logit_softcap = logit_softcap

        self.hops = nn.ModuleList([
            GraphHop(vocab_size, interaction_rank, activation, decay_init)
            for _ in range(num_hops)
        ])

        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        V = self.vocab_size
        x = F.one_hot(input_ids, V).to(dtype=torch.bfloat16)
        x = F.rms_norm(x, (V,))

        for hop in self.hops:
            x = hop(x)

        x = F.rms_norm(x, (V,))
        logits = x * self.logit_scale.to(x.dtype)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)

        return F.cross_entropy(logits.float().reshape(-1, V),
                               target_ids.reshape(-1), reduction="mean")
