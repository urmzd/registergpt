"""
v7: True Linear Genetic Programming — Differentiable Register Machine.

Not "inspired by LGP" — this IS a register machine with a learned program.
Each instruction selects source registers, picks an operation from a bank,
and writes to destination registers. All via soft addressing (differentiable).

Key differences from all other versions:
- Operations are SELECTED from a bank, not a single fixed transform
- Register addressing is LEARNED (soft attention over register dims)
- The model learns a PROGRAM, not just weights
- Cross-position via associative memory (same as v3/v4)

Instruction format (per step):
  1. READ:   soft-select source registers via learned address weights
  2. SELECT: choose operation from bank via Gumbel-softmax
  3. APPLY:  run selected operation
  4. WRITE:  soft-select destination registers

The program is the sequence of instructions. Each instruction is unique.
No attention. No embedding. No output projection.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def make_fourier_basis(dim: int, n_basis: int) -> Tensor:
    pos = torch.arange(dim, dtype=torch.float32) / dim
    basis = torch.zeros(dim, 2 * n_basis)
    for k in range(n_basis):
        basis[:, 2 * k] = torch.cos(2 * math.pi * (k + 1) * pos)
        basis[:, 2 * k + 1] = torch.sin(2 * math.pi * (k + 1) * pos)
    return basis


# ---------------------------------------------------------------------------
# Primitive operations bank
# ---------------------------------------------------------------------------

class OpBank(nn.Module):
    """Bank of K primitive operations. Each is a small channel transform.

    The instruction selects which operation to apply via soft weights.
    This is the "opcode" of the LGP instruction.
    """

    def __init__(self, n_ops: int, n_channels: int):
        super().__init__()
        self.n_ops = n_ops
        # Each op is a (n_channels → n_channels) transform + bias
        self.op_weights = nn.Parameter(torch.randn(n_ops, n_channels, n_channels) * 0.02)
        self.op_biases = nn.Parameter(torch.zeros(n_ops, n_channels))
        # Fixed nonlinearities per op (identity, relu, gelu, square, negate, abs, tanh, sigmoid)
        self.nonlinearities = [
            lambda x: x,                    # identity / copy
            lambda x: F.relu(x),            # threshold
            lambda x: F.gelu(x),            # smooth threshold
            lambda x: x.square(),           # quadratic
            lambda x: -x,                   # negate
            lambda x: x.abs(),              # magnitude
            lambda x: torch.tanh(x),        # squash
            lambda x: torch.sigmoid(x),     # gate
        ]

    def forward(self, x: Tensor, op_select: Tensor) -> Tensor:
        """Apply soft-selected operation.

        x: (B, T, C) input
        op_select: (n_ops,) selection logits
        returns: (B, T, C)
        """
        weights = F.softmax(op_select, dim=-1)  # (n_ops,)
        dtype = x.dtype

        result = torch.zeros_like(x)
        for i in range(self.n_ops):
            # Linear transform
            h = x @ self.op_weights[i].to(dtype) + self.op_biases[i].to(dtype)
            # Apply this op's nonlinearity
            nl = self.nonlinearities[i % len(self.nonlinearities)]
            h = nl(h)
            result = result + weights[i] * h

        return result


# ---------------------------------------------------------------------------
# LGP Instruction
# ---------------------------------------------------------------------------

class LGPInstruction(nn.Module):
    """One instruction in the register machine program.

    READ  → SELECT OP → APPLY → WRITE

    Read/write use Fourier-parameterized soft addressing over vocab registers.
    Op selection uses learned logits over the operation bank.
    """

    def __init__(self, n_basis: int, n_channels: int, n_ops: int, op_bank: OpBank):
        super().__init__()
        s = 0.02
        # Soft register addressing (read/write patterns via Fourier)
        self.read_coeffs = nn.Parameter(torch.randn(n_channels, 2 * n_basis) * s)
        self.write_coeffs = nn.Parameter(torch.randn(n_channels, 2 * n_basis) * s)

        # Operation selection logits
        self.op_logits = nn.Parameter(torch.zeros(n_ops))

        # Shared op bank (not owned, just referenced)
        self.op_bank = op_bank

        self.out_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor, basis: Tensor) -> Tensor:
        dtype = x.dtype

        # READ: soft-select source registers via Fourier addressing
        read_w = torch.softmax(basis @ self.read_coeffs.T, dim=0).to(dtype)  # (V, C)
        values = x @ read_w  # (B, T, C)

        # SELECT + APPLY: choose and run operation from bank
        values = self.op_bank(values, self.op_logits)

        # WRITE: soft-select destination registers
        write_w = (basis @ self.write_coeffs.T).to(dtype)  # (V, C)
        return values @ write_w.T * self.out_scale.to(dtype)


# ---------------------------------------------------------------------------
# Cross-position: causal decay memory (same mechanism as v3)
# ---------------------------------------------------------------------------

class CausalMemory(nn.Module):
    """Cross-position mixing via causal decay-weighted associative memory."""

    def __init__(self, n_basis: int, n_channels: int, decay_init: float = 3.0):
        super().__init__()
        self.n_channels = n_channels
        s = 0.02
        self.q_coeffs = nn.Parameter(torch.randn(n_channels, 2 * n_basis) * s)
        self.k_coeffs = nn.Parameter(torch.randn(n_channels, 2 * n_basis) * s)
        self.v_coeffs = nn.Parameter(torch.randn(n_channels, 2 * n_basis) * s)
        self.o_coeffs = nn.Parameter(torch.randn(n_channels, 2 * n_basis) * s)
        self.decay_logit = nn.Parameter(torch.tensor(decay_init))
        self.out_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor, basis: Tensor) -> Tensor:
        B, T, V = x.shape
        dtype = x.dtype

        q_w = (basis @ self.q_coeffs.T).to(dtype)
        k_w = (basis @ self.k_coeffs.T).to(dtype)
        v_w = (basis @ self.v_coeffs.T).to(dtype)
        o_w = (basis @ self.o_coeffs.T).to(dtype)

        queries = x @ q_w
        keys = x @ k_w
        values = x @ v_w

        scores = torch.bmm(queries, keys.transpose(1, 2))

        decay = torch.sigmoid(self.decay_logit)
        pos = torch.arange(T, device=x.device)
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)
        causal = (diff > 0)
        weights = (decay ** (diff.float() - 1).clamp(min=0)) * causal
        scores = scores * weights.to(dtype).unsqueeze(0)

        retrieved = torch.bmm(scores, values)
        return retrieved @ o_w.T * self.out_scale.to(dtype)


# ---------------------------------------------------------------------------
# Program step: memory + instruction
# ---------------------------------------------------------------------------

class ProgramStep(nn.Module):
    """One step of program execution: cross-position memory + LGP instruction."""

    def __init__(self, n_basis: int, n_channels: int, n_ops: int,
                 op_bank: OpBank, decay_init: float = 3.0):
        super().__init__()
        self.memory = CausalMemory(n_basis, n_channels, decay_init)
        self.instruction = LGPInstruction(n_basis, n_channels, n_ops, op_bank)
        self.mem_scale = nn.Parameter(torch.ones(1))
        self.op_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor, basis: Tensor) -> Tensor:
        D = x.size(-1)
        x = x + self.mem_scale.to(x.dtype) * self.memory(
            F.rms_norm(x, (D,)), basis)
        x = x + self.op_scale.to(x.dtype) * self.instruction(
            F.rms_norm(x, (D,)), basis)
        return x


# ---------------------------------------------------------------------------
# LGPGPT: the full model
# ---------------------------------------------------------------------------

class LGPGPT(nn.Module):
    """Differentiable register machine with a learned program.

    The model IS a program: a sequence of instructions that read registers,
    select operations from a shared bank, and write results back.
    Cross-position mixing via causal decay memory.

    No attention. No embedding. No output projection.
    """

    def __init__(self, vocab_size: int = 1024, num_instructions: int = 16,
                 n_fourier_basis: int = 16, n_channels: int = 64,
                 n_ops: int = 8, logit_softcap: float = 30.0,
                 decay_init: float = 3.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_instructions = num_instructions
        self.logit_softcap = logit_softcap

        # Shared operation bank (like a CPU's ALU — same ops available to all instructions)
        self.op_bank = OpBank(n_ops, n_channels)

        # Program: sequence of instructions
        self.program = nn.ModuleList([
            ProgramStep(n_fourier_basis, n_channels, n_ops, self.op_bank, decay_init)
            for _ in range(num_instructions)
        ])

        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("fourier_basis",
                             make_fourier_basis(vocab_size, n_fourier_basis))

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        V = self.vocab_size
        x = F.one_hot(input_ids, V).to(dtype=torch.bfloat16)
        x = F.rms_norm(x, (V,))

        for step in self.program:
            x = step(x, self.fourier_basis)

        x = F.rms_norm(x, (V,))
        logits = x * self.logit_scale.to(x.dtype)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)

        return F.cross_entropy(logits.float().reshape(-1, V),
                               target_ids.reshape(-1), reduction="mean")
