"""
RegisterGPT v3 — Register machine with associative memory, PyTorch/CUDA training.
Outer-product memory for cross-position mixing. Fourier ops for within-position transforms.
No attention. No embedding. No output projection.
Compatible with torchrun for multi-GPU training.
"""
from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Model
    model_version = os.environ.get("MODEL_VERSION", "v3")
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_steps = int(os.environ.get("NUM_STEPS", 8))
    n_fourier_basis = int(os.environ.get("N_FOURIER_BASIS", 16))
    n_channels = int(os.environ.get("N_CHANNELS", 128))
    activation = os.environ.get("ACTIVATION", "gelu")
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    decay_init = float(os.environ.get("DECAY_INIT", 3.0))

    # v4-specific
    n_heads = int(os.environ.get("N_HEADS", 4))
    transform_rank = int(os.environ.get("TRANSFORM_RANK", 8))
    unique_steps = int(os.environ.get("UNIQUE_STEPS", 5))
    invocations_per_step = int(os.environ.get("INVOCATIONS_PER_STEP", 2))

    # wave-specific
    slow_decay_init = float(os.environ.get("SLOW_DECAY_INIT", 4.0))
    fast_decay_init = float(os.environ.get("FAST_DECAY_INIT", 2.0))
    band_split = os.environ.get("BAND_SPLIT", "4,4,8")

    # Optimizer
    lr = float(os.environ.get("LR", 0.03))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.999))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", 0.0))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 1.0))


# Patterns for control tensors (kept in float32)
CONTROL_TENSOR_NAME_PATTERNS = (
    "mem_scale", "op_scale", "read_coeffs", "write_coeffs",
    "channel_mix", "bias", "out_scale", "logit_scale", "decay_logit",
    "coeffs",
    # v4 additions
    "diag", "mix_down", "mix_up", "decay_logits", "_override",
    # gauss additions
    "freq_to_ch", "ch_to_freq", "weight",
)

# -----------------------------
# VALIDATION + QUANTIZATION
# -----------------------------

def build_sentencepiece_luts(sp, vocab_size, device):
    sv = int(sp.vocab_size())
    ts = max(sv, vocab_size)
    bb = np.zeros((ts,), dtype=np.int16)
    hs = np.zeros((ts,), dtype=np.bool_)
    ib = np.ones((ts,), dtype=np.bool_)
    for tid in range(sv):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        ib[tid] = False
        if sp.is_byte(tid):
            bb[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("▁"):
            hs[tid] = True
            piece = piece[1:]
        bb[tid] = len(piece.encode("utf-8"))
    return (torch.tensor(bb, dtype=torch.int16, device=device),
            torch.tensor(hs, dtype=torch.bool, device=device),
            torch.tensor(ib, dtype=torch.bool, device=device))


def load_data_shard(file):
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Bad header: {file}")
    n = int(header[2])
    tokens = np.fromfile(file, dtype="<u2", count=n, offset=256 * 4)
    return torch.from_numpy(tokens.astype(np.uint16, copy=False))


def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[:usable + 1]


class TokenStream:
    def __init__(self, pattern):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n):
        chunks = []
        left = n
        while left > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance()
                continue
            k = min(left, avail)
            chunks.append(self.tokens[self.pos:self.pos + k])
            self.pos += k
            left -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern, rank, world_size, device):
        self.rank, self.world_size, self.device = rank, world_size, device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens, seq_len, grad_accum_steps):
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        span = local_tokens + 1
        chunk = self.stream.take(span * self.world_size)
        start = self.rank * span
        local = chunk[start:start + span].to(dtype=torch.int64)
        x, y = local[:-1].reshape(-1, seq_len), local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


def eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, bbl, hsl, ibl):
    lbt = args.val_batch_size // (world_size * grad_accum_steps)
    if lbt < args.train_seq_len:
        raise ValueError("VAL_BATCH_SIZE too small")
    lbs = lbt // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    ss, se = (total_seqs * rank) // world_size, (total_seqs * (rank + 1)) // world_size
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    tok_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bs in range(ss, se, lbs):
            be = min(bs + lbs, se)
            local = val_tokens[bs * args.train_seq_len:be * args.train_seq_len + 1].to(device=device, dtype=torch.int64)
            x, y = local[:-1].reshape(-1, args.train_seq_len), local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                bl = model(x, y).detach()
            ct = float(y.numel())
            loss_sum += bl.to(torch.float64) * ct
            tok_count += ct
            tb = bbl[y.reshape(-1)].to(torch.float64)
            tb += (hsl[y.reshape(-1)] & ~ibl[x.reshape(-1)]).to(torch.float64)
            byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, tok_count, byte_count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    vl = (loss_sum / tok_count).item()
    model.train()
    return vl, vl / math.log(2.0) * (tok_count.item() / byte_count.item())


INT8_CLIP_Q = 99.99984 / 100.0
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536

def quantize_state_dict_int8(sd):
    quantized, scales, dtypes, passthrough = {}, {}, {}, {}
    passthrough_orig_dtypes, qmeta = {}, {}
    stats = dict.fromkeys(("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"), 0)
    for name, tensor in sd.items():
        t = tensor.detach().cpu().contiguous()
        stats["param_count"] += t.numel()
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += t.numel() * t.element_size()
        if not t.is_floating_point():
            passthrough[name] = t
            stats["int8_payload_bytes"] += t.numel() * t.element_size()
            continue
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
                kept = t.float().contiguous()
            elif t.dtype in {torch.float32, torch.bfloat16}:
                passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
                kept = t.to(torch.float16).contiguous()
            else:
                kept = t
            passthrough[name] = kept
            stats["int8_payload_bytes"] += kept.numel() * kept.element_size()
            continue
        stats["num_float_tensors"] += 1
        t32 = t.float()
        if t32.ndim == 2:
            ca = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            clipped = torch.clamp(t32, -ca[:, None], ca[:, None])
            s = (ca / 127.0).clamp_min(1.0 / 127.0)
            q = torch.clamp(torch.round(clipped / s[:, None]), -127, 127).to(torch.int8)
            qmeta[name] = {"scheme": "per_row", "axis": 0}
            scales[name] = s.to(torch.float16).contiguous()
        else:
            ca = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
            s = torch.tensor(ca / 127.0 if ca > 0 else 1.0, dtype=torch.float32)
            q = torch.clamp(torch.round(torch.clamp(t32, -ca, ca) / s), -127, 127).to(torch.int8)
            scales[name] = s
        quantized[name] = q.contiguous()
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += q.numel() + (scales[name].numel() * scales[name].element_size())
    obj = {"__quant_format__": "int8_clean_per_row_v1", "quantized": quantized, "scales": scales, "dtypes": dtypes, "passthrough": passthrough}
    if qmeta: obj["qmeta"] = qmeta
    if passthrough_orig_dtypes: obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int8(obj):
    out = {}
    qmeta = obj.get("qmeta", {})
    pod = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dt = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name].to(torch.float32)
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dt)
        else:
            out[name] = (q.float() * s.item()).to(dt)
    for name, t in obj["passthrough"].items():
        od = pod.get(name)
        out[name] = t.to(getattr(torch, od)) if isinstance(od, str) else t.detach().cpu()
    return out


# -----------------------------
# MODEL
# -----------------------------

def apply_activation(x, activation):
    if activation == "gelu":
        return F.gelu(x)
    elif activation == "relu2":
        return F.relu(x).square()
    elif activation == "swish":
        return F.silu(x)
    return F.gelu(x)


class FourierProjection(nn.Module):
    """Project vocab-space to channels via Fourier basis coefficients."""
    def __init__(self, n_basis, n_channels, soft=False):
        super().__init__()
        self.soft = soft
        self.coeffs = nn.Parameter(torch.randn(n_channels, 2 * n_basis) * 0.02)

    def forward(self, basis):
        w = basis @ self.coeffs.T
        if self.soft:
            w = torch.softmax(w, dim=0)
        return w


class AssociativeMemoryStep(nn.Module):
    """Cross-position mixing via causal decay-weighted associative memory (parallel)."""
    def __init__(self, n_basis, n_channels, decay_init=3.0):
        super().__init__()
        self.n_channels = n_channels
        self.query_proj = FourierProjection(n_basis, n_channels, soft=False)
        self.key_proj = FourierProjection(n_basis, n_channels, soft=False)
        self.value_proj = FourierProjection(n_basis, n_channels, soft=False)
        self.output_proj = FourierProjection(n_basis, n_channels, soft=False)
        self.decay_logit = nn.Parameter(torch.tensor(decay_init))
        self.out_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, basis):
        B, T, V = x.shape
        dtype = x.dtype

        q_w = self.query_proj(basis).to(dtype)
        k_w = self.key_proj(basis).to(dtype)
        v_w = self.value_proj(basis).to(dtype)
        o_w = self.output_proj(basis).to(dtype)

        queries = x @ q_w
        keys = x @ k_w
        values = x @ v_w

        # Content similarity in channel space
        scores = torch.bmm(queries, keys.transpose(1, 2))  # (B, T, T)

        # Causal decay mask: decay^(t-s-1) for s < t, 0 otherwise
        decay = torch.sigmoid(self.decay_logit)
        pos = torch.arange(T, device=x.device)
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # (T, T)
        causal_mask = (diff > 0)
        decay_weights = (decay ** (diff.float() - 1).clamp(min=0)) * causal_mask
        scores = scores * decay_weights.to(dtype).unsqueeze(0)

        # Retrieve: weighted sum of values
        retrieved = torch.bmm(scores, values)  # (B, T, C)

        return retrieved @ o_w.T * self.out_scale.to(dtype)


class FourierRegisterOp(nn.Module):
    """Within-position register transform."""
    def __init__(self, n_basis, n_channels, activation="gelu"):
        super().__init__()
        self.activation = activation
        s = 0.02
        self.read_coeffs = nn.Parameter(torch.randn(n_channels, 2 * n_basis) * s)
        self.write_coeffs = nn.Parameter(torch.randn(n_channels, 2 * n_basis) * s)
        self.channel_mix = nn.Parameter(torch.randn(n_channels, n_channels) * s)
        self.bias = nn.Parameter(torch.zeros(n_channels))
        self.out_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, basis):
        read_p = torch.softmax(basis @ self.read_coeffs.T, dim=0).to(x.dtype)
        values = x @ read_p
        values = values @ self.channel_mix.to(x.dtype) + self.bias.to(x.dtype)
        values = apply_activation(values, self.activation)
        write_p = (basis @ self.write_coeffs.T).to(x.dtype)
        return values @ write_p.T * self.out_scale.to(x.dtype)


class RegisterStep(nn.Module):
    """One LGP instruction: memory read/write + register transform."""
    def __init__(self, n_basis, n_channels, activation="gelu", decay_init=3.0):
        super().__init__()
        self.memory = AssociativeMemoryStep(n_basis, n_channels, decay_init)
        self.register_op = FourierRegisterOp(n_basis, n_channels, activation)
        self.mem_scale = nn.Parameter(torch.ones(1))
        self.op_scale = nn.Parameter(torch.ones(1))

    def forward(self, x, basis):
        D = x.size(-1)
        x = x + self.mem_scale.to(x.dtype) * self.memory(F.rms_norm(x, (D,)), basis)
        x = x + self.op_scale.to(x.dtype) * self.register_op(F.rms_norm(x, (D,)), basis)
        return x


class AssocRegisterLM(nn.Module):
    """Register machine with associative memory. No attention. No embedding."""
    def __init__(self, vocab_size, num_steps, n_fourier_basis, n_channels,
                 logit_softcap, activation="gelu", decay_init=0.95):
        super().__init__()
        dim = vocab_size
        self.vocab_size = vocab_size
        self.num_steps = num_steps
        self.logit_softcap = logit_softcap

        self.steps = nn.ModuleList([
            RegisterStep(n_fourier_basis, n_channels, activation, decay_init)
            for _ in range(num_steps)
        ])
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

        positions = torch.arange(dim, dtype=torch.float32) / dim
        basis = torch.zeros(dim, 2 * n_fourier_basis)
        for k in range(n_fourier_basis):
            basis[:, 2 * k] = torch.cos(2 * math.pi * (k + 1) * positions)
            basis[:, 2 * k + 1] = torch.sin(2 * math.pi * (k + 1) * positions)
        self.register_buffer("fourier_basis", basis, persistent=True)

    def forward(self, input_ids, target_ids):
        V = self.vocab_size
        x = F.one_hot(input_ids, V).to(dtype=torch.bfloat16)
        x = F.rms_norm(x, (V,))
        basis = self.fourier_basis

        for step in self.steps:
            x = step(x, basis)

        x = F.rms_norm(x, (V,))
        logits = x * self.logit_scale.to(x.dtype)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return F.cross_entropy(logits.float().reshape(-1, V), target_ids.reshape(-1), reduction="mean")


# -----------------------------
# GAUSS MODEL (FFT-based)
# -----------------------------

class GaussProjection(nn.Module):
    """Project vocab-space to channels via FFT."""
    def __init__(self, n_freq, n_channels):
        super().__init__()
        self.n_freq = n_freq
        self.weight = nn.Parameter(torch.randn(n_channels, 2 * n_freq) * 0.02)

    def forward(self, x):
        X = torch.fft.rfft(x.float(), dim=-1)
        X = X[..., 1:self.n_freq + 1]
        X_ri = torch.cat([X.real, X.imag], dim=-1)
        return (X_ri @ self.weight.T).to(x.dtype)


class GaussSynthesis(nn.Module):
    """Project channels back to vocab-space via IFFT."""
    def __init__(self, n_freq, n_channels, vocab_size):
        super().__init__()
        self.n_freq = n_freq
        self.vocab_size = vocab_size
        self.weight = nn.Parameter(torch.randn(2 * n_freq, n_channels) * 0.02)

    def forward(self, h):
        n, V = self.n_freq, self.vocab_size
        Y_ri = h.float() @ self.weight
        Y = torch.complex(Y_ri[..., :n], Y_ri[..., n:])
        shape = list(h.shape[:-1]) + [V // 2 + 1]
        full = torch.zeros(shape, dtype=Y.dtype, device=h.device)
        full[..., 1:n + 1] = Y
        return torch.fft.irfft(full, n=V, dim=-1).to(h.dtype)


class GaussMemoryStep(nn.Module):
    """Cross-position mixing via FFT projections + causal decay memory."""
    def __init__(self, n_freq, n_channels, vocab_size, decay_init=3.0):
        super().__init__()
        self.n_channels = n_channels
        self.q_proj = GaussProjection(n_freq, n_channels)
        self.k_proj = GaussProjection(n_freq, n_channels)
        self.v_proj = GaussProjection(n_freq, n_channels)
        self.o_proj = GaussSynthesis(n_freq, n_channels, vocab_size)
        self.decay_logit = nn.Parameter(torch.tensor(decay_init))
        self.out_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        B, T, V = x.shape
        dtype = x.dtype
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        scores = torch.bmm(queries, keys.transpose(1, 2))
        decay = torch.sigmoid(self.decay_logit)
        pos = torch.arange(T, device=x.device)
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)
        causal = (diff > 0)
        weights = (decay ** (diff.float() - 1).clamp(min=0)) * causal
        scores = scores * weights.to(dtype).unsqueeze(0)
        retrieved = torch.bmm(scores, values)
        return self.o_proj(retrieved) * self.out_scale.to(dtype)


class GaussRegisterOp(nn.Module):
    """Within-position transform via FFT → channels → IFFT."""
    def __init__(self, n_freq, n_channels, vocab_size, activation="gelu"):
        super().__init__()
        self.activation = activation
        self.n_freq = n_freq
        self.vocab_size = vocab_size
        s = 0.02
        self.freq_to_ch = nn.Parameter(torch.randn(n_channels, 2 * n_freq) * s)
        self.channel_mix = nn.Parameter(torch.randn(n_channels, n_channels) * s)
        self.bias = nn.Parameter(torch.zeros(n_channels))
        self.ch_to_freq = nn.Parameter(torch.randn(2 * n_freq, n_channels) * s)
        self.out_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        B, T, V = x.shape
        dtype = x.dtype
        n = self.n_freq
        X = torch.fft.rfft(x.float(), dim=-1)
        X = X[..., 1:n + 1]
        X_ri = torch.cat([X.real, X.imag], dim=-1)
        h = X_ri @ self.freq_to_ch.T
        h = h @ self.channel_mix + self.bias
        h = apply_activation(h, self.activation)
        Y_ri = h @ self.ch_to_freq.T
        Y = torch.complex(Y_ri[..., :n], Y_ri[..., n:])
        full = torch.zeros(B, T, V // 2 + 1, dtype=Y.dtype, device=x.device)
        full[..., 1:n + 1] = Y
        return torch.fft.irfft(full, n=V, dim=-1).to(dtype) * self.out_scale.to(dtype)


class GaussRegisterStep(nn.Module):
    """One step: FFT memory + FFT register op."""
    def __init__(self, n_freq, n_channels, vocab_size, activation="gelu", decay_init=3.0):
        super().__init__()
        self.memory = GaussMemoryStep(n_freq, n_channels, vocab_size, decay_init)
        self.register_op = GaussRegisterOp(n_freq, n_channels, vocab_size, activation)
        self.mem_scale = nn.Parameter(torch.ones(1))
        self.op_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        D = x.size(-1)
        x = x + self.mem_scale.to(x.dtype) * self.memory(F.rms_norm(x, (D,)))
        x = x + self.op_scale.to(x.dtype) * self.register_op(F.rms_norm(x, (D,)))
        return x


class GaussRegisterGPT(nn.Module):
    """Register machine with FFT-based operations. No stored Fourier basis."""
    def __init__(self, vocab_size, num_steps, n_freq, n_channels,
                 logit_softcap, activation="gelu", decay_init=3.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_steps = num_steps
        self.logit_softcap = logit_softcap
        self.steps = nn.ModuleList([
            GaussRegisterStep(n_freq, n_channels, vocab_size, activation, decay_init)
            for _ in range(num_steps)
        ])
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, target_ids):
        V = self.vocab_size
        x = F.one_hot(input_ids, V).to(dtype=torch.bfloat16)
        x = F.rms_norm(x, (V,))
        for step in self.steps:
            x = step(x)
        x = F.rms_norm(x, (V,))
        logits = x * self.logit_scale.to(x.dtype)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return F.cross_entropy(logits.float().reshape(-1, V), target_ids.reshape(-1), reduction="mean")


# -----------------------------
# V4 MODEL (param-optimized, see model_v4.py for documented implementation)
# -----------------------------

from model_v4 import RegisterGPTv4


# -----------------------------
# TRAINING
# -----------------------------

def main():
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    grad_accum_steps = int(os.environ.get("GRAD_ACCUM_STEPS", "16"))
    grad_scale = 1.0 / grad_accum_steps

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    logfile = None
    if master:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg, console=True):
        if not master: return
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        if console: print(line)
        if logfile:
            with open(logfile, "a") as f: print(line, file=f)

    log0(code, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE mismatch")
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    bbl, hsl, ibl = build_sentencepiece_luts(sp, args.vocab_size, device)

    if args.model_version == "wave":
        from wave_model import BrainWaveGPT
        band_split = tuple(int(x) for x in args.band_split.split(","))
        base_model = BrainWaveGPT(
            vocab_size=args.vocab_size,
            num_cycles=args.num_steps,
            n_fourier_basis=args.n_fourier_basis,
            n_channels=args.n_channels,
            logit_softcap=args.logit_softcap,
            activation=args.activation,
            slow_decay_init=args.slow_decay_init,
            fast_decay_init=args.fast_decay_init,
            band_split=band_split,
        ).to(device).bfloat16()
    elif args.model_version == "v4":
        base_model = RegisterGPTv4(
            vocab_size=args.vocab_size,
            unique_steps=args.unique_steps,
            invocations_per_step=args.invocations_per_step,
            n_fourier_basis=args.n_fourier_basis,
            n_channels=args.n_channels,
            n_heads=args.n_heads,
            transform_rank=args.transform_rank,
            logit_softcap=args.logit_softcap,
            activation=args.activation,
            decay_init=args.decay_init,
        ).to(device).bfloat16()
    elif args.model_version == "gauss":
        n_freq = args.n_fourier_basis  # reuse basis count as freq count
        base_model = GaussRegisterGPT(
            vocab_size=args.vocab_size,
            num_steps=args.num_steps,
            n_freq=n_freq,
            n_channels=args.n_channels,
            logit_softcap=args.logit_softcap,
            activation=args.activation,
            decay_init=args.decay_init,
        ).to(device).bfloat16()
    else:
        base_model = AssocRegisterLM(
            vocab_size=args.vocab_size,
            num_steps=args.num_steps,
            n_fourier_basis=args.n_fourier_basis,
            n_channels=args.n_channels,
            logit_softcap=args.logit_softcap,
            activation=args.activation,
            decay_init=args.decay_init,
        ).to(device).bfloat16()

    # Keep small control params in fp32
    with torch.no_grad():
        for name, p in base_model.named_parameters():
            if (p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)) and p.dtype != torch.float32:
                p.data = p.data.float()

    use_compile = bool(int(os.environ.get("TORCH_COMPILE", "0")))
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True) if use_compile else base_model
    model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    optimizer = torch.optim.Adam(
        base_model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        fused=True,
    )
    for g in optimizer.param_groups:
        g["base_lr"] = args.lr

    n_params = sum(p.numel() for p in base_model.parameters())
    n_trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    log0(f"run_id:{args.run_id}")
    log0(f"model_version:{args.model_version}")
    log0(f"model_params:{n_params} trainable:{n_trainable} vocab=dim={args.vocab_size}")
    if args.model_version == "wave":
        log0(f"architecture:BrainWaveGPT (oscillatory dynamics, cross-frequency coupling)")
        log0(f"cycles:{args.num_steps} channels:{args.n_channels} fourier:{args.n_fourier_basis} bands:{args.band_split}")
        log0(f"slow_decay_init:{args.slow_decay_init} fast_decay_init:{args.fast_decay_init}")
    elif args.model_version == "v4":
        log0(f"unique_steps:{args.unique_steps} invocations:{args.invocations_per_step} depth:{args.unique_steps * args.invocations_per_step}")
        log0(f"channels:{args.n_channels} heads:{args.n_heads} rank:{args.transform_rank} fourier:{args.n_fourier_basis}")
    else:
        log0(f"steps:{args.num_steps} channels:{args.n_channels} fourier:{args.n_fourier_basis}")
    log0(f"activation:{args.activation} lr:{args.lr} grad_clip:{args.grad_clip_norm} decay_init:{args.decay_init}")
    log0(f"NO attention. NO embedding. NO output projection.")
    log0(f"world_size:{world_size} grad_accum:{grad_accum_steps} batch:{args.train_batch_tokens}")

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    max_wc_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step, elapsed_ms):
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wc_ms is None:
            ws = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if ws <= step < args.iterations else 1.0
        sms = elapsed_ms / max(step, 1)
        wms = args.warmdown_iters * sms
        rms = max(max_wc_ms - elapsed_ms, 0.0)
        return rms / max(wms, 1e-9) if rms <= wms else 1.0

    # Warmup
    for ws in range(args.warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum_steps):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            (loss * grad_scale).backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        optimizer.step()
        torch.cuda.synchronize()
        if master and (ws + 1) % 10 == 0 or ws + 1 == args.warmup_steps:
            log0(f"warmup:{ws + 1}/{args.warmup_steps}")

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # Training
    train_ms = 0.0
    stop_after = None
    t0 = time.perf_counter()
    step = 0
    while True:
        last = step == args.iterations or (stop_after is not None and step >= stop_after)
        if last or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            train_ms += 1000.0 * (time.perf_counter() - t0)
            vl, vbpb = eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, bbl, hsl, ibl)
            log0(f"step:{step}/{args.iterations} val_loss:{vl:.4f} val_bpb:{vbpb:.4f} train_time:{train_ms:.0f}ms")
            t0 = time.perf_counter()
        if last:
            if stop_after is not None and step < args.iterations:
                log0(f"stopping_early: step:{step}/{args.iterations}")
            break

        lm = lr_mul(step, train_ms + 1000.0 * (time.perf_counter() - t0))
        for g in optimizer.param_groups:
            g["lr"] = g["base_lr"] * lm

        step_t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        train_loss_accum = 0.0
        for _ in range(grad_accum_steps):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            (loss * grad_scale).backward()
            train_loss_accum += loss.item() * grad_scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        optimizer.step()
        torch.cuda.synchronize()

        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        approx_ms = train_ms + 1000.0 * (time.perf_counter() - t0)
        step += 1
        if master and args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log0(f"step:{step}/{args.iterations} train_loss:{train_loss_accum:.4f} time:{approx_ms:.0f}ms avg:{approx_ms / step:.1f}ms tok/s:{args.train_batch_tokens / (step_ms / 1000):.0f}")
        if max_wc_ms and stop_after is None and approx_ms >= max_wc_ms:
            stop_after = step

    # Serialize
    if master:
        sd = {k: v for k, v in base_model.state_dict().items()}
        out = f"logs/{args.run_id}_model.pt"
        torch.save(sd, out)
        log0(f"saved:{out} bytes:{Path(out).stat().st_size}")

        qobj, qstats = quantize_state_dict_int8(sd)
        buf = io.BytesIO()
        torch.save(qobj, buf)
        compressed = zlib.compress(buf.getvalue(), 9)
        qpath = f"logs/{args.run_id}_model.int8.ptz"
        Path(qpath).write_bytes(compressed)
        log0(f"quantized:{qpath} bytes:{len(compressed)} ratio:{qstats['baseline_tensor_bytes'] / max(qstats['int8_payload_bytes'], 1):.2f}x")

        dq = dequantize_state_dict_int8(torch.load(io.BytesIO(zlib.decompress(Path(qpath).read_bytes())), weights_only=False))
        base_model.load_state_dict(dq, strict=False)
        qvl, qvbpb = eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, bbl, hsl, ibl)
        log0(f"final_int8_zlib_roundtrip val_loss:{qvl:.4f} val_bpb:{qvbpb:.4f}")


if __name__ == "__main__":
    main()
