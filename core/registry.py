from __future__ import annotations

import importlib

REGISTRY = {
    "v1_attention": {
        "module": "v1_shared_attention.model",
        "class_name": "RegisterGPT",
        "kwargs": lambda a: dict(
            vocab_size=a.vocab_size, num_heads=a.num_heads,
            num_kv_heads=a.num_kv_heads, num_steps=a.num_steps,
            n_fourier_basis=a.n_fourier_basis, n_channels=a.n_channels,
            logit_softcap=a.logit_softcap, rope_base=a.rope_base,
            qk_gain_init=a.qk_gain_init, activation=a.activation,
        ),
    },
    "v2_conv": {
        "module": "v2_causal_conv.model",
        "class_name": "RegisterGPT",
        "kwargs": lambda a: dict(
            vocab_size=a.vocab_size, num_steps=a.num_steps,
            kernel_size=a.kernel_size, n_fourier_basis=a.n_fourier_basis,
            n_channels=a.n_channels, logit_softcap=a.logit_softcap,
            activation=a.activation,
        ),
    },
    "v3_assoc": {
        "module": "v3_assoc_memory.model",
        "class_name": "RegisterGPT",
        "kwargs": lambda a: dict(
            vocab_size=a.vocab_size, num_steps=a.num_steps,
            n_fourier_basis=a.n_fourier_basis, n_channels=a.n_channels,
            logit_softcap=a.logit_softcap, activation=a.activation,
            decay_init=a.decay_init,
        ),
    },
    "v4_golf": {
        "module": "v4_param_optimized.model",
        "class_name": "RegisterGPTv4",
        "kwargs": lambda a: dict(
            vocab_size=a.vocab_size, unique_steps=a.unique_steps,
            invocations_per_step=a.invocations_per_step,
            n_fourier_basis=a.n_fourier_basis, n_channels=a.n_channels,
            n_heads=a.n_heads, transform_rank=a.transform_rank,
            logit_softcap=a.logit_softcap, activation=a.activation,
            decay_init=a.decay_init,
        ),
    },
    "v5_gauss": {
        "module": "v5_gauss_fft.model",
        "class_name": "GaussRegisterGPT",
        "kwargs": lambda a: dict(
            vocab_size=a.vocab_size, num_steps=a.num_steps,
            n_freq=a.n_fourier_basis, n_channels=a.n_channels,
            logit_softcap=a.logit_softcap, activation=a.activation,
            decay_init=a.decay_init,
        ),
    },
    "v6_wave": {
        "module": "v6_brain_wave.model",
        "class_name": "BrainWaveGPT",
        "kwargs": lambda a: dict(
            vocab_size=a.vocab_size, num_cycles=a.num_steps,
            n_fourier_basis=a.n_fourier_basis, n_channels=a.n_channels,
            logit_softcap=a.logit_softcap, activation=a.activation,
            slow_decay_init=a.slow_decay_init, fast_decay_init=a.fast_decay_init,
            band_split=tuple(int(x) for x in a.band_split.split(",")),
        ),
    },
    "v7_lgp": {
        "module": "v7_lgp.model",
        "class_name": "LGPGPT",
        "kwargs": lambda a: dict(
            vocab_size=a.vocab_size, num_instructions=a.num_steps,
            n_fourier_basis=a.n_fourier_basis, n_channels=a.n_channels,
            n_ops=a.n_ops, logit_softcap=a.logit_softcap,
            decay_init=a.decay_init,
        ),
    },
    "v8_graph": {
        "module": "v8_word_graph.model",
        "class_name": "WordGraphGPT",
        "kwargs": lambda a: dict(
            vocab_size=a.vocab_size, num_hops=a.num_steps,
            interaction_rank=a.interaction_rank,
            logit_softcap=a.logit_softcap, activation=a.activation,
            decay_init=a.decay_init,
        ),
    },
    "v9_meta": {
        "module": "v9_meta_state.model",
        "class_name": "MetaStateGPT",
        "kwargs": lambda a: dict(
            vocab_size=a.vocab_size, num_steps=a.num_steps,
            state_dim=a.state_dim, inner_dim=a.inner_dim,
            logit_softcap=a.logit_softcap, activation=a.activation,
            decay_init=a.decay_init,
        ),
    },
    "v10_policy": {
        "module": "v10_policy.model",
        "class_name": "PolicyGPT",
        "kwargs": lambda a: dict(
            vocab_size=a.vocab_size, num_steps=a.num_steps,
            state_dim=a.state_dim, n_ops=a.n_ops,
            logit_softcap=a.logit_softcap, decay_init=a.decay_init,
        ),
    },
    "v11_brainwave": {
        "module": "v11_brainwave.model",
        "class_name": "BrainWaveGPT",
        "kwargs": lambda a: dict(
            vocab_size=a.vocab_size, num_steps=a.num_steps,
            state_dim=a.state_dim, inner_dim=a.inner_dim,
            gate_dim=a.state_dim, logit_softcap=a.logit_softcap,
        ),
    },
    "v11_tpg": {
        "module": "v11_tpg.model",
        "class_name": "TPGGPT",
        "kwargs": lambda a: dict(
            vocab_size=a.vocab_size, num_steps=a.num_steps,
            state_dim=a.state_dim, inner_dim=a.inner_dim,
            logit_softcap=a.logit_softcap, tau=a.gumbel_tau,
            halt_threshold=a.halt_threshold, ponder_lambda=a.ponder_lambda,
        ),
    },
    "v12_sparse": {
        "module": "v12_sparse_register.model",
        "class_name": "SparseRegisterGPT",
        "kwargs": lambda a: dict(
            vocab_size=a.vocab_size, num_steps=a.num_steps,
            k_active=a.k_active, inner_mul=a.inner_mul,
            logit_softcap=a.logit_softcap, activation=a.activation,
            decay_init=a.decay_init, parallel_waves=a.parallel_waves,
            grad_checkpoint=a.grad_checkpoint,
        ),
    },
    "v13_embed": {
        "module": "v13_sparse_embed.model",
        "class_name": "SparseEmbedGPT",
        "kwargs": lambda a: dict(
            vocab_size=a.vocab_size, num_steps=a.num_steps,
            embed_dim=a.embed_dim, k_active=a.k_active,
            inner_mul=a.inner_mul, logit_softcap=a.logit_softcap,
            activation=a.activation, decay_init=a.decay_init,
            parallel_waves=a.parallel_waves, grad_checkpoint=a.grad_checkpoint,
        ),
    },
    "v14_adaptive": {
        "module": "v14_adaptive.model",
        "class_name": "AdaptiveGPT",
        "kwargs": lambda a: dict(
            vocab_size=a.vocab_size, num_steps=a.num_steps,
            k_active=a.k_active, kernel_size=a.kernel_size,
            n_fourier_basis=a.n_fourier_basis, n_channels=a.n_channels,
            logit_softcap=a.logit_softcap, activation=a.activation,
            decay_init=a.decay_init,
        ),
    },
    "v15_predictive": {
        "module": "v15_predictive.model",
        "class_name": "PredictiveGPT",
        "kwargs": lambda a: dict(
            vocab_size=a.vocab_size, num_steps=a.num_steps,
            k_active=a.k_active, inner_mul=a.inner_mul,
            logit_softcap=a.logit_softcap, activation=a.activation,
            decay_init=a.decay_init, sparsity_k=a.sparsity_k,
            aux_loss_weight=a.aux_loss_weight,
            aux_loss_decay=a.aux_loss_decay,
        ),
    },
    "v16_columnar": {
        "module": "v16_columnar.model",
        "class_name": "ColumnarGPT",
        "kwargs": lambda a: dict(
            vocab_size=a.vocab_size, num_columns=a.num_columns,
            steps_per_column=a.steps_per_column, k_active=a.k_active,
            inner_mul=a.inner_mul, n_branches=a.n_branches,
            n_fourier_basis=a.n_fourier_basis, n_channels=a.n_channels,
            logit_softcap=a.logit_softcap, activation=a.activation,
            decay_init=a.decay_init,
        ),
    },
}


def build_model(version, args):
    """Lazy-import and instantiate a model by version string."""
    if version not in REGISTRY:
        raise ValueError(f"Unknown model version: {version!r}. Available: {list(REGISTRY.keys())}")
    entry = REGISTRY[version]
    mod = importlib.import_module(entry["module"])
    cls = getattr(mod, entry["class_name"])
    return cls(**entry["kwargs"](args))
