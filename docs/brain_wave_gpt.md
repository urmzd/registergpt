# BrainWaveGPT — Oscillatory Dynamics as Computation

## The Core Thesis

Everyone in ML copies the brain's **structure**: neurons become hidden units, synapses become weights, cortical layers become transformer layers. This is like trying to fly by building mechanical feathers.

We copy the brain's **output phenomenon** instead: oscillatory waves at multiple frequencies with cross-frequency coupling. The hypothesis is that the wave dynamics *are* the computation — not a side effect of computation happening in neurons, but the actual algorithmic substrate. Mimic the symptom, not the organ.

---

## The Hardware/Software Distinction

Nobody looks at a GPU and says "the transistors are thinking." The transistors are the substrate. The software is the computation. We understand this distinction perfectly for silicon. We refuse to apply it to the brain.

Neuroscience has spent a century studying the hardware. Mapping neurons. Tracing synapses. Counting layers in the cortex. The entire field of deep learning is built on: neurons are the computation, weights are the knowledge, the structure is the algorithm.

**What if neurons are the transistors?** What if the oscillatory dynamics — the brainwaves — are the *software running on the neural hardware?*

### The Neuron Doctrine vs. The Oscillatory Alternative

The **neuron doctrine** says: information is encoded in firing rates and synaptic weights. A neuron fires or doesn't. The pattern of firing across neurons is the representation. Learning changes synaptic weights.

The **oscillatory alternative** says: information is encoded in *wave patterns*. Phase relationships between oscillations. Amplitude modulations across frequency bands. Constructive and destructive interference between signals. The neurons are just the medium these waves propagate through — like copper wire carrying a radio signal. You wouldn't study the metallurgy of the wire to understand the broadcast.

### Evidence That the Waves Are Doing Real Work

**Transcranial stimulation:** When you disrupt specific oscillatory patterns — forcing the brain into a different wave state without changing any synaptic connections — cognition changes immediately. Boost gamma in the prefrontal cortex, working memory improves. Disrupt theta in the hippocampus, memory encoding fails. The hardware didn't change. No synapses were modified. You changed the *software* and the computation changed.

**Anesthesia:** Anesthesia doesn't destroy neurons or disconnect synapses. It disrupts cross-frequency coupling. The hardware is intact. The software stops running. Consciousness disappears. When the anesthetic wears off, the same hardware resumes the same oscillatory patterns and consciousness returns. If the neurons were the computation, anesthesia wouldn't work the way it does.

**Sleep cycles are software updates.** During slow-wave sleep, the brain replays and consolidates memories by running specific oscillatory patterns across the hippocampus and cortex. The synaptic weights change as a *result* of the wave dynamics, not the other way around. The waves are the program. The weight changes are the program's output written to storage.

### The Implication for Language Models

Every architecture we've built — transformers, register machines, everything in deep learning — is an attempt to build better *hardware*. Better neuron analogs. Better synapse analogs. Better connectivity patterns. We're optimizing the transistor layout when we should be writing software.

What does a "brainwave software" language model look like? Not a stack of layers. Not a sequence of operations. A set of **coupled oscillators** running simultaneously on a shared state. The state is the register bank. The oscillators are continuous dynamics, not discrete steps. Multiple frequency bands process the same state at the same time, and the interaction between frequencies — the coupling, the interference, the resonance — is where the computation happens.

---

## What Neuroscience Shows About Brain Waves

### The Five Bands

| Band | Frequency | Function | Computational Role |
|-------|-----------|----------|-------------------|
| **Delta** | 0.5-4 Hz | Deep processing, memory consolidation | Coarse structure. The "topic" of thought. Global state. |
| **Theta** | 4-8 Hz | Learning, memory encoding, sequential processing | Temporal binding. Links this word to the previous one. Coordinates sequential intake. |
| **Alpha** | 8-13 Hz | Suppression of irrelevant processing | **Inhibition, not activation.** Gates what gets through. High alpha = region deliberately silenced. This is the sparsity mechanism. |
| **Beta** | 13-30 Hz | Maintaining current cognitive state | "Hold this thought." Keeps working memory representations stable. Resists change. |
| **Gamma** | 30-100+ Hz | Feature binding, recognition | Binding mechanism. Distributed representations become unified percepts through gamma synchronization. |

### Cross-Frequency Coupling: The Key Phenomenon

The brain doesn't run five independent oscillators. The frequencies interact through **cross-frequency coupling**:

- Gamma oscillations ride on theta waves — the **phase** of the slow theta oscillation modulates the **amplitude** of fast gamma bursts.
- The broad sequential structure (theta) *controls when* the fine-grained feature binding (gamma) happens.
- The fast wave tells the slow wave *what it found*. The slow wave tells the fast wave *when to act*.

**Reading "The cat sat on the mat":**
- Theta tracks sequential position through the sentence
- At each theta cycle, gamma bursts fire to bind the current word's features
- Delta maintains "we're in a simple declarative sentence about a cat"
- Alpha suppresses irrelevant semantic domains (finance, medicine, etc.)
- Beta holds the subject ("cat") stable in working memory for verb agreement

Nobody in ML has taken cross-frequency coupling seriously. This is the architectural innovation.

---

## Mapping Onto a Language Model

### From Layers to Frequency Bands

Instead of a stack of identical layers with forward passes, we have **superimposed waves at different frequencies** operating simultaneously and interacting:

**Low frequency components (delta/theta analog):**
- Handle broad context — topic, semantic domain, sentence structure
- Change slowly across the sequence
- Wide context windows, coarse outputs
- Few parameters (smooth, low-dimensional signals)

**Mid frequency components (alpha/beta analog):**
- Handle gating and maintenance
- Alpha-analog: **inhibit** — zero out irrelevant registers
- Beta-analog: **stabilize** — resist change in registers holding useful state
- This is the sparsity and persistence mechanism

**High frequency components (gamma analog):**
- Handle binding and specific prediction — which word comes next
- Change fast, operate locally
- **Modulated by** the slow waves — this is the coupling

### Cross-Frequency Coupling as Architecture

The amplitude of high-frequency processing is modulated by the phase of low-frequency processing. Concretely: low-frequency components produce a **gating signal** that controls when and where the high-frequency component is allowed to act.

```python
low_freq_output = process(x, frequencies[0:4])     # broad context
gate = sigmoid(low_freq_output)                      # alpha: what to suppress
high_freq_output = process(x, frequencies[4:16])     # fine detail
output = gate * high_freq_output                     # coupling: slow gates fast
```

Three lines. Explicit multi-scale processing with cross-scale interaction. No transformer layer does this.

---

## Mapping Onto the Existing Fourier Register Architecture

The current RegisterGPT v3 (`model.py`) already computes in frequency space. The gap is small.

### What We Already Have

1. **Fourier basis** (`make_fourier_basis`): Frequencies 1..K over vocabulary indices. Projections from vocab-space to channel-space go through these Fourier coefficients.

2. **FourierProjection**: Learned coefficients `(n_channels, 2*n_basis)` that weight the Fourier basis vectors. Each channel is a learned mixture of frequency components.

3. **AssociativeMemoryStep**: Cross-position mixing (the sequential/temporal mechanism). Query-key-value projections all go through Fourier projections. Causal decay weighting gives temporal structure.

4. **FourierRegisterOp**: Within-position transforms via Fourier read/write projections with channel mixing.

5. **RegisterStep**: Residual composition of memory + register op.

### What's Missing: Frequency Band Structure and Coupling

Currently, all K Fourier components act independently with independent coefficients. There is no distinction between "low frequency" and "high frequency" processing, and no coupling between bands.

The current architecture is `N` identical `RegisterStep` blocks applied sequentially:

```
for step in self.steps:
    x = step(x, self.fourier_basis)  # every step sees all frequencies equally
```

### The v4 Transformation

#### Step 1: Partition the Fourier basis into frequency bands

```python
# Current: 16 basis functions, all treated equally
# Proposed: partition into bands
bands = {
    'delta_theta': frequencies[0:4],    # broad context (low freq)
    'alpha_beta':  frequencies[4:8],    # gating + maintenance (mid freq)
    'gamma':       frequencies[8:16],   # binding + prediction (high freq)
}
```

This is a zero-cost structural change. Same parameters, organized differently.

#### Step 2: Band-specific processing

Each band gets its own processing with role-appropriate behavior:

```python
class OscillatoryStep(nn.Module):
    """One processing cycle: all frequency bands fire, then couple."""

    def __init__(self, n_basis, n_channels):
        super().__init__()
        # Band boundaries (indices into the Fourier basis)
        self.low_end = 4      # delta/theta: basis[0:4]
        self.mid_end = 8      # alpha/beta: basis[4:8]
        # gamma: basis[8:n_basis]

        # Band-specific processors
        self.low_proc = FourierRegisterOp(n_basis=4, n_channels=n_channels)
        self.mid_proc = FourierRegisterOp(n_basis=4, n_channels=n_channels)
        self.high_proc = FourierRegisterOp(n_basis=8, n_channels=n_channels)

        # Cross-frequency coupling parameters
        self.low_to_high_gate = nn.Linear(n_channels, n_channels)  # alpha gating
        self.low_to_high_mod = nn.Linear(n_channels, n_channels)   # theta modulation
        self.beta_stability = nn.Parameter(torch.ones(n_channels) * 0.9)
```

#### Step 3: Cross-frequency coupling

The critical innovation. Low frequencies gate and modulate high frequencies:

```python
def forward(self, x, basis):
    low_basis = basis[:, :2*self.low_end]
    mid_basis = basis[:, 2*self.low_end:2*self.mid_end]
    high_basis = basis[:, 2*self.mid_end:]

    # Low frequency: broad context (changes slowly)
    low_out = self.low_proc(x, low_basis)

    # Mid frequency: gating and stability
    mid_out = self.mid_proc(x, mid_basis)
    alpha_gate = torch.sigmoid(self.low_to_high_gate(low_out))  # inhibition
    beta_hold = torch.sigmoid(self.beta_stability)               # persistence

    # High frequency: fine detail, GATED by low frequency
    high_out = self.high_proc(x, high_basis)
    high_out = alpha_gate * high_out  # cross-frequency coupling

    # Compose: low provides base, mid gates, high adds detail
    update = low_out + mid_out * beta_hold + high_out
    return x + update
```

#### Step 4: Temporal coupling via the associative memory

The `AssociativeMemoryStep` already handles cross-position (temporal) mixing. In the brain wave frame:

- **Theta's role** (sequential coordination) maps onto the causal decay mechanism — it controls how far back in the sequence information flows
- **Gamma bursts** riding on theta cycles map onto the gating of memory retrieval by the low-frequency state

```python
class WaveMemoryStep(nn.Module):
    """Cross-position mixing with frequency-dependent temporal coupling."""

    def __init__(self, n_basis, n_channels):
        super().__init__()
        # Separate memory for different temporal scales
        self.slow_memory = AssociativeMemoryStep(n_basis=4, n_channels=n_channels,
                                                  decay_init=4.0)   # high decay = long range
        self.fast_memory = AssociativeMemoryStep(n_basis=8, n_channels=n_channels,
                                                  decay_init=2.0)   # low decay = local

        # Theta-gamma coupling: slow memory gates fast memory retrieval
        self.coupling_gate = nn.Linear(n_channels, n_channels)

    def forward(self, x, basis):
        low_basis = basis[:, :8]
        high_basis = basis[:, 8:]

        slow_retrieved = self.slow_memory(x, low_basis)      # broad context
        theta_gate = torch.sigmoid(self.coupling_gate(slow_retrieved))
        fast_retrieved = self.fast_memory(x, high_basis)      # local detail
        fast_retrieved = theta_gate * fast_retrieved           # theta gates gamma

        return slow_retrieved + fast_retrieved
```

---

## Why This Might Be Radically More Parameter-Efficient

### The Transformer Problem

A transformer layer treats every feature at every scale identically. The same `d_model x d_model` weight matrix processes broad semantic features and fine lexical distinctions with the same mechanism. No structural hierarchy. Every dimension is equal. The model must **learn** from scratch that some features are broad context and some are fine detail — burning millions of parameters on implicit scale discovery.

A weight matrix stores knowledge as static numbers. Every possible input-output relationship needs its own weight configuration. That's why transformers need billions of parameters — they're storing every pattern as a separate static lookup.

### The Wave Solution

An oscillatory system stores knowledge as **dynamics**. A few parameters defining a coupled oscillator can produce infinitely complex temporal behavior. A simple wave equation with three parameters (frequency, amplitude, phase) can, through superposition and interference, produce any signal. The information isn't in the number of parameters. It's in the *time evolution* of the system.

A 16MB transformer is a static lookup table with 16 million bytes of entries. A 16MB oscillatory system is a **dynamical system** whose temporal evolution produces behavior that no static lookup table of any size could match — because the behavior emerges from the dynamics, not from stored patterns.

The oscillatory architecture **builds the hierarchy in**:

- Low frequencies handle broad patterns with few parameters (smooth, low-dimensional)
- High frequencies handle fine detail with few parameters (sparse, gated by slow waves)
- The interaction between scales is parameterized by the coupling mechanism — a few scalars/vectors controlling how the slow wave modulates the fast wave

**Parameter comparison:**

| Component | Transformer (d=512) | Wave Architecture |
|-----------|--------------------:|------------------:|
| Layer mixing | 512 x 512 = 262K | 3 band processors + coupling ~ 30K |
| Scale structure | Implicit (learned) | Explicit (architectural) |
| Gating | None (or learned gates at full dim) | Frequency-band coupling ~ few hundred params |
| Total per "layer" | ~2M (with FFN) | ~50K |

The multi-scale structure that a transformer discovers through millions of parameters of implicit learning, we provide explicitly through oscillatory architecture.

### What Children Actually Do

A child doesn't store language as weights. A child's brain develops oscillatory patterns that **resonate** with linguistic structure. Theta rhythms entrain to syllable rates. Gamma bursts synchronize with phoneme boundaries. The child's brain literally *tunes its oscillators* to the temporal structure of speech.

The "parameters" being learned aren't synaptic weights — they're oscillatory frequencies and coupling patterns. And that's why it's fast. Tuning a few oscillator parameters to resonate with the input is a low-dimensional optimization problem. Learning millions of synaptic weights from data is a high-dimensional optimization problem. Same task, radically different search space.

### Not Computation — Resonance

The parameters of the system are: how many frequency bands, what are their natural frequencies, how strongly does each band couple to each other band, what are the damping and gain characteristics, and how does the input excite the system. That might be a few hundred parameters defining dynamics that, when run forward in time, produce language prediction.

Not a model that *computes* language. A model that *resonates* with it.

---

## The Deeper Insight

### Structure vs. Dynamics

Maybe the brain's structure — neurons, synapses, cortical layers — is just the substrate. The **algorithm** is the wave dynamics. Evolution converged on neural tissue as a convenient medium to produce those dynamics, but the dynamics are what matter.

A silicon implementation that produces the same oscillatory patterns with the same cross-frequency coupling would compute the same way — not because it looks like a brain, but because it **oscillates** like a brain.

### What We're Mimicking

Not neurons. Not synapses. Not cortical columns.

We're mimicking:
1. **Simultaneous multi-frequency oscillation** — multiple scales of processing in parallel
2. **Cross-frequency coupling** — slow modulates fast (theta-gamma coupling)
3. **Phase-based gating** — alpha suppression (sparsity through inhibition)
4. **Temporal binding through synchronization** — gamma binding (feature composition)
5. **Sequential coordination through slow waves** — theta rhythm (reading order)

These are all **dynamics**, not structures. Patterns of how information flows over time, not descriptions of hardware. They may be the actual computational principles, not side effects.

### The Framework Is Already Here

The Fourier register architecture is **already almost there**:
- We already compute in frequency space
- We already have multiple frequency components in the basis
- We already have cross-position temporal dynamics (associative memory with decay)
- We already have within-position transforms (register ops)

The missing piece: **coupling between frequency bands** and **phase-dependent gating**. That's the delta from v3 to v4.

---

## Mathematical Framework

### Notation

| Symbol | Meaning |
|--------|---------|
| $V$ | Vocabulary size (= register dimension) |
| $T$ | Sequence length |
| $B$ | Batch size |
| $K$ | Total Fourier basis pairs |
| $C$ | Channel dimension |
| $K_L, K_M, K_H$ | Basis pairs per band (low, mid, high); $K_L + K_M + K_H = K$ |

### Fourier Basis

$$\phi_k(v) = \left[\cos\!\left(\frac{2\pi k v}{V}\right),\; \sin\!\left(\frac{2\pi k v}{V}\right)\right], \quad k = 1, \dots, K$$

Full basis matrix $\Phi \in \mathbb{R}^{V \times 2K}$.

**Key property:** Low-$k$ basis functions vary smoothly across vocabulary indices (broad semantic groupings). High-$k$ basis functions oscillate rapidly (fine lexical distinctions). This is the structural prior — frequency in vocabulary space maps to representational scale.

### Frequency Band Partition

$$\Phi_L = \Phi_{:,\;1:2K_L}, \quad \Phi_M = \Phi_{:,\;2K_L+1:2(K_L+K_M)}, \quad \Phi_H = \Phi_{:,\;2(K_L+K_M)+1:2K}$$

Default partition with $K = 16$: $K_L = 4$, $K_M = 4$, $K_H = 8$.

Low band sees only smooth patterns. High band sees only sharp patterns. Neither can access the other's frequency range. Cross-frequency coupling is the *only* pathway between scales.

### Band Projection (Analysis)

For band $b \in \{L, M, H\}$ with sub-basis $\Phi_b \in \mathbb{R}^{V \times 2K_b}$ and learned coefficients $\alpha_b \in \mathbb{R}^{C \times 2K_b}$:

$$R_b = \text{softmax}\!\left(\Phi_b \, \alpha_b^\top,\; \text{dim}=0\right) \in \mathbb{R}^{V \times C}$$
$$z_b = x \, R_b \in \mathbb{R}^{B \times T \times C}$$

This decomposes the register state into a band-specific channel representation. The softmax ensures each channel reads a normalized mixture over vocabulary items, weighted by that band's frequency components.

### Band Transform (Within-Position)

$$h_b = \sigma\!\left(z_b \, A_b + d_b\right)$$

where $A_b \in \mathbb{R}^{C \times C}$ is a channel mixing matrix, $d_b \in \mathbb{R}^C$ is a bias, and $\sigma$ is an activation function (GELU). This is the "computation" within each frequency band — how it processes what it reads from the registers.

### Cross-Frequency Coupling (Alpha Gate)

The low band's channel representation gates the high band's output:

$$g_\alpha = \sigma_{\text{sigmoid}}\!\left(h_L \, W_\alpha + b_\alpha\right) \in \mathbb{R}^{B \times T \times C}$$
$$h_H \leftarrow g_\alpha \odot h_H$$

where $W_\alpha \in \mathbb{R}^{C \times C}$, $b_\alpha \in \mathbb{R}^C$.

This is alpha suppression: the broad context (low band) decides which fine-detail channels (high band) are relevant. Irrelevant high-frequency content is zeroed out before it can influence the register state.

### Synthesis (Write-Back)

For each band with write coefficients $\beta_b \in \mathbb{R}^{C \times 2K_b}$:

$$W_b = \Phi_b \, \beta_b^\top \in \mathbb{R}^{V \times C}$$
$$\Delta x_{\text{band}} = \gamma_L \cdot h_L \, W_L^\top + \gamma_M \cdot h_M \, W_M^\top + \gamma_H \cdot h_H \, W_H^\top$$

where $\gamma_L, \gamma_M, \gamma_H$ are learned scale parameters. The register state is updated:

$$x \leftarrow x + \Delta x_{\text{band}}$$

### Cross-Position Temporal Mixing

Two associative memories operating at different temporal scales, coupled:

**Slow memory (theta analog):** Long-range context.

$$Q_L, K_L, V_L = x \, R_L^Q, \; x \, R_L^K, \; x \, R_L^V$$

where $R_L^Q, R_L^K, R_L^V$ are softmax-normalized projections through $\Phi_L$.

$$S_L[t, s] = (Q_L[t] \cdot K_L[s]) \cdot \lambda_L^{t - s - 1} \cdot \mathbb{1}[s < t]$$
$$r_L = S_L \, V_L \in \mathbb{R}^{B \times T \times C}$$

with $\lambda_L = \sigma_{\text{sigmoid}}(\ell_L)$, $\ell_L$ initialized high $\Rightarrow$ slow decay $\Rightarrow$ long-range memory.

**Fast memory (gamma analog):** Local detail.

Same structure through $\Phi_H$, but with $\lambda_H$ initialized low $\Rightarrow$ fast decay $\Rightarrow$ short-range.

**Theta-gamma temporal coupling:**

$$g_\theta = \sigma_{\text{sigmoid}}\!\left(r_L \, W_\theta + b_\theta\right)$$
$$r_H \leftarrow g_\theta \odot r_H$$

The slow memory's retrieval gates what the fast memory is allowed to contribute. Broad sequential context controls when fine-grained binding acts.

**Temporal update:**

$$\Delta x_{\text{temporal}} = s_L \cdot r_L \, O_L^\top + s_H \cdot r_H \, O_H^\top$$

where $O_L, O_H$ are output projections back to vocabulary space.

### Full Oscillatory Cycle

One cycle combines temporal mixing and band processing:

$$\tilde{x} = \text{RMSNorm}(x)$$
$$\Delta x_{\text{temporal}} = \text{TemporalMixing}(\tilde{x}, \Phi_L, \Phi_H)$$
$$x \leftarrow x + \Delta x_{\text{temporal}}$$
$$\tilde{x} = \text{RMSNorm}(x)$$
$$\Delta x_{\text{band}} = \text{BandProcessing}(\tilde{x}, \Phi_L, \Phi_M, \Phi_H) \quad \text{with coupling}$$
$$x \leftarrow x + \Delta x_{\text{band}}$$

### Output

After $N$ cycles:

$$\text{logits} = \text{RMSNorm}(x) \cdot s_{\text{logit}}$$
$$\text{logits} = C_{\text{cap}} \cdot \tanh\!\left(\text{logits} / C_{\text{cap}}\right)$$
$$\mathcal{L} = \text{CrossEntropy}(\text{logits}, \text{targets})$$

### Parameter Count Per Cycle

| Component | Parameters |
|-----------|-----------|
| Slow memory ($\Phi_L$, 4 projections) | $4 \times C \times 2K_L + 2$ |
| Fast memory ($\Phi_H$, 4 projections) | $4 \times C \times 2K_H + 2$ |
| Low band register op | $2 \times C \times 2K_L + C^2 + C + 1$ |
| Mid band register op | $2 \times C \times 2K_M + C^2 + C + 1$ |
| High band register op | $2 \times C \times 2K_H + C^2 + C + 1$ |
| Alpha coupling ($W_\alpha, b_\alpha$) | $C^2 + C$ |
| Theta-gamma coupling ($W_\theta, b_\theta$) | $C^2 + C$ |
| Scales | $5$ |

With $C = 128$, $K_L = 4$, $K_M = 4$, $K_H = 8$: **~103K per cycle**. With 8 cycles: **~824K total**.

For comparison, v3 with 8 steps: ~330K. The wave architecture uses ~2.5x more parameters for richer dynamics. To match parameter count, use 4 cycles or reduce $C$.

---

## Concrete v4 Implementation Plan

### Phase 1: Minimal Coupling (Test the Hypothesis)

Add a single coupling mechanism to the existing architecture:

```python
# In RegisterStep.forward(), after computing memory and register outputs:
# Split the Fourier basis into low/high bands
# Use low-band output to gate high-band output
# ~20 new parameters per step
```

This tests whether cross-frequency coupling improves loss at all, with minimal code change.

### Phase 2: Band-Specific Processing

Replace the single `FourierRegisterOp` per step with band-specific processors:
- Low-band processor (broad context, few channels)
- Mid-band processor (gating, inhibition)
- High-band processor (fine detail, gated by low)

### Phase 3: Temporal Coupling

Split the `AssociativeMemoryStep` into slow and fast memories with different decay rates, where slow memory gates fast memory retrieval (theta-gamma coupling).

### Phase 4: Full Oscillatory Dynamics — From Steps to Continuous Time

Replace the sequential step loop with simultaneous oscillatory processing. This is the conceptual leap: stop thinking about layers, stop thinking about forward passes, start thinking about coupled oscillators processing a shared register bank over continuous time.

The input is a sequence of tokens. Each token sets the initial state of the register bank. Then you don't process it step by step. You **let it oscillate**. Multiple frequency bands activate simultaneously. Low frequencies establish broad context. High frequencies resolve specific details. Cross-frequency coupling means the broad context shapes the fine detail. The system runs for some number of oscillatory cycles — not layers, not steps, but *time* in the oscillatory dynamics — and the final state of the register bank is the prediction.

The parameters aren't weights in a matrix. They're properties of the oscillators — natural frequencies, coupling strengths between bands, damping coefficients, phase offsets. A small number of parameters defining the *dynamics* of a system that produces complex behavior through its temporal evolution.

Concretely:
- All bands process in parallel within each cycle
- Coupling happens within each cycle (slow gates fast, fast informs slow)
- The number of cycles replaces the number of layers
- Band interactions create emergent multi-scale behavior
- A neural ODE formulation could make the "time" truly continuous

### Expected Signatures of Success

If the brain wave hypothesis is correct, we should see:

1. **Better loss per parameter** — the structural prior should reduce the parameters needed
2. **Interpretable band specialization** — low bands should learn broad context, high bands should learn local prediction
3. **Alpha-band sparsity** — the gating mechanism should learn to suppress most registers (sparse activation)
4. **Coupling strength correlating with task difficulty** — harder predictions should show stronger theta-gamma coupling
5. **Graceful degradation** — removing high-frequency bands should hurt precision but preserve coherence; removing low-frequency bands should destroy coherence even if local statistics survive

---

## Relationship to Prior Work

### What This Is Not

- **Not multi-scale attention** (like Longformer/BigBird) — those use structural sparsity patterns, not frequency-based coupling
- **Not mixture of experts** — MoE routes tokens to different experts; this routes *frequency bands* through coupled processors
- **Not simply using Fourier features** (like FNet) — FNet replaces attention with FFT but has no cross-frequency coupling
- **Not a biological neural network simulator** — we don't model neurons, we model the oscillatory dynamics they produce

### What This Extends

- **Fourier register architecture (v1-v3)**: We already compute in frequency space. v4 adds the coupling between frequencies that makes it oscillatory rather than merely spectral.
- **Linear attention with decay**: The `AssociativeMemoryStep` is already a causal temporal mechanism. Splitting it into slow/fast memories with coupling makes it theta-gamma like.
- **State space models (Mamba, etc.)**: SSMs also model sequences through continuous dynamics, but use a single state that evolves — not multi-frequency oscillation with coupling.

---

## Open Questions

1. **Band boundaries**: Should the frequency partitioning be fixed or learned? The brain's bands are relatively fixed across individuals, suggesting a strong prior — but the right boundaries for language may differ from auditory/visual processing.

2. **Coupling direction**: We described slow-gates-fast (top-down). Should there also be fast-informs-slow (bottom-up)? In the brain, gamma phase resets can influence theta — this would be the high-frequency output feeding back into the low-frequency state.

3. **Number of bands**: Three (low/mid/high) is the minimum for the alpha-gating story. Five (delta/theta/alpha/beta/gamma) maps directly onto neuroscience. Is there a sweet spot for language?

4. **Oscillatory vs. residual dynamics**: Currently we use residual connections (`x = x + update`). True oscillatory dynamics would have the state *oscillate* — alternating between phases. Should the register state literally oscillate, or is the residual stream + frequency-band processing sufficient?

5. **Phase**: We use the Fourier basis for spatial (vocabulary) frequencies, not temporal frequencies. Should there be explicit temporal oscillation — a clock signal that modulates processing differently at different sequence positions?
