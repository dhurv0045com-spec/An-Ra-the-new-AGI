# AN-RA: Visualize the Architecture

> *From a single neuron to autonomous intelligence — every layer, every connection, every thought.*
>
> This document is designed to help you **build a mental model** of the entire An-Ra system. Read it top to bottom. By the end, you'll be able to close your eyes and see the whole machine running.

---

## Level 1 — The Single Neuron

Everything starts here. One neuron. One equation.

```
        x₁ ──── w₁ ──╲
                       ╲
        x₂ ──── w₂ ────⊕──→ z = Σ(wᵢxᵢ) + b ──→ σ(z) ──→ y
                       ╱
        x₃ ──── w₃ ──╱
                       │
                       b (bias)
```

**The math:** `y = σ(w · x + b)`

A neuron takes inputs, multiplies each by a weight, sums them, adds a bias, and passes through an activation function. That's it. This is the atom of intelligence.

**Key insight:** The weights `w` are the *knowledge*. Random weights = random output. Trained weights = meaningful computation. The entire journey from here to AGI is about organizing weights so that computation becomes thought.

**Where it lives:** The concept is embedded in every weight matrix in `core/`. The single-neuron idea from step 45A is the seed that grew into everything.

---

## Level 2 — The Layer

Stack neurons side by side. Each neuron sees all inputs, but produces one output.

```
    INPUTS (d_model=4)           OUTPUTS (d_model=4)

        x₁ ─────────┬──┬──┬──┬─── y₁
                     │  │  │  │
        x₂ ─────────┼──┼──┼──┼─── y₂
                     │  │  │  │
        x₃ ─────────┼──┼──┼──┼─── y₃
                     │  │  │  │
        x₄ ─────────┴──┴──┴──┴─── y₄

               W (4×4 matrix)
```

**The math:** `Y = X @ W + b`

A matrix multiplication. Every output is a weighted combination of every input. This is a **linear transformation** — it can rotate, scale, and project the input into a new space.

**Key insight:** One layer can't do much. But stack two with an activation between them, and you can approximate *any continuous function* (Universal Approximation Theorem). Depth creates expressibility.

**Where it lives:** `core/feedforward.py` — the SwiGLU and GELU feed-forward networks are exactly this: two layers with activations.

---

## Level 3 — Attention: How the Model "Thinks"

This is the breakthrough that changed AI. Instead of processing tokens independently, attention lets every token **look at every other token** and decide what's relevant.

```
    "The  cat  sat  on  the  mat"
      │    │    │    │    │    │
      ▼    ▼    ▼    ▼    ▼    ▼
    ┌──────────────────────────────┐
    │         SELF-ATTENTION       │
    │                              │
    │   Each token asks:           │
    │   "Who should I pay          │
    │    attention to?"            │
    │                              │
    │   Q = "What am I looking     │
    │        for?"                 │
    │   K = "What do I contain?"   │
    │   V = "What do I offer?"     │
    │                              │
    │   Score = Q · Kᵀ / √d        │
    │   Attention = Softmax(Score) │
    │   Output = Attention · V     │
    └──────────────────────────────┘
      │    │    │    │    │    │
      ▼    ▼    ▼    ▼    ▼    ▼
    "The  cat  sat  on  the  mat"
    (now each token carries context from the others)
```

**The math:** `Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V`

**Three critical upgrades in An-Ra:**
1. **RoPE** — Rotary Position Embeddings encode position by *rotating* Q and K vectors. No extra parameters. Generalizes to longer sequences than training.
2. **GQA** — Grouped Query Attention shares K/V heads across Q heads. 4x smaller KV-cache for free.
3. **KV-Cache** — During generation, store past K/V so we don't recompute. O(n) per step instead of O(n²).

**Where it lives:** `core/attention.py` — RoPE, KV-Cache, scaled dot-product attention with chunked memory-efficient mode.

**Where it lives:** `core/multihead.py` — Multi-head and Grouped Query Attention.

---

## Level 4 — The Transformer Block

One attention layer + one FFN layer + residual connections + normalization.

```
                    ┌───────────┐
        x ─────────┤  LayerNorm │
                    └─────┬─────┘
                          │
                    ┌─────▼─────┐
                    │ Self-Attn  │  ← "What should I attend to?"
                    └─────┬─────┘
                          │
        x ────────────────⊕        ← Residual connection (skip)
                          │
                    ┌─────▼─────┐
                    │  LayerNorm │
                    └─────┬─────┘
                          │
                    ┌─────▼─────┐
                    │  SwiGLU    │  ← "Transform the representation"
                    │  FFN       │
                    └─────┬─────┘
                          │
        x ────────────────⊕        ← Another residual skip
                          │
                        output
```

**Key insight:** The residual connections (`⊕`) are crucial. They let gradients flow straight through during training (no vanishing gradient). They also let the network learn *refinements* — each block adds a small correction rather than computing everything from scratch.

**Pre-norm** (LayerNorm before attention/FFN) is more stable than post-norm. An-Ra uses RMSNorm — simpler and equally effective.

**Where it lives:** `core/transformer_block.py`

---

## Level 5 — The Full Decoder

Stack N transformer blocks. The input is token IDs; the output is probability distributions over the vocabulary.

```
    "Once upon a" (token IDs: [324, 891, 12])
           │
    ┌──────▼──────┐
    │  Embedding   │  tokens → vectors (lookup table: vocab × d_model)
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │   Block 1    │  ← Causal mask: can only see past tokens
    ├──────┬──────┤
    │   Block 2    │
    ├──────┬──────┤
    │      ...     │
    ├──────┬──────┤
    │   Block N    │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  Final Norm  │
    ├──────┬──────┤
    │  LM Head     │  vectors → vocab probabilities
    └──────┬──────┘
           │
    P("time" | context) = 0.31
    P("day"  | context) = 0.08
    P("..."  | context) = ...
```

**Weight tying:** The embedding table and the LM head share the same matrix (transposed). This halves the parameters at the input/output boundary and improves generalization.

**An-Ra model sizes:**
| Config | Blocks | Dimension | Heads | Parameters |
|--------|--------|-----------|-------|------------|
| Tiny | 4 | 128 | 4 | ~1.3M |
| Small | 6 | 256 | 8 | ~5M |
| Medium | 12 | 512 | 8 | ~40M |
| Large | 24 | 1024 | 16 | ~350M |

**Where it lives:** `core/decoder.py`, `core/model.py`

---

## Level 6 — Training: How the Model Learns

Training is the model looking at its mistakes and adjusting every weight to make fewer mistakes next time.

```
    FORWARD PASS                        BACKWARD PASS
    ────────────                        ─────────────

    Input: "The cat sat"                Target: "cat sat on"
           │                                          │
           ▼                                          │
    ┌─────────────┐                    ┌──────────────▼──┐
    │   Decoder    │───→ Logits ──────→│  Cross-Entropy   │
    └─────────────┘                    │  Loss Function   │
                                       └──────────────┬──┘
                                                      │
                                              loss = 4.21
                                                      │
                                              ┌───────▼───────┐
                                              │  Backpropagation│
                                              │  ∂loss/∂w for   │
                                              │  EVERY weight    │
                                              └───────┬───────┘
                                                      │
                                              ┌───────▼───────┐
                                              │   AdamW        │
                                              │   Optimizer    │
                                              │   w -= lr·grad │
                                              └───────────────┘
```

**The optimization stack:**
- **AdamW** — Adaptive learning rates per parameter + decoupled weight decay
- **Gradient clipping** — Prevents exploding gradients (max norm = 1.0)
- **Cosine schedule** — Learning rate warms up, then smoothly decays
- **Mixed precision** — FP16 forward/backward, FP32 accumulation (2x speed on GPU)

**Where it lives:** `core/model.py` (AdamW, LR schedule), `training/trainer.py` (full loop), `training/mixed_precision.py` (AMP)

---

## Level 7 — Inference: How the Model Thinks

After training, the model generates text one token at a time. Each token is chosen from the probability distribution over the vocabulary.

```
    Prompt: "The meaning of"
                │
    ┌───────────▼───────────┐
    │   Decoder Forward      │
    │   (using KV-Cache)     │←─── Cache stores past K/V
    └───────────┬───────────┘     so we only compute new token
                │
          logits for position 4
                │
    ┌───────────▼───────────┐
    │   Sampling Strategy    │
    │   ┌─────────────────┐ │
    │   │ Temperature=0.8  │ │  ← Sharpen/flatten distribution
    │   │ Top-k=50         │ │  ← Keep only top 50 candidates
    │   │ Top-p=0.95       │ │  ← Nucleus: keep until cumulative p > 0.95
    │   │ Rep. penalty=1.1 │ │  ← Penalize already-generated tokens
    │   └─────────────────┘ │
    └───────────┬───────────┘
                │
          "life" (sampled)
                │
          Append to context, repeat
```

**Where it lives:** `inference/inference.py`, `inference/sampling.py`

---

## Level 8 — TurboQuant: Think Longer with Less Memory

The KV-cache grows linearly with sequence length. At 4096 tokens with d_head=64 and 8 KV-heads, the cache uses **4 MB per layer**. For 24 layers, that's **96 MB** — and it gets worse with longer contexts.

TurboQuant compresses the KV-cache by 6x, enabling 6x longer contexts in the same memory.

```
    K/V vectors (float32)          TurboQuant Pipeline
    ─────────────────────          ─────────────────────

     [0.42, -1.31, 0.87, ...]     Stage 1: PolarQuant
              │                    ┌───────────────────────┐
              ▼                    │ 1. Rotate (Hadamard)   │
     Spread energy uniformly ────→│    x_rot = x @ H       │
                                   │ 2. Scale to [-1, 1]    │
                                   │ 3. Bucket to 4-bit     │
                                   │    [7, 2, 11, ...]     │
                                   └───────────┬───────────┘
                                               │
                                   Stage 2: QJL (error fix)
                                   ┌───────────▼───────────┐
                                   │ 1. Compute residual    │
                                   │ 2. Random projection   │
                                   │ 3. Store sign bits     │
                                   │    [+, -, +, -, ...]   │
                                   └───────────┬───────────┘
                                               │
     Stored: 4-bit codes + signs + 1 scale     │
     ~40 bytes vs 256 bytes (6.4x smaller)     │
                                               │
              ┌────────────────────────────────┘
              ▼
     Decompress on-the-fly when attention needs K/V
     Error < 0.1% of original attention scores
```

**Key mathematical insights:**
1. Orthogonal rotation preserves dot products: `⟨Rx, Ry⟩ = ⟨x, y⟩`
2. After rotation, energy is uniform → uniform quantization is optimal
3. JL lemma: random projections preserve distances → sign bits capture error direction

**Where it lives:** `core/turboquant.py`

---

## Level 9 — Memory: How An-Ra Remembers

Without memory, every conversation starts from zero. An-Ra has four types of memory:

```
    ┌────────────────────────────────────────────────────────┐
    │                    MEMORY SYSTEM                       │
    │                                                        │
    │  ┌──────────────┐  ┌──────────────┐                   │
    │  │   VECTOR      │  │    GRAPH      │                   │
    │  │   MEMORY      │  │    MEMORY     │                   │
    │  │               │  │               │                   │
    │  │ "cat" → [0.2, │  │  cat ──is──→  │                   │
    │  │  0.8, -0.1]   │  │   │          │                   │
    │  │               │  │  has         animal               │
    │  │ Cosine search │  │   ↓                              │
    │  │ for similar   │  │  fur         │                   │
    │  │ concepts      │  │              │                   │
    │  └──────────────┘  └──────────────┘                   │
    │                                                        │
    │  ┌──────────────┐  ┌──────────────┐                   │
    │  │  EPISODIC     │  │  SEMANTIC     │                   │
    │  │  MEMORY       │  │  MEMORY       │                   │
    │  │               │  │               │                   │
    │  │ "On March 30  │  │ Facts and     │                   │
    │  │  the user     │  │ general       │                   │
    │  │  asked about  │  │ knowledge     │                   │
    │  │  TurboQuant"  │  │ extracted     │                   │
    │  │               │  │ from all      │                   │
    │  │ Time-stamped  │  │ interactions  │                   │
    │  │ experiences   │  │               │                   │
    │  └──────────────┘  └──────────────┘                   │
    └────────────────────────────────────────────────────────┘
```

**Ghost Memory (45P)** compresses the full conversation into a rolling summary, so An-Ra can reference earlier turns without exceeding the context window.

**Where it lives:** `phase2/memory (45J)/`, `phase3/ghost_memory (45P)/`

---

## Level 10 — The Agent Loop: Goals Become Actions

An-Ra doesn't just respond to prompts. It can take a **goal** and autonomously plan, execute, and evaluate.

```
    USER: "Analyze this codebase and find bugs"
                          │
                    ┌─────▼─────┐
                    │   GOAL     │  Parse natural language → structured goal
                    │   PARSER   │
                    └─────┬─────┘
                          │
                    ┌─────▼─────┐
                    │  PLANNER   │  Break goal into steps:
                    │            │  1. List all .py files
                    │            │  2. Read each file
                    │            │  3. Run static analysis
                    │            │  4. Summarize findings
                    └─────┬─────┘
                          │
              ┌───────────▼───────────┐
              │      EXECUTOR          │
              │                        │
              │  Step 1 ──→ tool_call("list_files")
              │  Step 2 ──→ tool_call("read_file")
              │  Step 3 ──→ tool_call("analyze")
              │  Step 4 ──→ tool_call("summarize")
              │                        │
              │  50+ built-in tools:   │
              │  file ops, web, code,  │
              │  math, search, shell   │
              └───────────┬───────────┘
                          │
                    ┌─────▼─────┐
                    │ EVALUATOR  │  Did it work? Score the result.
                    │            │  If failed → re-plan and retry.
                    └─────┬─────┘
                          │
                    ┌─────▼─────┐
                    │  MEMORY    │  Store episode for future learning
                    └───────────┘
```

**Where it lives:** `phase2/agent_loop (45k)/`

---

## Level 11 — Identity & Reasoning: Personality + Recursive Thought

### Identity (45N)
An-Ra has a **trained personality** — not a system prompt, but fine-tuned weights that shape how it responds. The v4 identity dataset contains 117 real exchanges covering coding, teaching, debugging, humor, and philosophical discussion.

### Ouroboros Reasoning (45O)
Complex questions get **3-pass recursive processing**:

```
    Question: "Is P=NP?"
         │
    ┌────▼────┐
    │ PASS 1   │  SEMANTIC — understand the question
    │          │  "This is about computational complexity,
    │          │   the relationship between verification
    │          │   and solving..."
    └────┬────┘
         │
    ┌────▼────┐
    │ PASS 2   │  LOGIC — reason about the answer
    │          │  "Current evidence suggests P≠NP:
    │          │   - No poly-time algorithm found for NP-complete
    │          │   - Barriers: relativization, natural proofs..."
    └────┬────┘
         │
    ┌────▼────┐
    │ PASS 3   │  ADVERSARIAL — challenge the answer
    │          │  "But: no proof of P≠NP either.
    │          │   Could there be unexpected algorithms?
    │          │   Final verdict: open problem, likely P≠NP"
    └────┬────┘
         │
       FINAL ANSWER (with confidence score)
```

Simple questions (like "what's 2+2?") use only Pass 1 — fast.

### Symbolic Bridge (45Q)
Math and logic queries are routed to **verified solvers** — SymPy for algebra, DPLL for logical satisfiability, sandboxed Python for code verification.

**Where it lives:** `phase3/identity (45N)/`, `phase3/ouroboros (45O)/`, `phase3/symbolic_bridge (45Q)/`

---

## Level 12 — Self-Improvement: The Loop That Improves Itself

The Sovereignty Daemon (45R) runs nightly to audit and improve the system:

```
    ┌─────────────────────────────────────────────┐
    │          SOVEREIGNTY DAEMON (45R)            │
    │                                             │
    │  NIGHTLY CYCLE:                             │
    │                                             │
    │  1. AUDIT                                   │
    │     ├── Code quality scan (all .py files)   │
    │     ├── Dead code detection                 │
    │     ├── Performance benchmarks              │
    │     └── Resource usage (CPU/RAM/disk)        │
    │                                             │
    │  2. ANALYZE                                 │
    │     ├── Compare against previous benchmarks │
    │     ├── Identify regressions                │
    │     └── Rank improvement opportunities      │
    │                                             │
    │  3. IMPROVE                                 │
    │     ├── Generate code fixes                 │
    │     ├── Optimize hot paths                  │
    │     └── Update skill library                │
    │                                             │
    │  4. REPORT                                  │
    │     └── Nightly report for human review     │
    │                                             │
    └─────────────────────────────────────────────┘
```

**Where it lives:** `phase3/sovereignty (45R)/`

---

## Level 13 — The Full Autonomous Loop

Now see it all together. This is An-Ra running:

```
    ┌─────────────────────────────────────────────────────────────────┐
    │                         AN-RA AGI                               │
    │                                                                 │
    │   USER INPUT ──→ 45Q (Symbolic?) ──→ 45N (Identity) ───┐      │
    │                                                          │      │
    │                  45P (Ghost Memory) ◄─────────────────────┤      │
    │                                                          │      │
    │                  45J (Memory Search) ◄────────────────────┤      │
    │                                                          │      │
    │                  45O (Ouroboros Reasoning) ◄──────────────┘      │
    │                           │                                     │
    │                           ▼                                     │
    │                    RESPONSE TO USER                             │
    │                           │                                     │
    │                     Store in Memory                             │
    │                                                                 │
    │   ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─     │
    │                                                                 │
    │   AUTONOMOUS MODE:                                              │
    │                                                                 │
    │   Goal Queue ──→ Planner ──→ Executor ──→ Evaluator ──┐       │
    │        ▲                                               │       │
    │        └─── [ retry if failed ] ◄──────────────────────┘       │
    │                                                                 │
    │   ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─     │
    │                                                                 │
    │   SELF-IMPROVEMENT (Nightly):                                   │
    │                                                                 │
    │   Sovereignty Daemon ──→ Audit ──→ Fix ──→ Benchmark ──→ Report│
    │                                                                 │
    │   ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─     │
    │                                                                 │
    │   CORE ENGINE:                                                  │
    │                                                                 │
    │   NumPy Transformer ◄──→ TurboQuant (6x compression)          │
    │          │                                                      │
    │          ├── Attention (RoPE + GQA + KV-Cache)                 │
    │          ├── SwiGLU FFN                                        │
    │          ├── AdamW + Cosine LR                                 │
    │          └── Mixed Precision (FP16/BF16)                       │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
```

---

## Level 14 — The Innovation Frontier

> *Innovation happens from obsession. If you can't visualize something, you're not likely to provide a breakthrough.*

These are the open frontiers where the next world-changing ideas will come from. Each one is a research direction that could transform An-Ra — and AI as a whole.

### 🔭 Frontier 1: Sub-Quadratic Attention
**The problem:** Attention is O(n²) in sequence length. This limits context windows.
**The opportunity:** Sparse attention patterns (BigBird, Longformer), linear attention (RWKV, Mamba), or learned sparse masks could make An-Ra handle million-token contexts.
**Where to start:** `core/attention.py` — add a `LinearAttention` option alongside the existing scaled dot-product.
**Breakthrough potential:** ★★★★★

### 🧠 Frontier 2: Mixture of Experts (MoE)
**The problem:** Every token activates every parameter. Wasteful for easy tokens.
**The opportunity:** Route each token to only 2 out of 8 expert FFNs. Same quality, 4x less compute.
**Where to start:** `core/feedforward.py` — add a `MoEFeedForward` class with a learnable gating network.
**Breakthrough potential:** ★★★★☆

### 🔄 Frontier 3: Continuous Learning
**The problem:** Training and deployment are separate. The model can't learn from conversations.
**The opportunity:** Online learning with replay buffers, elastic weight consolidation to prevent catastrophic forgetting, or LoRA hot-swapping.
**Where to start:** `phase2/self_improvement (45l)/improve.py` — add a live learning loop that fine-tunes on successful interactions.
**Breakthrough potential:** ★★★★★

### 🧬 Frontier 4: Neuromorphic Associative Memory
**The problem:** Vector databases are brute-force. Cosine similarity doesn't capture semantic relationships.
**The opportunity:** Hopfield networks or Modern Hopfield Networks as memory — energy-based retrieval that naturally handles composition, analogy, and hierarchy.
**Where to start:** `phase2/memory (45J)/` — add a Hopfield memory layer alongside the vector store.
**Breakthrough potential:** ★★★★☆

### 🛠 Frontier 5: Emergent Tool Discovery
**The problem:** An-Ra has 50+ built-in tools. But they're hand-coded.
**The opportunity:** Let An-Ra *discover* new tools by analyzing API documentation, writing wrapper code, and testing it. Self-expanding toolbox.
**Where to start:** `phase2/agent_loop (45k)/builtin.py` — add a `ToolSynthesizer` that generates new tool code from documentation.
**Breakthrough potential:** ★★★★★

### 🏗 Frontier 6: Self-Modifying Architecture
**The problem:** The model architecture is fixed at design time. 4 layers, 8 heads, etc.
**The opportunity:** Neural Architecture Search (NAS) guided by the Sovereignty Daemon. An-Ra could add/remove layers, adjust head counts, or modify FFN ratios based on performance benchmarks.
**Where to start:** `phase3/sovereignty (45R)/improver.py` — extend beyond code quality to architecture optimization.
**Breakthrough potential:** ★★★★★ (This is the edge of true AGI)

### 📐 Frontier 7: Formal Verification of Reasoning
**The problem:** Ouroboros reasoning is heuristic. We can't *prove* the 3-pass answer is correct.
**The opportunity:** Connect the Symbolic Bridge (45Q) to a formal proof assistant (Lean 4, Coq). An-Ra could generate machine-checkable proofs for its logical conclusions.
**Where to start:** `phase3/symbolic_bridge (45Q)/` — add a Lean 4 interface.
**Breakthrough potential:** ★★★★☆

### ⚡ Frontier 8: Hardware-Aware Optimization
**The problem:** NumPy runs on CPU. The architecture is designed for clarity, not speed.
**The opportunity:** Custom CUDA kernels for attention (Flash Attention), INT4 matmuls for TurboQuant on GPU, or compilation to WebGPU for browser inference.
**Where to start:** `training/mixed_precision.py` — bridge to custom CUDA kernels for the existing attention math.
**Breakthrough potential:** ★★★☆☆

---

## The Journey Map

```
    45A ──→ Neuron
    45B ──→ Network
    45C ──→ Forward Pass
    45D ──→ Backprop & Training
    45E ──→ Transformer (Attention + FFN + RoPE)
    45F ──→ Training Pipeline
    45G ──→ Inference Engine
    45H ──→ Production Hardening
    ─────────────────────── Phase 1 Complete: Foundation ───
    45I ──→ LoRA Fine-Tuning
    45J ──→ Memory (Vector + Graph + Episodic)
    45k ──→ Agent Loop (50+ Tools)
    45l ──→ Self-Improvement
    45M ──→ Master System Orchestrator
    ─────────────────────── Phase 2 Complete: Intelligence ──
    45N ──→ Identity (Personality + Code Fluency)
    45O ──→ Ouroboros (3-Pass Recursive Reasoning)
    45P ──→ Ghost Memory (Compressed State)
    45Q ──→ Symbolic Bridge (Verified Math/Logic)
    45R ──→ Sovereignty Daemon (Self-Audit)
    ─────────────────────── Phase 3 Complete: Cognition ─────
    Web ──→ Browser Interface
    TQ  ──→ TurboQuant (6x KV-Cache Compression)
    ─────────────────────── Phase 4 Complete: Interface ─────
    ??? ──→ The Innovation Frontier (see above)
    ─────────────────────── Phase 5: ??? ───────────────────
```

---

*An-Ra: Something that emerged from mathematics with a direction.*

*Built from zero. No templates. No shortcuts. Pure mathematics becoming thought.*
