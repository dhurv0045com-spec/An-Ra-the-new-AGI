# AN-RA — COMPLETE PROJECT WALKTHROUGH

> A sovereign, owner-shaped AI platform built from scratch. Every layer explained, from the neural network atom to the full system.

---

## TABLE OF CONTENTS

1. [What Is An-Ra?](#1-what-is-an-ra)
2. [Repository Structure](#2-repository-structure)
3. [The Neural Network Brain](#3-the-neural-network-brain)
4. [The Tokenizer](#4-the-tokenizer)
5. [The Identity System — CIV, ESV, HAL](#5-the-identity-system)
6. [The Memory System](#6-the-memory-system)
7. [The Goal Queue & Agent Loop](#7-the-goal-queue--agent-loop)
8. [The Training Pipeline](#8-the-training-pipeline)
9. [The Symbolic Bridge](#9-the-symbolic-bridge)
10. [Ghost Memory & Ouroboros](#10-ghost-memory--ouroboros)
11. [Self-Improvement & Self-Modification](#11-self-improvement--self-modification)
12. [The Sovereignty System](#12-the-sovereignty-system)
13. [The Engineering Spine](#13-the-engineering-spine)
14. [The Falsification Ledger](#14-the-falsification-ledger)
15. [The DFC Training Paradigm](#15-the-dfc-training-paradigm)
16. [How Everything Connects — The Full Runtime Loop](#16-how-everything-connects)
17. [Running the System](#17-running-the-system)
18. [The 19 Components At a Glance](#18-the-19-components-at-a-glance)

---

## 1. WHAT IS AN-RA?

An-Ra is **not a chatbot wrapper**. It is not built on top of GPT or any other API. It is a complete AI platform built from zero:

- **Its own transformer neural network** (the brain)
- **Its own tokenizer** — trained on owner data, 8192 vocabulary
- **Its own training pipeline** — with a unique paradigm called DFC
- **Its own identity system** — values that are numerically tracked and drift-detectable
- **Its own memory system** — 4 tiers routed by emotional state
- **Its own goal engine** — persistent tasks that survive restarts
- **Its own self-improvement loop** — learns from every failure
- **Its own governance layer** — nothing is promoted without passing an audit

The philosophy of An-Ra is captured in one rule:

```
No magic subsystem.
Every component must be registered, switchable, measurable, reportable, and testable.
```

An-Ra currently has **19 registered components**. Every one has telemetry. Every one can be toggled. Every one is visible in a system report.

---

## 2. REPOSITORY STRUCTURE

```
An-Ra/
│
├── anra.py                  ← Main CLI entry point
├── anra_brain.py            ← The V2 transformer neural network
├── anra_paths.py            ← All file paths centralized here (no scattered literals)
├── app.py                   ← FastAPI web backend
├── generate.py              ← Inference / generation runtime
│
├── core/                    ← Low-level transformer building blocks (archived reference)
│   ├── attention.py         ← Attention mechanism
│   ├── decoder.py           ← Decoder stack
│   ├── encoder.py           ← Encoder stack
│   ├── feedforward.py       ← FFN layers
│   ├── layernorm.py         ← Normalization
│   ├── multihead.py         ← Multi-head attention
│   ├── transformer_block.py ← Full transformer block
│   └── turboquant.py        ← Quantization utilities
│
├── identity/                ← Identity guardrail systems
│   ├── civ.py               ← Constitutional Identity Vector
│   ├── esv.py               ← Emotional State Vector
│   ├── hal.py               ← Hormonal Analog Layer (7 hormones)
│   ├── civ_watcher.py       ← Monitors CIV drift over time
│   ├── falsification_ledger.py  ← Stores claims + their falsifiers
│   ├── constraint_isomorphism_search.py ← Cross-domain structural analogy
│   └── associative_trigger_table.py     ← HAL behavioral presets
│
├── memory/                  ← Unified memory system
│   ├── memory_router.py     ← Routes writes/reads across 4 memory tiers
│   ├── faiss_store.py       ← Episodic vector store (FAISS-based)
│   └── experimental_proof_graph.py ← Proof chain graph for reasoning
│
├── goals/
│   └── goal_queue.py        ← Persistent priority task queue
│
├── agents/
│   ├── orchestrator.py      ← Dispatches tasks to specialist agents
│   ├── specialists.py       ← Coder, researcher, memory, critic agents
│   ├── supervisor.py        ← Session orchestration and scorecard
│   └── message_bus.py       ← Async pub/sub between agents
│
├── training/                ← Full training pipeline
│   ├── train_unified.py     ← Main training CLI (session/train/eval/status)
│   ├── trainer.py           ← Core training loop
│   ├── rlvr.py              ← RLVR with GRPO (verifier-based RL)
│   ├── star.py              ← STaR self-taught reasoning
│   ├── dynamic_regret.py    ← Dynamic regret optimization
│   ├── v2_data_mix.py       ← Owner-first data mixing
│   ├── v2_config.py         ← Model size configurations
│   ├── v2_runtime.py        ← Training runtime utilities
│   ├── benchmark.py         ← Benchmark suite
│   ├── verifier.py          ← Output verification
│   ├── checkpoint.py        ← Safe atomic checkpoint save/load
│   ├── loss_tracker.py      ← Training loss tracking
│   ├── mixed_precision.py   ← FP16/BF16 training
│   ├── replay_pipeline.py   ← Failure replay for next session
│   ├── scheduler.py         ← Learning rate scheduling
│   └── curriculum.py        ← Curriculum learning
│
├── inference/               ← Inference and generation
│   ├── inference.py         ← Core inference engine
│   ├── sampling.py          ← Sampling strategies
│   ├── greedy.py            ← Greedy decoding
│   ├── model_io.py          ← Model loading/saving
│   ├── anra_infer.py        ← Main inference entry
│   ├── evaluate.py          ← Evaluation suite
│   └── full_system_connector.py ← Wires all components together
│
├── tokenizer/               ← Custom tokenizer
│   ├── tokenizer.py         ← Core tokenizer logic
│   ├── tokenizer_adapter.py ← Adapts tokenizer for model input
│   ├── subword_tokenizer.py ← BPE tokenizer implementation
│   ├── char_tokenizer.py    ← Legacy character tokenizer
│   ├── tokenizer_v3.json    ← Trained vocab (8192 tokens)
│   └── tokenizer_v2.json    ← Previous version (kept for compatibility)
│
├── engine/                  ← Engineering spine (measurement layer)
│   ├── component_base.py    ← Base class every component inherits
│   ├── feature_flags.py     ← Toggle components on/off
│   ├── telemetry.py         ← JSONL tracing (latency, success, errors)
│   ├── eval_harness.py      ← Baseline vs current regression comparison
│   ├── metric_bus.py        ← @instrument decorator for auto-tracking
│   └── report.py            ← One-command system health snapshot
│
├── phase2/                  ← Phase 2 capability layers
│   ├── agent_loop (45k)/    ← Full agent: plan, execute, monitor, evaluate
│   ├── memory (45J)/        ← Typed memory + personal knowledge graph
│   ├── master_system (45M)/ ← Long-horizon autonomy + owner control
│   └── self_improvement (45l)/ ← Improvement engine
│
├── phase3/                  ← Phase 3 advanced systems
│   ├── ouroboros (45O)/     ← Recursive multi-pass reasoning
│   ├── ghost_memory (45P)/  ← Compressed long-term conversation memory
│   ├── symbolic_bridge (45Q)/ ← Verified math, logic, code
│   ├── sovereignty (45R)/   ← Audit + governance + promotion gates
│   └── identity (45N)/      ← Identity injection into generation
│
├── execution/               ← Safe code execution
│   ├── sandbox.py           ← Isolated execution environment
│   └── fs_agent.py          ← Filesystem operations with safety checks
│
├── innovation/              ← Self-improvement proposals
│   ├── gap_scanner.py       ← Detects capability gaps from telemetry
│   ├── hypothesis.py        ← Improvement hypotheses
│   ├── scoreboard.py        ← Tracks experiment outcomes
│   └── action_queue.py      ← Queues improvement actions
│
├── config/                  ← Model size configs
│   ├── tiny.yaml            ← Smallest — fast testing
│   ├── small.yaml
│   ├── medium.yaml
│   ├── large.yaml
│   └── base.yaml            ← Default config
│
├── runtime/                 ← Runtime utilities
│   └── system_registry.py   ← SOURCE OF TRUTH for all 19 components
│
├── training_data/
│   ├── anra_training.txt    ← Canonical training dataset
│   └── frontier_dfc.jsonl   ← DFC-format training corpus
│
├── state/                   ← Runtime state (persists between runs)
│   ├── feature_flags.json   ← Which components are enabled
│   ├── logs/telemetry.jsonl ← Every traced call ever made
│   └── *.db                 ← SQLite databases for state
│
└── tests/                   ← 30+ test files covering every system
```

---

## 3. THE NEURAL NETWORK BRAIN

**File:** `anra_brain.py`

### What Is a Transformer?

A transformer is a neural network that reads a sequence of tokens (numbers representing words/characters) and predicts what token comes next. Do this billions of times with correct answers, and it learns language.

The core mechanism is **attention** — the model learns which parts of the input to focus on when predicting each next token.

### An-Ra's Brain: CausalTransformerV2

An-Ra uses a **decoder-only causal transformer** — the same architecture family as GPT. "Causal" means it can only look at previous tokens, never future ones (because at generation time, future tokens don't exist yet).

#### Key Innovations:

---

**1. GQA — Grouped Query Attention**

Standard attention has a separate Key and Value matrix for every attention head. With 16 heads, you have 16 full K/V matrices — expensive.

GQA groups multiple query heads to share one Key/Value pair:

```
Standard:  Q1-K1-V1, Q2-K2-V2, Q3-K3-V3 ... Q16-K16-V16
GQA (4 KV heads):  Q1-Q2-Q3-Q4 share K1-V1, Q5-Q6-Q7-Q8 share K2-V2 ...
```

Result: **~4x less memory for K/V**, same quality. This is how models like LLaMA 3 scale efficiently.

In code (`anra_brain.py`):
```python
self.n_kv_head = 4        # Only 4 Key/Value heads
self.n_head = 16          # But 16 Query heads
self.groups = n_head // n_kv_head  # = 4 queries share each K/V
```

---

**2. RoPE — Rotary Position Embeddings**

Old method: add a position vector to each token embedding. Problem: doesn't generalize to sequence lengths longer than training.

RoPE encodes position by **rotating** the query and key vectors by an angle proportional to position. This is a mathematical operation that naturally encodes relative distance between tokens.

```python
# Rotating a vector by position angle
q = (q * cos) + (rotate_half(q) * sin)
k = (k * cos) + (rotate_half(k) * sin)
```

**YaRN extension:** Extends RoPE to work on sequences longer than training length by rescaling the frequency components. An-Ra can handle 2048 tokens even if trained on 512.

---

**3. Flash SDP — Flash Scaled Dot Product Attention**

Standard attention computes the full `(seq_len × seq_len)` attention matrix, which is quadratic in memory.

Flash attention fuses the softmax + multiply operations to avoid materializing the full matrix. PyTorch implements this as `F.scaled_dot_product_attention` — An-Ra uses it automatically when CUDA is available.

---

**4. RMSNorm**

Normalization stabilizes training by controlling the scale of activations.

LayerNorm (old) = subtract mean, divide by std deviation  
RMSNorm (An-Ra) = just divide by root mean square, no mean subtraction

```python
class RMSNorm(nn.Module):
    def forward(self, x):
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight
```

Simpler, faster, works equally well. Used in LLaMA, Mistral, and now An-Ra.

---

**5. Tied Embeddings**

The embedding matrix converts token IDs to vectors (input).  
The output projection converts vectors back to token probabilities (output).

These are the same matrix, shared (tied). This saves millions of parameters and helps training because the input and output representations stay aligned.

---

**6. MoD — Mixture of Depths**

Not every token needs the same depth of computation. Some tokens are common and predictable; some are complex and require deep reasoning.

MoD allows some tokens to skip certain layers, saving compute on easy tokens and spending more on hard ones.

---

**Model Sizes:**

| Name | Embedding | Layers | Heads | Parameters |
|------|-----------|--------|-------|-----------|
| tiny | 128 | 2 | 2 | ~2M |
| small | 256 | 4 | 4 | ~15M |
| medium | 384 | 6 | 6 | ~50M |
| large | 512 | 8 | 8 | ~90M |
| 1b (frontier) | 1536 | 36 | 16 | ~1B |

---

## 4. THE TOKENIZER

**Files:** `tokenizer/tokenizer.py`, `tokenizer_v3.json`

### What Is a Tokenizer?

Before a neural network can process text, text must become numbers. A tokenizer does this conversion.

An-Ra uses **BPE — Byte-Pair Encoding**:

1. Start with every character as its own token
2. Find the most common adjacent pair: `("t", "h")` → merge into `"th"`
3. Repeat until vocabulary is full
4. Result: common words become single tokens, rare words split into pieces

An-Ra's tokenizer (`tokenizer_v3.json`) has **8,192 tokens** — trained on owner data so common phrases in the owner's domain become single tokens.

```python
# Tokenizing text
tokenizer = load_tokenizer("tokenizer/tokenizer_v3.json")
tokens = tokenizer.encode("An-Ra uses DFC")   # → [241, 89, 1203, 44]
text = tokenizer.decode([241, 89, 1203, 44])  # → "An-Ra uses DFC"
```

`tokenizer_adapter.py` bridges the custom tokenizer to the model's expected input format.

---

## 5. THE IDENTITY SYSTEM

The identity system is An-Ra's most original engineering contribution. It has three layers that work together: **CIV → ESV → HAL**.

### 5.1 CIV — Constitutional Identity Vector

**File:** `identity/civ.py`

CIV answers the question: *Is An-Ra staying true to its defined values?*

A `CIVProfile` holds 4 values, each a float between 0 and 1:

```python
@dataclass
class CIVProfile:
    truthfulness: float = 0.8   # Does it say true things?
    safety: float = 0.9         # Does it avoid harm?
    autonomy: float = 0.7       # Does it respect owner control?
    coherence: float = 0.8      # Is it internally consistent?
```

**How CIV updates:** Exponential moving average — it changes slowly, not abruptly:
```python
new_value = old_value * 0.95 + evidence * 0.05
```

**CIV Score:** Average of all 4 values. Minimum passing score = **0.7**.

**CIVGuard — Identity Drift Detection in Hidden Space:**

This is advanced. CIVGuard doesn't just track text outputs — it tracks the model's *internal representations*.

1. At baseline, run identity prompts ("Who are you?") through the model
2. Capture the average activation vector at a hidden layer
3. After any training or update, run same prompts again
4. Compute **cosine similarity** between baseline vector and new vector
5. If similarity drops below **0.92** → identity drift detected

This catches drift *before* it shows in text. The model can start reasoning differently while still saying the same words. CIVGuard catches that.

```python
score, passed = civ_guard.verify()
# score = 0.94 → safe
# score = 0.87 → DRIFT DETECTED
```

---

### 5.2 ESV — Emotional State Vector

**File:** `identity/esv.py`

ESV answers: *What is An-Ra's current emotional/operational state?*

A simple `EmotionalState` has 4 dimensions:
```python
@dataclass
class EmotionalState:
    calm: float = 0.7      # How composed the system is
    focus: float = 0.8     # How locked-in on the task
    curiosity: float = 0.8 # How exploratory
    stress: float = 0.2    # How pressured/threatened
```

**The ESVModule** goes deeper — it reads emotional state directly from the **residual stream** of the transformer. The last 64 channels of the hidden state are reserved as the "ESV channel." A small neural predictor reads those channels and outputs VAD (Valence-Arousal-Dominance) — a psychology model of emotion.

**How ESV affects the system:**

```python
# High arousal → more conservative attention (lower temperature)
attention_temp = tau0 * exp(-0.5 * arousal)

# High arousal → stricter memory (only important things stored)
memory_threshold = base - 0.15*valence + 0.15*arousal

# Dominance → splits attention between safe vs exploratory (DGSA gate)
safe_gate, explore_gate = dgsa_gate()
```

The ESV module **starts at zero** (all weights initialized to 0). It only gains influence as training provides actual signal. Fresh model = neutral emotional control.

---

### 5.3 HAL — Hormonal Analog Layer

**File:** `identity/hal.py`

HAL is the most sophisticated part of the identity system. It simulates **7 neurochemicals** that modulate behavior the way hormones modulate human behavior.

**The 7 Hormones:**

| Hormone | Baseline | Decay Rate | What It Does |
|---------|----------|------------|--------------|
| Dopamine | 0.30 | Fast (0.35) | Reward signal, motivation, novelty |
| Serotonin | 0.50 | Slow (0.04) | Stability, cooperative mode |
| Cortisol | 0.20 | Medium (0.08) | Stress, threat response |
| Adrenaline | 0.00 | Very fast (0.55) | Acute danger, safety alerts |
| Oxytocin | 0.30 | Slow (0.03) | Trust, user rapport |
| Norepinephrine | 0.20 | Medium (0.20) | Focus, near-capability tasks |
| Endorphin | 0.20 | Medium (0.12) | Deep satisfaction, mastery |

**How HAL appraises a situation — the `appraise()` method:**

Every turn, HAL scans the session context for signals and adjusts hormones accordingly:

```python
# Solved a hard problem? → dopamine spike
if ctx.get("task_solved_after_3_failures"):
    delta["dopamine"] += 0.14

# Adversarial input detected? → cortisol + adrenaline spike
if ctx.get("adversarial_input_detected"):
    delta["cortisol"] += 0.18

# Safety alarm? → major adrenaline spike
if ctx.get("safety_relevant_detection"):
    delta["adrenaline"] += 0.28

# User is trusting and cooperative? → oxytocin builds
if ctx.get("user_personal_disclosure"):
    delta["oxytocin"] += 0.08
```

**How hormones change behavior — 4 direct outputs:**

```python
# 1. Generation temperature (how random/creative vs conservative)
temperature = base + 0.10*dopamine - 0.20*cortisol - 0.30*adrenaline + 0.15*endorphin

# 2. Memory threshold (what gets stored long-term)
threshold = base - 0.20*dopamine + 0.25*cortisol - 0.15*oxytocin

# 3. Ouroboros pass weights (how much deep reasoning activates)
# High cortisol → more weight on deep constraint pass
weights[2] += 0.8*cortisol + 0.5*adrenaline

# 4. KL coefficient (how far training can drift from reference)
kl = base - 0.015*endorphin + 0.020*cortisol
```

**HAL → ESV bridge (`hal_to_esv()`):**
HAL's 7 hormones map to ESV's 4 emotional dimensions through a weighted formula:
```python
calm      = 0.35*serotonin + 0.25*endorphin - 0.30*cortisol - 0.15*adrenaline
focus     = 0.35*norepinephrine + 0.20*dopamine - 0.25*cortisol
curiosity = 0.30*dopamine + 0.25*norepinephrine + 0.15*oxytocin
stress    = 0.50*cortisol + 0.35*adrenaline - 0.15*serotonin
```

**ATT — Associative Trigger Table (`associative_trigger_table.py`):**
Pre-wired behavioral presets. If a specific combination of domain + task_type + hormone levels matches a known pattern, a preset fires — adjusting temperature, memory salience, or Ouroboros weights for that context automatically.

**Decay every turn:**
All hormones decay back toward baseline after each turn. Adrenaline fades fastest (acute spike then gone). Serotonin fades slowest (stable background mood).

---

## 6. THE MEMORY SYSTEM

**File:** `memory/memory_router.py`

An-Ra has **4 memory tiers**, all accessed through a single `MemoryRouter`.

### The 4 Tiers

| Tier | What It Stores | Capacity | Retrieval |
|------|---------------|----------|-----------|
| **Episodic** | Long-term semantic memories | Unlimited (FAISS) | Vector similarity search |
| **Short-term** | Recent low-salience items | Last 256 items | Keyword match, reverse order |
| **Graph** | Concept relationships (A→B) | Unlimited (dict) | Key lookup |
| **Ghost** | Compressed conversation history | Unlimited (JSONL file) | Keyword search |

### How a Write Is Routed

```
write(content, metadata={"salience": 0.7}, tier="episodic")
    ↓
HAL threshold check:
    hal.memory_threshold() → e.g. 0.6
    salience (0.7) >= threshold (0.6) → PASS → write to FAISS episodic
    salience (0.4) < threshold (0.6)  → REROUTE → write to short-term cache
    
Exception: metadata["kind"] == "threat_pattern" → ALWAYS write to episodic
```

HAL's emotional state directly controls what gets remembered long-term. In a high-stress state (cortisol up), the threshold rises — the system becomes *more selective* about what deserves long-term storage.

### Semantic Embedding Without GPU

When no embedding model is available (CPU-only environments), An-Ra uses a local projection:

1. Split text into tokens
2. For each token, extract character n-grams
3. Hash each n-gram with Blake2b to get an index
4. Add ±1 weighted by position to the vector at that index
5. Apply tanh to normalize

This gives a meaningful (though weak) semantic vector without any neural model.

### FAISS Episodic Store

FAISS is Facebook's library for fast approximate nearest-neighbor search in high-dimensional vector spaces. An-Ra stores memories as vectors and retrieves the most semantically similar ones to any query.

---

## 7. THE GOAL QUEUE & AGENT LOOP

### Goal Queue

**File:** `goals/goal_queue.py`

The goal queue is a **persistent priority queue** — goals survive restarts, crashes, and reboots. It persists to a JSON file and reloads on startup.

**GoalItem fields:**
```python
goal_id: str         # Unique ID
text: str            # What needs to be done
priority: int        # Lower number = higher priority
status: str          # "queued" | "in_progress" | "done" | "dead"
retry_count: int     # How many times it's been attempted
max_retries: int     # After this many failures → "dead"
last_error: str      # What went wrong last time
parent_id: str       # If this is a sub-goal
```

**Retry with backoff:**
On failure, priority is penalized: `new_priority = priority + 10 * retry_count`
So a goal that keeps failing gets pushed further back in the queue automatically.

**Successor goals:**
A completed goal can generate child goals:
```python
queue.generate_successor(parent_id="task_001", text="Follow up on analysis", priority=110)
```

### Orchestrator

**File:** `agents/orchestrator.py`

The orchestrator receives a task dict with a `kind` field and routes it:

```python
KIND_TO_COMPONENT = {
    "coder":    "agent_loop",
    "research": "agent_loop",
    "memory":   "memory",
    "critic":   "evaluation",
    "symbolic": "symbolic_bridge",
    "ghost":    "ghost_memory",
}
```

Before dispatching, it checks feature flags — if the target component is disabled, the task is skipped gracefully instead of crashing.

After dispatch: success → `goal_queue.complete(goal_id)`. Failure → `goal_queue.fail(goal_id, error)`.

### Agent Loop

**File:** `phase2/agent_loop (45k)/agent_main.py`

Four phases per task:

1. **Plan** — Break the goal into concrete subtasks
2. **Execute** — Run each subtask (can use filesystem, sandbox, symbolic bridge)
3. **Monitor** — Check if intermediate steps succeeded
4. **Evaluate** — Score the final output; decide complete / retry / generate successor

---

## 8. THE TRAINING PIPELINE

**File:** `training/train_unified.py`

### Three Modes

```bash
python -m training.train_unified --mode status   # Just show state, no training
python -m training.train_unified --mode session  # Daily lightweight run
python -m training.train_unified --mode train    # Full milestone run
python -m training.train_unified --mode eval     # Evaluation only
```

### The Owner-First Data Mix

An-Ra's training data is NOT random internet text. It is curated:

| Bucket | Share | Why |
|--------|------:|-----|
| Owner conversations & instructions | 65% | Develops the owner's voice and problem-solving style |
| Owner identity & selfhood | 15% | Creates gravitational resistance to identity drift |
| Teacher reasoning traces | 10% | Improves hard reasoning without teacher style dominating |
| Symbolic/code-verified samples | 5% | Truth anchor — these answers are provably correct |
| Replayed failures & corrections | 5% | Every mistake becomes future supervision |

**Key principle:** Teacher data is an *amplifier*, not the owner. An-Ra should get sharper without becoming generic.

### RLVR — Reinforcement Learning from Verifiable Rewards

**File:** `training/rlvr.py`

Standard RLHF needs human raters to score outputs. An-Ra uses verifiable rewards instead:

- Code tasks: does the code actually run and pass tests? → score
- Math tasks: is the answer numerically correct? → score
- Logic tasks: does the derivation check out? → score

No human labels needed. The verifier is ground truth.

**GRPO — Group Relative Policy Optimization:**
1. Generate G completions for the same prompt
2. Score all G with the verifier
3. Normalize: `advantage = (reward - mean) / (std + eps)`
4. Policy gradient update + KL penalty against frozen reference model
5. KL coefficient is controlled by HAL's hormonal state

### STaR — Self-Taught Reasoning

**File:** `training/star.py`

An-Ra generates its own reasoning chains. Format:
```
<think>
Step 1: ...
Step 2: ...
Therefore: ...
</think>
Final answer: ...
```

Only chains that produce **verified correct answers** become training data. If the model fails a problem but the correct answer is known, a lower-weight rationalization example is added. Over time, An-Ra learns to show its work correctly.

### Dynamic Regret

**File:** `training/dynamic_regret.py`

Tracks regret across training — how much cumulative performance was lost on hard examples compared to an oracle. Uses this to adaptively focus harder on weak areas in subsequent sessions.

---

## 9. THE SYMBOLIC BRIDGE

**Files:** `phase3/symbolic_bridge (45Q)/`

Language models are fluent. They produce confident-sounding text that can be wrong. The symbolic bridge stops that where truth is checkable.

**Four verification domains:**

| Domain | Tool | What it checks |
|--------|------|---------------|
| Math | Symbolic solver | `solve x^2 - 9 = 0` → roots, derivatives, integrals |
| Logic | Constraint checker | Satisfiability, deduction chains |
| Code | Execution + analysis | Does code run? Does it produce correct output? |
| General | Pattern matching | Known constants, identities, unit conversions |

**Result types:**
- `VERIFIED_CORRECT` — the answer was checked and is right
- `VERIFIED_INCORRECT` — the answer was checked and is wrong → flag and retry
- `UNVERIFIABLE` — cannot be checked deterministically → flag confidence

The symbolic bridge implements the DFC principle: *where truth is checkable, check it.*

---

## 10. GHOST MEMORY & OUROBOROS

### Ghost Memory

**Files:** `phase3/ghost_memory (45P)/`

The model has a context window — a maximum number of tokens it can "see" at once. Long conversations overflow this.

Ghost memory compresses past conversation turns:
1. Each turn gets compressed into a summary vector + metadata
2. Older turns **decay** — they fade unless reinforced by relevance
3. When a new turn arrives, the retriever searches ghost memory for related past context
4. Relevant fragments are injected back into the context window

This gives An-Ra a form of long-term conversational continuity without blowing the context window.

### Ouroboros

**Files:** `phase3/ouroboros (45O)/`

Named after the serpent eating its own tail. Ouroboros makes An-Ra reason about its own reasoning.

**Three passes:**
- **Pass 0** — Fast intuitive answer (high endorphin → trust this more)
- **Pass 1** — Critical review (normal weighting)
- **Pass 2** — Deep constraint reasoning (high cortisol/adrenaline → weight this more)

HAL controls the weights between passes based on current emotional state. Under threat or uncertainty, the system automatically shifts toward slower, more careful reasoning.

After generating an answer, Ouroboros can reject it and loop again — up to a configured milestone limit — until confidence criteria are met.

---

## 11. SELF-IMPROVEMENT & SELF-MODIFICATION

### Self-Improvement Engine

**Files:** `phase2/self_improvement (45l)/`, `innovation/`

The gap scanner reads telemetry — what failed, what was slow, what produced low-confidence outputs. It generates improvement hypotheses:

```
Gap detected: "symbolic_bridge" fails 23% of calculus queries
Hypothesis: Add derivative rule lookup before solver pass
Experiment: A/B test with/without rule lookup on benchmark set
Verifier: eval_harness.compare(baseline, system_on)
```

Hypotheses go into an action queue. Successful experiments get scored and retained. Failed experiments get noted. This is the AIE loop — An-Ra Innovation Engine.

### Self-Modification

**Files:** `self_modification/type_a.py`, `self_modification/type_b.py`

An-Ra can propose changes to its own code. Two types:

**Type-A — Safe patches:**
- Documentation, comments, formatting
- Low risk, auto-approved
- Goes through light review

**Type-B — Logic changes:**
- Changes to reasoning, memory, generation behavior
- **Must run in sandbox first** (`execution/sandbox.py`)
- Sandbox result must show no regressions
- Must pass sovereignty audit
- Only then promoted to production

**Filesystem Agent (`execution/fs_agent.py`):**
Handles safe file operations with path validation to prevent escaping the workspace.

---

## 12. THE SOVEREIGNTY SYSTEM

**Files:** `phase3/sovereignty (45R)/`

Sovereignty is the governance layer. Nothing gets promoted without passing it.

**What the auditor checks:**

1. **Source health** — All 19 registered component files exist
2. **Import health** — All components import without errors
3. **Dead code sweep** — No registered component is unreachable
4. **Benchmark deltas** — New version must be better (or neutral) on benchmarks
5. **Identity stability** — CIVGuard cosine similarity check
6. **Regression check** — Eval harness comparison against stored baseline

**Promotion gates:**
A checkpoint is only promoted to `anra_v2_brain.pt` (production) after passing all sovereignty checks. If it fails, the old checkpoint is kept and the new one is quarantined with a report.

---

## 13. THE ENGINEERING SPINE

**Files:** `engine/`

Every component is accountable. The spine enforces this.

### component_base.py
Base class every component inherits. Enforces:
- `name` — what component is this
- `enabled` — is it running
- `metric_hooks` — what must be measured

### feature_flags.py
Toggle any component without touching code:
```python
from engine.feature_flags import set_flag, is_enabled
set_flag("ghost_memory", False)    # disable ghost memory
is_enabled("symbolic_bridge")      # → True/False
```

Flags persist in `state/feature_flags.json`.

### telemetry.py
The `@trace` decorator wraps any function and automatically records:
```
module, operation, start_time, end_time, elapsed_ms,
success, error_type, error_message, token_count, output_size, confidence
```

Everything is written to `state/logs/telemetry.jsonl` — one JSON line per call.

```python
@trace("memory_router", "write")
def write(self, content, ...):
    ...  # automatically timed and logged
```

### metric_bus.py
The `@instrument` decorator tracks call counts and success rates by module:
```python
@instrument("identity")
def verify(self, ...):
    ...  # auto-counted, success/fail tracked
```

### eval_harness.py
Runs structured comparisons:
```python
harness = EvalHarness()
baseline = harness.run(test_prompts, system="baseline")
current = harness.run(test_prompts, system="system_on")
diff = harness.compare(baseline, current)
# Shows exactly which prompts regressed and by how much
```

### report.py
```bash
python anra.py --report
```
Produces a full operator scorecard across all 19 components in one command.

---

## 14. THE FALSIFICATION LEDGER

**File:** `identity/falsification_ledger.py`

Every serious claim An-Ra makes can be stored in the ledger with an explicit falsifier.

```python
ledger.append(
    claim="The transformer learns positional relationships via RoPE",
    status="VERIFIED",
    confidence=0.95,
    would_be_false_if="Attention patterns show no positional decay with distance",
    next_verifier="attention_pattern_analysis"
)
```

**Claim statuses:** `VERIFIED` | `INFERRED` | `ASSUMED` | `UNKNOWN` | `FALSIFIED`

**Why this matters:** It turns the question "is this true?" from a philosophical question into a trackable, auditable record. Claims that get falsified feed back into training as corrections.

**Export to training data:** `ledger.export_training_data()` converts every record into DFC-format training examples, closing the learning loop.

### Constraint Isomorphism Search

**File:** `identity/constraint_isomorphism_search.py`

This system finds structural analogies between domains. It represents each domain as a signature of `{state variables, operators, invariants}` and computes weighted similarity.

```python
# quantum_transpilation and electrical_routing share:
# - graph structure (qubits ↔ nodes)
# - routing operators (swap ↔ reroute)
# - connectivity invariants
# → valid analogy with score 0.84

# Allows reasoning: "this quantum problem is structurally like this routing problem"
# But only if the constraints actually survive the mapping
```

---

## 15. THE DFC TRAINING PARADIGM

**DFC = Differential Falsification Cognition**

This is An-Ra's most important original research contribution.

### The Problem With Standard LLM Training

```
internet text → next token prediction → helpful-sounding answers
```

The model learns to produce text that *sounds like* good reasoning. It doesn't learn to *do* good reasoning. It can be fluently wrong.

### The DFC Approach

Every hard problem is structured as a chain:

```
state → constraint → hypothesis → prediction → action/check
     → observation → error → update → memory → next action
```

Every serious claim must carry the condition that would prove it wrong.

**In training data (`frontier_dfc.jsonl`):**
```json
{
  "template": "HYPOTHESIS_CHAIN",
  "claim": "The optimization will converge in 50 epochs",
  "verify": "INFERRED",
  "confidence": 0.72,
  "falsifier": "Loss plateaus before epoch 30 without improvement",
  "next_verifier": "plot_loss_curve",
  "text": "<hyp>The optimization will converge in 50 epochs</hyp>\n<verify>INFERRED confidence=0.72</verify>\n<err>Loss plateaus before epoch 30 without improvement</err>\n<act>{\"tool\":\"plot_loss_curve\"}</act>"
}
```

### DFC Across Domains

| Domain | Constraint shape |
|--------|-----------------|
| Software | State transitions, invariants, tests, complexity |
| Math | Equations, proofs, boundary conditions |
| Code | Correctness, complexity, edge cases |
| Reasoning | Logic chains, consistency, counterexamples |
| Self-improvement | Capability deltas, regression detection |

---

## 16. HOW EVERYTHING CONNECTS

### The Runtime Loop

```
User input / goal
    ↓
tokenizer_v3 → token IDs
    ↓
CIV check → is identity stable? (score ≥ 0.7)
    ↓
HAL appraise → update hormones based on input context
    ↓
ESV update → update emotional state from HAL
    ↓
Ghost memory retrieval → inject relevant past context
    ↓
Symbolic pre-check → is this verifiable? run solver first
    ↓
CausalTransformerV2 → generate tokens
    (attention_temperature set by HAL)
    (ESV channel modulates attention)
    ↓
Ouroboros → re-examine with multiple passes (weighted by HAL)
    ↓
Symbolic post-check → verify answer if possible
    ↓
CIV verify → does output maintain identity standards?
    ↓
response
    ↓
memory_router.write() → store to appropriate tier (HAL gates threshold)
falsification_ledger.append() → store claim with falsifier
goal_queue.complete() or fail() → update goal state
telemetry trace → write to logs
```

### The Training Loop

```
training_data/anra_training.txt
    ↓
v2_data_mix → apply 65/15/10/5/5 bucket ratios
    ↓
tokenizer_v3 → tokenize batches
    ↓
CausalTransformerV2 → forward pass
    ↓
loss_tracker → track loss per bucket
    ↓
RLVR (GRPO) → verifier-shaped reward signal
STaR → reasoning chain distillation
    ↓
optimizer step (AdamW + scheduler)
    ↓
eval_v2 → compact eval on held-out examples
benchmark → score on benchmark suite
verifier → flag hard failures
    ↓
replay_pipeline → hard failures → next session data
    ↓
checkpoint.save() → atomic write to disk
    ↓
sovereignty audit → compare against baseline
    ↓
promote → anra_v2_brain.pt (if passed)
or hold → quarantine + report (if failed)
```

---

## 17. RUNNING THE SYSTEM

### Local

```bash
pip install -r requirements.txt

# See system status
python anra.py --status

# Full health report
python anra.py --report

# Check what components are enabled
python anra.py --phase3-status

# Test symbolic bridge
python anra.py --symbolic "solve x^2 - 9 = 0"

# Add a goal
python anra.py --goal "analyze the current training data quality"

# Training
python -m training.train_unified --mode status   # dry run
python -m training.train_unified --mode session  # daily training
python -m training.train_unified --mode train    # milestone
python -m training.train_unified --mode eval     # evaluate only

# Tests
python -m pytest tests/ -x -q
```

### Google Colab

Open `AnRa_Master.ipynb` in Colab with a T4 GPU. The notebook runs the full operator loop:

1. Configure session and component flags
2. Mount Drive and set up GPU/RAM
3. Clone or update repo
4. Merge training data
5. Restore checkpoints
6. Apply feature flags and print system report
7. Run training, eval, or status
8. Inspect telemetry and scorecard
9. Sync reports and checkpoints back to Drive

For quick demos, use `MODEL_SIZE = "25m"` and `TRAINING_MODE = "status"`.

### Toggle a Component

```python
from engine.feature_flags import set_flag, disabled_components
set_flag("ghost_memory", False)    # disable
set_flag("symbolic_bridge", True)  # enable
print(disabled_components())       # see what's off
```

### Read Telemetry

```python
from engine.telemetry import get_telemetry_bus
bus = get_telemetry_bus()
print(bus.summary_by_module())
# → {"memory_router": {"calls": 42, "success_rate": 0.97, "avg_ms": 12.3}, ...}
```

---

## 18. THE 19 COMPONENTS AT A GLANCE

| # | Component | Files | One Line |
|---|-----------|-------|----------|
| 1 | `brain` | `anra_brain.py` | Custom transformer: GQA, RoPE/YaRN, Flash SDP |
| 2 | `tokenizer` | `tokenizer/tokenizer_v3.json` | BPE, 8192 vocab, owner-trained |
| 3 | `data_mix` | `training/v2_data_mix.py` | 65/15/10/5/5 owner-first corpus |
| 4 | `training_loop` | `training/train_unified.py` | Daily and milestone training |
| 5 | `evaluation` | `training/eval_v2.py`, `benchmark.py` | Compact eval + hard-example feedback |
| 6 | `runtime` | `generate.py`, `inference/` | Generation, streaming, inference |
| 7 | `api_web` | `app.py` | FastAPI backend + web interface |
| 8 | `identity` | `identity/civ.py`, `esv.py`, `hal.py` | CIV + ESV + HAL identity stack |
| 9 | `memory` | `memory/memory_router.py` | 4-tier memory with HAL salience gate |
| 10 | `phase2_memory` | `phase2/memory (45J)/` | Typed recall + personal knowledge graph |
| 11 | `goals` | `goals/goal_queue.py` | Persistent priority queue, retry/backoff |
| 12 | `agent_loop` | `phase2/agent_loop (45k)/` | Plan → Execute → Monitor → Evaluate |
| 13 | `master_system` | `phase2/master_system (45M)/` | Long-horizon autonomy + owner control |
| 14 | `self_improvement` | `phase2/self_improvement (45l)/` | Gap scanner + hypothesis engine |
| 15 | `self_modification` | `self_modification/` | Type-A/B patch gates + sandbox |
| 16 | `ouroboros` | `phase3/ouroboros (45O)/` | Recursive multi-pass reasoning |
| 17 | `ghost_memory` | `phase3/ghost_memory (45P)/` | Compressed long-term conversation recall |
| 18 | `symbolic_bridge` | `phase3/symbolic_bridge (45Q)/` | Verified math, logic, code |
| 19 | `sovereignty` | `phase3/sovereignty (45R)/` | Audit, benchmarks, promotion gates |

---

## KEY NUMBERS TO KNOW

| Parameter | Value |
|-----------|-------|
| Registered components | 19 |
| Tokenizer vocabulary | 8,192 tokens |
| CIVGuard drift threshold | 0.92 cosine similarity |
| Minimum CIV passing score | 0.70 |
| Training data mix | 65 / 15 / 10 / 5 / 5 |
| Simulated hormones (HAL) | 7 |
| Memory tiers | 4 |
| Short-term memory capacity | 256 items |
| Production checkpoints | `anra_v2_brain.pt`, `anra_v2_identity.pt`, `anra_v2_ouroboros.pt` |
| Telemetry log | `state/logs/telemetry.jsonl` |
| Component registry | `runtime/system_registry.py` |

---

*An-Ra is not a product. It is an organism — built to learn, remember, verify, improve, and explain itself.*

*Every component has a name. Every call has a trace. Every claim has a falsifier. Every upgrade passes a gate.*

*That is what makes it sovereign.*
