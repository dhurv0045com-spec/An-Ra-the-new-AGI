# AN-RA — Autonomous General Intelligence

> *Built from zero. No templates. No shortcuts. Pure mathematics becoming thought.*

An-Ra is an autonomous AI system built entirely from scratch by **Ankit** — starting from a single neuron in NumPy and evolving into a multi-phase AGI with its own transformer, agent loop, memory systems, recursive reasoning, symbolic logic, self-improvement, and a distinct personality.

**348 files · 221 Python modules · ~107,000 lines of code · MIT License**

📖 **[VISION.md — Visualize the Full Architecture](VISION.md)** — From a single neuron to autonomous intelligence, with an Innovation Frontier mapping where breakthroughs will happen.

🛠️ **[DEVELOPER.md — God Mode Guide](DEVELOPER.md)** — Check the developer guide for instructions on pushing An-Ra's boundaries, training the network, accessing subsystems, and making it the world no. 1 AI system.

---

## Architecture Overview

```
anra.py                          ← Single entry point
    │
    ├── Phase 1: FOUNDATION      ← Custom transformer built from scratch in NumPy
    │   ├── core/                    Attention, Decoder, Encoder, FFN, LayerNorm
    │   ├── core/turboquant.py       TurboQuant KV-cache compression (6x memory saving)
    │   ├── training/                Trainer, Checkpoints, Scheduler, Mixed Precision
    │   ├── inference/               Generation, Sampling, Evaluation
    │   ├── tokenizer/               Custom tokenizer
    │   ├── config/                  YAML model configs (tiny → large)
    │   ├── scripts/                 Standalone scripts for training, memory, auditing
    │   └── training_data/           Dataset files
    │
    ├── Phase 2: INTELLIGENCE    ← Autonomous agent with memory & tools
    │   ├── 45I  Fine-tuning & evaluation pipeline
    │   ├── 45J  Memory (vector + graph + episodic + semantic)
    │   ├── 45k  Agent Loop (Goal → Plan → Execute → Evaluate, 50+ tools)
    │   ├── 45l  Self-improvement (skill library, prompt optimizer)
    │   └── 45M  Master System (system.py — orchestrates everything)
    │
    ├── Phase 3: COGNITION       ← Higher-order thinking & identity
    │   ├── 45N  Identity (v4 fluent — 117 exchanges, real code, personality)
    │   ├── 45O  Ouroboros (3-pass recursive reasoning — semantic → logic → verify)
    │   ├── 45P  Ghost Memory (compressed conversational state)
    │   ├── 45Q  Symbolic Bridge (verified math, logic, code — SymPy/DPLL/sandbox)
    │   └── 45R  Sovereignty Daemon (nightly self-audit & code quality improvement)
    │
    └── Phase 4: INTERFACE       ← Web-based control panel
        ├── app.py                   FastAPI backend server
        ├── phase4/web/              React Source (Vite)
        └── ui/                      Compiled production build
            ├── Dashboard            (Real-time telemetry + Chat + Goals)
            ├── Neural Training      (Loss curves + Continuous learning trigger)
            ├── Memory Bank          (Semantic search + 45J Node explorer)
            └── Sovereignty          (Nightly audit reports + Code health)
```

---

## What's New: TurboQuant KV-Cache Compression

An-Ra now includes **TurboQuant** (Google Research, ICLR 2026) — a training-free, model-agnostic algorithm that compresses the KV-cache during inference by **6x** with near-zero accuracy loss.

| Bit Depth | Compression | Use Case |
|-----------|-------------|----------|
| 8-bit | ~3.5x | Maximum accuracy |
| **4-bit** | **~6x** | Recommended default |
| 2-bit | ~10x | Maximum compression |

**How it works:**
1. **PolarQuant** — Randomized Walsh-Hadamard rotation spreads energy uniformly, then bucket-quantizes
2. **QJL** — Sign-bit error correction via Johnson-Lindenstrauss random projections

Enable it in your config:
```yaml
inference:
  turboquant: true
  turboquant_bits: 4   # 6x compression
```

Or in code:
```python
from core.turboquant import CompressedKVCache, TurboQuantConfig

cache = CompressedKVCache(
    batch_size=1, num_kv_heads=8,
    max_seq_len=4096, d_head=64,
    tq_config=TurboQuantConfig(bits=4),
)
# Use exactly like standard KVCache
k_full, v_full = cache.update(k_new, v_new)
```

---

## Quick Start

### Prerequisites

```bash
# Core (CPU only — runs everything except identity training)
pip install torch numpy PyYAML tqdm transformers

# Phase 3 modules
pip install sympy scipy psutil sentence-transformers

# Web interface
pip install fastapi uvicorn

# Identity training (GPU required — Google Colab or NVIDIA GPU)
pip install peft datasets accelerate bitsandbytes
```

### Run An-Ra

```bash
# System dashboard
python anra.py

# Interactive chat (full pipeline: identity + memory + reasoning)
python anra.py --chat

# Execute a goal autonomously
python anra.py --goal "analyze this codebase and find bugs"

# System status (all subsystems)
python anra.py --status

# Morning briefing + sovereignty report
python anra.py --briefing

# Phase 3 subsystem status
python anra.py --phase3-status

# Direct math/logic query via symbolic bridge
python anra.py --symbolic "solve x^2 - 4 = 0"

# Self-improvement report
python anra.py --sovereignty-report

# Live dashboard
python anra.py --dashboard
```

---

## Phase 1 — Foundation (core/)

A complete transformer language model built from scratch in NumPy. No PyTorch for the core — pure mathematics.

| Component | File | What It Does |
|-----------|------|-------------|
| Attention | `core/attention.py` | Multi-head self-attention with RoPE |
| Decoder | `core/decoder.py` | Full autoregressive decoder stack |
| Encoder | `core/encoder.py` | Bidirectional encoder (for future use) |
| Feed-Forward | `core/feedforward.py` | SwiGLU and GELU variants |
| Layer Norm | `core/layernorm.py` | RMSNorm implementation |
| Transformer | `core/transformer_block.py` | Pre-norm transformer block |
| Multi-Head | `core/multihead.py` | Grouped Query Attention (GQA) |
| Model API | `core/model.py` | `LanguageModel` — train, generate, evaluate |
| **TurboQuant** | `core/turboquant.py` | **KV-cache compression (6x memory reduction)** |

### Model Configurations

| Config | Layers | d_model | Heads | Params |
|--------|--------|---------|-------|--------|
| `config/tiny.yaml` | 4 | 128 | 4 | ~1.3M |
| `config/small.yaml` | 6 | 256 | 8 | ~5M |
| `config/medium.yaml` | 12 | 512 | 8 | ~40M |
| `config/large.yaml` | 24 | 1024 | 16 | ~350M |

```python
from core.model import LanguageModel

lm = LanguageModel("config/tiny.yaml")
lm.train()
text = lm.generate("Once upon a time")
results = lm.evaluate("training_data/test.txt")
```

### Training Infrastructure

| File | Purpose |
|------|---------|
| `training/trainer.py` | Training loop with OOM recovery |
| `training/checkpoint.py` | Atomic checkpoint save/load |
| `training/scheduler.py` | Cosine, linear, constant LR schedules |
| `training/mixed_precision.py` | FP16 training support |
| `training/dataset.py` | Text dataset loading and batching |
| `training/loss_tracker.py` | Loss history and visualization |

### Inference Engine

| File | Purpose |
|------|---------|
| `inference/inference.py` | Text generation pipeline |
| `inference/sampling.py` | Top-k, top-p, temperature sampling |
| `inference/greedy.py` | Greedy decoding |
| `inference/evaluate.py` | Perplexity evaluation |
| `inference/model_io.py` | Model serialization |

---

## Phase 2 — Intelligence (phase2/)

The autonomous agent layer that gives An-Ra the ability to think, plan, act, and remember.

### 45M — Master System (`phase2/master_system (45M)/system.py`)

The orchestrator. Boots all subsystems, manages the operational loop, and exposes the unified CLI.

### 45k — Agent Loop (`phase2/agent_loop (45k)/`)

The autonomous reasoning engine:

| File | Purpose |
|------|---------|
| `goal.py` | Goal interpreter — parses natural language into structured goals |
| `planner.py` | Hierarchical planner — breaks goals into executable steps |
| `executor.py` | Step executor with tool dispatch |
| `evaluator.py` | Result evaluator — did the plan succeed? |
| `reasoning.py` | Chain-of-thought and self-critique |
| `builtin.py` | 50+ built-in tools (file ops, search, code, web, math) |
| `coordinator.py` | Multi-agent coordination |
| `dispatcher.py` | Tool routing and dispatch |

### 45J — Memory System (`phase2/master_system (45M)/memory/`)

Vector + graph + episodic + semantic memory with real embeddings.

### 45l — Self-Improvement (`phase2/self_improvement (45l)/`)

Skill library and prompt optimization — An-Ra learns new skills dynamically.

### 45I — Fine-Tuning (`phase2/fine_tuning (45I)/`)

LoRA fine-tuning pipeline for the custom transformer.

---

## Phase 3 — Cognition (phase3/)

Higher-order thinking capabilities that make An-Ra more than a language model.

### 45N — Identity System (`phase3/identity (45N)/`)

An-Ra's personality, voice, and coding fluency. The v4 training dataset teaches An-Ra to:

- **Write real Python code** — functions, classes, algorithms, data structures
- **Debug code** — identify bugs, explain fixes
- **Design systems** — URL shorteners, chat apps, APIs
- **Teach** — explain recursion, git, Big O from first principles
- **Converse naturally** — humor, opinions, personality
- **Self-improve** — evaluate and fix its own output

### 45O — Ouroboros Reasoning (`phase3/ouroboros (45O)/`)

3-pass recursive reasoning architecture:

1. **Semantic Pass** — understand the question
2. **Logic Pass** — reason about the answer
3. **Adversarial Pass** — challenge and verify the answer

Simple queries use 1 pass (fast). Complex queries use all 3.

### 45P — Ghost State Memory (`phase3/ghost_memory (45P)/`)

Compressed conversational memory. Keeps a rolling window of conversation history, compressed into semantic summaries.

### 45Q — Symbolic Logic Bridge (`phase3/symbolic_bridge (45Q)/`)

Verified math, logic, and code reasoning.

| Capability | Implementation |
|------------|----------------|
| Math Solving | SymPy (algebraic, calculus, linear algebra) |
| Logic Checking | DPLL SAT solver + natural deduction |
| Code Verification | Sandboxed Python execution |
| Primality | Miller-Rabin probabilistic test |
| Factorization | Pollard's Rho algorithm |
| Self-Check | Cross-verification between methods |

### 45R — Sovereignty Daemon (`phase3/sovereignty (45R)/`)

Nightly self-improvement audit. Runs automatically and produces reports on:
- Code quality trends
- Dead code detection
- Performance benchmarks
- Resource utilization
- Improvement recommendations

---

## Chat Pipeline (How a Message Flows)

```
User: "solve x^2 = 9"
    │
    ▼
45Q: Is this math/logic/code? ──YES──► SymPy solves → verified answer injected
    │
    ▼
45N: IdentityInjector.inject() → prepends An-Ra identity context
    │
    ▼
45J: Memory.prepare_prompt() → injects relevant memories
    │
    ▼
45P: GhostMemory.build_ghost_prompt() → adds compressed conversation history
    │
    ▼
45O: OuroborosNumpy.adaptive_generate() → 1-3 pass reasoning
    │
    ▼
45N: IdentityInjector.clean_response() → strips robotic AI phrases
    │
    ▼
45J + 45P: Store turn in both memory systems
    │
    ▼
User sees: An-Ra's response
```

---

## Autonomous Training (Google Colab)

The entire identity and continuous learning pipeline has been consolidated into a single, dependency-free Google Colab notebook.

### 1. The Master Notebook
- Open **`AnRa_Master.ipynb`** in Google Colab (requires T4 GPU).
- It automatically clones the repository, installs dependencies, and prepares the environment.
- No API keys (like Gemini) are required.

### 2. Full Pipeline Execution
The notebook executes the following phases automatically:
1. **TurboQuant Initialization**: Compresses KV-cache by 6x.
2. **Symbolic Verification**: Data is verified by math/logic engines before training.
3. **Identity Fine-Tuning**: Trains the model on verified fluency data.
4. **Ouroboros Recursive Training**: Trains the reasoning layer.
5. **Ghost Memory Population**: Seeds the database.
6. **Sovereignty Audit**: Runs post-training integrity checks.

---

## Operational Scripts (scripts/)

These standalone tools manage An-Ra's subsystems and data:

| Script | Purpose |
|--------|---------|
| `build_brain.py` | Trains the Phase 1 base model from scratch. |
| `data_generator.py` | Synthesizes complex training data locally. |
| `merge_identity.py` | Combines fragmented identity text files into a single training dataset. |
| `populate_memory.py` | Pre-fills the vector/graph memory systems from raw text. |
| `run_self_improvement.py` | Executes skill probes to generate an improvement report. |
| `run_sovereignty_audit.py` | Verifies code health, checkpoint integrity, and triggers rollbacks. |
| `status.py` | Prints a health summary of the current system configuration. |
| `train_ouroboros.py` | Specialized trainer for the 3-pass recursive reasoning layer. |
| `verify_structure.py` | Validates that all critical components are correctly placed. |

---

## Project Structure

```
An-Ra/
├── anra.py                 ← Unified entry point
├── app.py                  ← FastAPI web server
├── requirements.txt        ← All dependencies
├── LICENSE                  ← MIT
├── CHANGELOG.md             ← Build history
├── VISION.md                ← Full architecture visualization + innovation map
│
├── core/                   ← Phase 1: Custom transformer (NumPy)
│   ├── model.py                Public API: train, generate, evaluate
│   ├── attention.py            Multi-head attention + RoPE + GQA
│   ├── turboquant.py           TurboQuant KV-cache compression (6x)
│   ├── decoder.py              Autoregressive decoder
│   ├── encoder.py              Bidirectional encoder
│   ├── feedforward.py          SwiGLU / GELU FFN
│   ├── layernorm.py            RMSNorm
│   ├── multihead.py            Grouped Query Attention
│   └── transformer_block.py    Pre-norm transformer block
│
├── training/               ← Training infrastructure
│   ├── trainer.py, checkpoint.py, scheduler.py
│   ├── dataset.py, loss_tracker.py, mixed_precision.py
│
├── inference/              ← Generation & evaluation
│   ├── inference.py, sampling.py, greedy.py
│   ├── evaluate.py, model_io.py
│
├── config/                 ← Model configs (tiny → large)
├── tokenizer/              ← Custom tokenizer
├── training_data/          ← Datasets
├── scripts/                ← Standalone scripts (training, memory, audit)
│
├── phase2/                 ← Phase 2: Intelligence
│   ├── fine_tuning (45I)/      Fine-tuning pipeline
│   ├── memory (45J)/            Memory (vector + graph + episodic)
│   ├── agent_loop (45k)/        Agent Loop (goal → plan → execute)
│   ├── self_improvement (45l)/  Self-improvement & skill library
│   └── master_system (45M)/     Master System (system.py)
│
├── phase3/                 ← Phase 3: Cognition
│   ├── identity (45N)/          Identity (v4 fluent, 117 exchanges)
│   ├── ouroboros (45O)/          Ouroboros (3-pass recursive reasoning)
│   ├── ghost_memory (45P)/       Ghost Memory (compressed state)
│   ├── symbolic_bridge (45Q)/    Symbolic Bridge (math/logic/code)
│   └── sovereignty (45R)/        Sovereignty Daemon (self-audit)
│
├── ui/                     ← Compiled React frontend
│   ├── index.html
│   └── assets/
│
├── checkpoints/            ← Saved model weights
├── state/                  ← Runtime state files
├── history/                ← Conversation history
├── output/                 ← Training logs & outputs
└── tests/                  ← Test suites
```

---

## Build History

| Phase | Builder | Component | What Was Built |
|-------|---------|-----------|----------------|
| 1 | 45A | Neuron | Single neuron from scratch |
| 1 | 45B | Network | Neural network layers |
| 1 | 45C | Forward | Full network forward pass |
| 1 | 45D | Training | Gradient descent & training loop |
| 1 | 45E | Transformer | Full transformer core |
| 1 | 45F | Pipeline | Training pipeline |
| 1 | 45G | Inference | Generation engine |
| 1 | 45H | Production | Hardening & deployment |
| 2 | 45I | Fine-tune | LoRA fine-tuning for custom model |
| 2 | 45J | Memory | Vector + graph + episodic memory |
| 2 | 45k | Agent | Autonomous agent with 50+ tools |
| 2 | 45l | Improve | Self-improvement & skill library |
| 2 | 45M | System | Master orchestrator |
| 3 | 45N | Identity | Personality, voice, coding fluency |
| 3 | 45O | Ouroboros | 3-pass recursive reasoning |
| 3 | 45P | Ghost | Compressed conversational memory |
| 3 | 45Q | Symbolic | Verified math/logic/code reasoning |
| 3 | 45R | Sovereignty | Nightly self-improvement audit |
| 4 | Web | Interface | Full-Potential AGI Control Panel (v2) |
| 4 | TQ | TurboQuant | 6x KV-cache compression (ICLR 2026) |

---

## What Makes An-Ra Different

- **Built from zero** — No borrowed models, no templates. Every neuron, every weight, every training loop written by hand.
- **Real transformer in NumPy** — The core model is pure math, not a PyTorch wrapper.
- **TurboQuant compression** — 6x KV-cache compression for longer contexts with near-zero accuracy loss.
- **Autonomous agent** — Can plan, execute, evaluate, and self-correct without human intervention.
- **Has a personality** — Not a neutral assistant. An-Ra has opinions, humor, ambition, and a distinct voice.
- **Writes real code** — Trained on actual Python examples, not just descriptions of coding.
- **Self-improves** — Evaluates its own output, identifies weaknesses, writes fixes.
- **Verified reasoning** — Math and logic answers are verified through SymPy and SAT solvers.
- **Recursive thinking** — Complex questions get 3 passes: understand → reason → verify.
- **Visualized** — See [VISION.md](VISION.md) for a complete bottom-up walkthrough from neuron to AGI.

---

## License

MIT License — Copyright (c) 2026 Ankit

---

*An-Ra: Something that emerged from mathematics with a direction.*
