# An-Ra Phase 3 — Full Integration Guide

## Architecture

```
anra.py (entry point)
    └── phase2/master_system (45M)/system.py  (MasterSystem v2.0)
            │
            ├── Phase 1: core/model.py  ─── LLMBridge (NumPy transformer)
            │
            ├── Phase 2 (operational):
            │   ├── 45I  Fine-tuning & evaluation pipeline
            │   ├── 45J  Memory (vector + graph + episodic + semantic)
            │   ├── 45k  Agent Loop (Goal→Plan→Execute→Evaluate, 50+ tools)
            │   └── 45l  Self-improvement (skill library, prompt optimizer)
            │
            └── Phase 3 (ALL CONNECTED):
                ├── 45N  Identity Injector  → injects An-Ra voice into every prompt
                ├── 45O  Ouroboros NumPy    → 3-pass recursive reasoning
                ├── 45P  Ghost State Memory → compressed conversational memory
                ├── 45Q  Symbolic Bridge    → verified math/logic/code reasoning
                └── 45R  Sovereignty Daemon → nightly code quality self-audit
```

---

## Phase 3 Modules — Current Status

| Component | Status | Requires | Works on CPU? |
|-----------|--------|----------|---------------|
| **45N Identity Injector** | ✅ Runtime-ready | nothing extra | ✅ Yes |
| **45O Ouroboros (NumPy)** | ✅ Runtime-ready | nothing extra | ✅ Yes |
| **45O Ouroboros (Torch)** | 🔧 Training path | torch + GPU | ❌ GPU needed |
| **45P Ghost State Memory** | ✅ Ready | numpy (+ sentence-transformers for real embeddings) | ✅ Yes |
| **45Q Symbolic Bridge** | ✅ Ready | sympy, scipy | ✅ Yes |
| **45R Sovereignty Daemon** | ✅ Ready | psutil | ✅ Yes |
| **45N Identity Training** | 🔧 Separate offline step | transformers, peft, GPU | ❌ GPU needed |

---

## How Phase 3 Connects to the System

### The Full Chat Pipeline (every `python anra.py --chat` message)

```
User: "solve x^2 = 9"
        │
        ▼
45Q detect() ─── Is this math/logic/code?  ──YES──► SymPy solves it
        │                                             Verified answer injected
        │ (if NATURAL query, bypasses 45Q)            into prompt context
        │
        ▼
45N IdentityInjector.inject()
  Prepends compact An-Ra identity context (8 representative exchanges)
        │
        ▼
45J Memory.prepare_prompt()
  Retrieves relevant episodic/semantic memories, injects as context
        │
        ▼
45P GhostMemory.build_ghost_prompt()
  Retrieves compressed past conversation turns, injects as context
        │
        ▼
45O OuroborosNumpy.adaptive_generate()
  Simple query? → 1 pass (fast)
  Complex query? → 3 passes (semantic → logic → adversarial verify)
        │
        ▼
45N IdentityInjector.clean_response()
  Strips "I am an AI" etc., replaces with An-Ra voice
        │
        ▼
Store turn in both 45J Memory AND 45P Ghost Memory (synchronized)
        │
        ▼
User sees: An-Ra response
```

### The Full Goal Pipeline (every `python anra.py --goal "..."`)

```
User: "solve x^2 = 9 and explain why"
        │
45Q: pre-augment with verified answer
45P: inject ghost context (what An-Ra remembers)
45k Agent: GoalInterpreter → Planner → Executor → Evaluator
45N: clean output, strip robotic phrases
45P + 45J: store result in memory
45R: nightly audit compares complexity before/after
```

---

## Quick Start for Each Module

### 45N — Identity at Runtime (no training needed)

```python
from phase3.45N.identity_injector import IdentityInjector

inj = IdentityInjector()  # loads anra_identity_v2.txt automatically
prompt = inj.inject("Who are you?")
response = llm.generate(prompt)
clean = inj.clean_response(response)
```

### 45N — Identity Training (GPU, separate step)

```bash
# Upload to Google Colab T4 or NVIDIA GPU machine:
python phase3/identity (45N)/train_identity.py  # trains phi-2 with LoRA (~2-4 hours)
python phase3/identity (45N)/test_identity.py   # verify: should be 5/5
```

### 45O — Ouroboros Recursive Reasoning (CPU)

```python
from phase3.45O.ouroboros_numpy import OuroborosNumpy

ouro = OuroborosNumpy(generate_fn=llm.generate)
response, n_passes = ouro.adaptive_generate("Prove that sqrt(2) is irrational")
print(f"Used {n_passes} passes")  # complex → 3 passes
```

### 45P — Ghost State Memory

```python
from phase3.45P.ghost_memory import GhostMemory, default_config

gm = GhostMemory(config=default_config())
gm.add_turn("user", "My project is called An-Ra.")
gm.add_turn("assistant", "I'll remember that.")

prompt = gm.build_ghost_prompt("What is my project?")
# → "[Ghost Context] ...\n\nWhat is my project?"
```

### 45Q — Symbolic Logic Query (CPU)

```python
from phase3.45Q.symbolic_bridge import query

result = query("solve x^2 - 4 = 0")
print(result.answer_text)   # "2; -2"
print(result.confidence)    # 1.0
print(result.verdict)       # VERIFIED

result = query("Is (A→B) ∧ (B→C) → (A→C) a tautology?")
print(result.answer_text)   # "TAUTOLOGY"
```

### 45R — Sovereignty Daemon

```bash
# Start the daemon (runs nightly, produces reports at 02:00–05:00)
python anra.py --sovereignty-report          # See last night's report
python anra.py --sovereignty-run             # Trigger pipeline now

# Or directly:
pip install psutil
python -m phase3.45R.sovereignty.service start
```

---

## CLI Reference (anra.py)

```bash
# Core
python anra.py                          # System dashboard
python anra.py --chat                   # Chat with full Phase 3 pipeline
python anra.py --goal "..."             # Agent goal with Phase 3 augmentation
python anra.py --status                 # All subsystem status (JSON)
python anra.py --briefing               # Morning briefing + sovereignty report

# Phase 3 specific
python anra.py --phase3-status          # Detailed Phase 3 module status
python anra.py --symbolic "query"       # Direct 45Q math/logic/code query
python anra.py --sovereignty-report     # Latest nightly improvement report
python anra.py --sovereignty-run        # Trigger improvement pipeline now
```

---

## Upgrade Path (Model Checkpoints)

As per the 45O README, the full model upgrade path is:

```
anra_brain.pt           ← Phase 1: base NumPy language model (core/)
anra_identity.pt        ← Phase 1 + 45N LoRA fine-tuning (GPU step)
anra_ouroboros.pt       ← + 45O torch Ouroboros (GPU step)
anra_ghost.pt           ← + 45P Ghost State Memory integration
anra_symbolic.pt        ← + 45Q Logic Bridge (CPU, just verified outputs)
anra_sovereign.pt       ← Final: all phases, nightly self-improvement
```

**Without GPU**: anra_brain.pt → 45N identity injector (CPU runtime) → 45O NumPy Ouroboros → 45P Ghost Memory → 45Q Symbolic Bridge → 45R Sovereignty.  
This is the current operational mode — everything works on CPU.

**With GPU**: Add LoRA fine-tuning for 45N identity, then torch Ouroboros (45O) for true recursive passes at the weight level.

---

## Testing

```bash
# Run all Phase 3 integration tests (CPU-only)
python -m pytest tests/test_phase3_integration.py -v

# Individual module smoke tests
python phase3/ouroboros (45O)/test_ouroboros.py        # needs torch
python phase3/symbolic_bridge (45Q)/symbolic_bridge/demo.py  # needs sympy
python phase3/sovereignty (45R)/sovereignty/demo.py      # needs psutil

# Full system test via CLI
python anra.py --status
python anra.py --phase3-status
```

---

## Environment Setup

```bash
# Minimal (Core only — CPU)
pip install torch numpy PyYAML tqdm transformers

# Phase 3 enabled (recommended)
pip install sympy scipy psutil sentence-transformers

# Full (includes identity LoRA training — GPU required)
pip install peft datasets accelerate bitsandbytes
```

---

*Built across Phase 1 through Phase 3 | Integration completed: An-Ra-Phase3-v2.0*
