# Phase 3 Integration

Phase 3 is the **deep-cognition band** of the 19-component stack — where answers get verified, identity gets reinforced, memory gets compressed, reasoning gets recursive, and promotion gets gated.

| Layer | Code | Job |
| --- | --- | --- |
| 08/19 | **45N** Identity | Inject owner context; clean outputs |
| 16/19 | **45O** Ouroboros | Multi-pass reflection; milestone refinement |
| 17/19 | **45P** Ghost Memory | Long recall without blowing context |
| 18/19 | **45Q** Symbolic Bridge | Math / logic / code with verdicts |
| 19/19 | **45R** Sovereignty | Audit, benchmark, promote or quarantine |

**Design rule:** Phase 3 deepens the mainline. It does not fork a second product.

---

## Integrated pipeline

```text
user prompt
  → 45Q pre-check (math / logic / code)
  → 45N identity context
  → 45J memory context
  → 45P ghost context
  → 45O adaptive passes (hard prompts)
  → generation
  → 45N cleanup
  → memory write
  → 45R artifacts (when scheduled)
```

---

## Commands (from repo root)

```bash
python anra.py --phase3-status
python anra.py --symbolic "solve x^2 - 9 = 0"
python anra.py --sovereignty-report
python anra.py --sovereignty-run
python scripts/status.py
```

---

## Module status

| Component | Source | Runtime notes |
| --- | --- | --- |
| 45N Identity | Active | Graceful fallback if identity file missing |
| 45O Ouroboros | Active | NumPy path = CPU-friendly; Torch = heavier |
| 45P Ghost | Active | Mock embeddings OK offline; real embeds optional |
| 45Q Symbolic | Active | Best with `sympy`, `scipy`, `numpy` |
| 45R Sovereignty | Active | `psutil` helpful; daemon/API local only |

---

## Import reality (folder names have spaces)

Phase folders look like `symbolic_bridge (45Q)/`. **Do not fight this from random scripts.**

| Approach | When |
| --- | --- |
| `python anra.py --symbolic "..."` | Normal operator path (paths injected) |
| `cd "phase3/symbolic_bridge (45Q)" && python demo.py` | Component-local experiments |
| `cd "phase3/ouroboros (45O)" && python test_ouroboros.py` | Ouroboros smoke |

---

## Checkpoints

```text
anra_v2_brain.pt
anra_v2_identity.pt
anra_v2_ouroboros.pt
```

Fresh clone + empty Drive = **source active, weights absent**. Train or restore before expecting generation quality.

---

## Per-component docs

| Component | README |
| --- | --- |
| Identity (45N) | [`identity (45N)/README.md`](identity%20(45N)/README.md) |
| Ouroboros (45O) | [`ouroboros (45O)/README.md`](ouroboros%20(45O)/README.md) |
| Ghost (45P) | [`ghost_memory (45P)/README.md`](ghost_memory%20(45P)/README.md) |
| Symbolic (45Q) | [`symbolic_bridge (45Q)/README.md`](symbolic_bridge%20(45Q)/README.md) |
| Sovereignty (45R) | [`sovereignty (45R)/README.md`](sovereignty%20(45R)/README.md) |
