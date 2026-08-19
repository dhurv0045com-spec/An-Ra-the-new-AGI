# THE X-FACTOR

**An-Ra does not get more capable by adding RAG, tools, agents, CoT, RL, memory, or named cognition modules.**

Those already exist as types on `iterate500` and do not form a closed loop. The 180M V4 Core is a normal dense decoder. What it uniquely *has* is forkable incremental state.

---

## The X-factor (one sentence)

**A Connector-owned failure-ablation loop:** on verifier failure, fork Core state, hold one factor fixed and change another (knowledge vs plan vs decode vs tools), map the flip pattern to a typed failure class, and update **exactly one** store — memory, plan, retrieval, decode policy, tool adapter, training queue, or **nothing**.

The Core does not explain itself. The Connector measures.

---

## Attempt

```text
Attempt = (K, π, δ, τ)
  K  knowledge / context pack
  π  plan
  δ  decode policy
  τ  tools / mocks
```

## Classes → one update

| If this arm uniquely flips success | Class | Update |
|---|---|---|
| add / swap / retrieve fact | `missing_knowledge` / `wrong_knowledge` / `bad_retrieval` | write/correct memory, or fix retrieval — not both |
| change plan, hold K | `bad_planning` | store the better plan template |
| greedy vs sample, hold K,π | `weak_reasoning` | raise candidates / verify; do not write facts |
| mock tool ok vs error | `tool_execution_failure` | fix tool adapter, not weights |
| compact K vs long K | `context_limit` | compress the pack |
| nothing flips | `model_limitation` | queue for training; **change nothing online** |
| bad token IDs / overflow | `representation_failure` | stop; do not learn |

Ties fail closed to `model_limitation`.

---

## Where

**Connector**, using Core `fork_state` / `rollback_state`. Not inside `AnRaCore.forward`. Not another `phase2` facade. Not a new neural gadget (HAL/ESV/MoD/RIM stay dormant).

Code: `anra_core/ablation.py`

---

## Why this is the X-factor

A normal LLM + RAG + tools + agent loop **fails opaquely**. It cannot tell missing knowledge from a bad plan from a model limit, so it writes the wrong memory, replans the wrong node, or updates weights for the wrong reason.

Forkable KV state lets An-Ra *test* the diagnosis. That is the only unusual primitive that is **demonstrated** in this repo.

---

## Status

| Claim | Label |
|---|---|
| Core is 180M IDs→logits with real fork/rollback | **demonstrated** |
| iterate500 overlay is mostly unwired names | **demonstrated** |
| Oracle ablation recovers 80/80 planted classes | **demonstrated** (loop is well-posed) |
| V4 checkpoint can use this loop | **untested** (no checkpoint in repo) |
| This beats scale/RAG/tools at 180M | **plausible**, falsifiable; if Core ignores `<k>`, the answer is ordinary SFT |

Full argument: [X_FACTOR.md](X_FACTOR.md)
Evidence: [X_FACTOR_CODEMAP.md](X_FACTOR_CODEMAP.md)
Week-1 numbers: [X_FACTOR_WEEK1.md](X_FACTOR_WEEK1.md)
