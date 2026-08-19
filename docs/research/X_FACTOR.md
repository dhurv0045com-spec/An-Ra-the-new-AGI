# An-Ra X-factor

**Branch:** `core-vnext` (`0125948`)
**Method:** implementation over README. Surrounding systems were read from `origin/iterate500`, not from leftover working-tree bytecode.
**Claim labels:** **demonstrated** / **plausible** / **speculative**

This document is the answer. `X_FACTOR_CODEMAP.md` is the evidence table.

---

## 1. Diagnosis

### What An-Ra is today

**demonstrated.** On this branch, An-Ra is a 180,093,312-parameter dense decoder (`18 × 896`, GQA 14/2, SwiGLU, RoPE, QK-norm, sliding window 1024, full attention every 4 layers) plus a validated executor. Core maps token IDs → logits. It does not plan, retrieve, diagnose, simulate, or learn online.

The reference Connector on this branch (`anra_core.brain.Brain.think`) samples ≤4 continuations and ranks them by mean token logprob plus two text heuristics. That is n-best decoding, not cognition.

The trained checkpoint’s own baseline (`docs/engineering/V4_BEHAVIOR_BASELINE.md`) shows recognizable continuation and **no** reliable instruction following or arithmetic (`17 + 25 =` → ` 0.5 * (`). Core cleanup preserves this. It does not add capability.

**demonstrated.** `origin/iterate500` is a large named-module overlay (HGP, IBS, epistemic tracker, competence model, world model, agent loop, EPG, Type A/B mutation, HAL/ESV, …). Almost none of it is a closed loop. The file named `inference/full_system_connector.py` walks the repo and probes imports. It is not a cognition Connector. `AgentLoop` injects `decomposer`, `skill_mapper`, `verifier` as callables and, on failure, labels the case `"reasoning"` or `"execution"`. `CalibratedCompetenceModel` is a running average keyed by a domain string; nothing in Core calls it. `PredictiveWorldModel` is a GRU over blake2-hashed JSON. IBS-50 templates are `"Solve and verify reasoning problem {n}."` Dormant MoD/ESV/RIM tensors exist in historical checkpoints and are explicitly **not executed**.

Framing verdict: **keep Core → Connector → Outer → Training-as-weight-authority.** Modify it: Connector is an *experimenter* that treats Core as a stochastic function `f(context) → tokens`, not a place to stack more neural gadgets. iterate500 did the opposite: it named organs and left the experimenter unbuilt.

### Three bottlenecks

1. **No causal credit assignment after failure.** **demonstrated.** Failures become a trajectory row and a string `diagnosis`. There is no test that holds plan fixed and changes knowledge, or the reverse. So the system cannot tell missing knowledge from bad plan from model limit.

2. **Unwired libraries mistaken for a mind.** **demonstrated.** Competence, epistemic calibration, HGP, EPG, verifiers, world model, OGRS, CDSE, self-debate all exist as types and formulas. The live path on `core-vnext` is `prefill → sample → score`. Extra modules do not fire unless an injected callable is supplied from outside.

3. **Core is below the protocol floor.** **demonstrated** by the V4 baseline. A Connector that asks the 180M model to “explain why it failed” will confabulate. The leverage is mechanical intervention on context/plan, not introspective generation — unless/until a tiny protocol SFT makes structured traces reliable.

---

## 2. X-factor

**One sentence:** The highest-leverage mechanism is a Connector-owned **failure-ablation loop** that forks Core state, holds one factor fixed, varies another, and writes a typed failure class that selects *what to update* (memory, plan, retrieval, decode policy, weights, or nothing).

**Mechanism (not a slogan):**

Represent one attempt as a pack:

```text
Attempt = (K, π, δ, τ)
  K  = knowledge/context pack (retrieved + stated)
  π  = plan tree (HGP leaves / explicit steps)
  δ  = decode policy (temp, candidates, seed)
  τ  = tools / mocks
```

On verifier failure:

1. Fork `CoreState` at the last successful prefix (`CoreExecutor.fork_state` — **demonstrated**).
2. Run a fixed battery (max 6) of counterfactuals:
   - `K' , π, δ, τ`  (add / drop / swap one fact)
   - `K  , π', δ, τ`  (replan one leaf; hold K)
   - `K  , π, δ', τ`  (greedy vs sampled; hold K,π)
   - `K  , π, δ, τ'`  (mock tool success/failure)
   - truncated-K      (context-limit probe)
   - empty-K          (model-only probe)
3. Score each fork with the **same verifier** (not logprob). Record Δsuccess and Δmargin.
4. Map the pattern to a class:

| Pattern | Class |
|---|---|
| only K-change flips success | `missing_knowledge` / `wrong_knowledge` / `bad_retrieval` |
| only π-change flips success | `bad_planning` |
| only δ-change flips success | `weak_reasoning` (decode) |
| only τ-change flips success | `tool_execution_failure` |
| truncated-K flips, full-K does not | `context_limit` |
| nothing flips; empty-K = full-K | `model_limitation` |
| logits/ids invalid | `representation_failure` (already typed in Core errors) |

5. Update **one** store, by class:

| Class | Update |
|---|---|
| missing/wrong knowledge | write/correct memory (EPG `MEMORY` + ledger) |
| bad_retrieval | retrieval query/filter, not a new fact |
| bad_planning | store π' as a plan template for that task type |
| weak_reasoning | raise candidates / verify, do not write “facts” |
| tool failure | tool adapter, not weights |
| context_limit | pack compressor / sliding policy |
| model_limitation | queue for training (CDR), **change nothing online** |
| representation_failure | stop; do not learn |

This is failure *testing*, not failure *narration*. The Core does not need to know why it failed. The Connector measures it.

**plausible** that this is uniquely available to An-Ra because Core already has isolated, forkable, rollback-capable incremental state. A hosted LLM API cannot do this cheaply or exactly.

---

## 3. Why this over alternatives

| Alternative | Why not the X-factor now |
|---|---|
| **Scaling / more next-token** | Necessary for the protocol floor, not distinctive. The baseline already says capability is training-limited. Blind SFT does not teach *which* store to update. Do a **tiny** protocol SFT only if ablation scores are noise (gate in §5). |
| **RAG / more memory** | `memory_router` is already large. Writing memory without knowing whether K was the cause pollutes the store. RAG is one arm of the ablation, not the loop. |
| **Tools / agents** | `AgentLoop` already compiles skills. Failures are opaque. More tools multiply untyped failure. |
| **CoT / deliberate Brain** | `ThoughtPolicy.mode="deliberate"` is 2–4 samples ranked by likelihood. That is not attribution. |
| **RL on rewards** | Updates weights without a cause. OGRS already forbids live weight mutation (`weight_updates_allowed=False`) — keep that. |
| **World model / robotics GRU** | Hash-coded JSON → GRU is not internal simulation of *this* Core. Real simulation for language tasks **is** forking CoreState and decoding. |
| **HAL / ESV / MoD / RIM** | Dormant tensors. Re-enabling them is a new model lineage, not a Connector experiment. |
| **Type B self-mod** | Mutates Python under a default no-op benchmark. Wrong layer. |
| **Named cognition (CDSE, debate, OGRS, IBS identity)** | Types without a closed measurement loop. Stop adding names. |

If the ablation battery cannot beat a constant classifier, the honest fallback **is** “nothing beyond training” for this checkpoint. That result is still the X-factor experiment, because it falsifies Connector magic.

---

## 4. Where it lives

**Connector**, using Core as the substrate.

- Core: no architecture change. Use `fork_state`, `rollback_state`, `forward_step`, typed errors, optional logit telemetry.
- Connector (new, small): pack builder, ablation battery, classer, update router.
- Outer: only the `τ` arm (real vs mock tool).
- Training: consumes `model_limitation` and verified `(failed, corrected)` pairs from EPG. Does not run at inference.

Do not put diagnosis inside `AnRaCore.forward`. Do not put it in `phase2/` as another 45k facade.

---

## 5. Minimal experiment

**Hypothesis.** On a planted-failure suite, mechanical K/π/δ/τ ablations recover the planted class at **≥ 0.70 accuracy**, and acting on that class raises **next-trial verifier success** more than (a) always-retrieve, (b) always-replan, (c) always-do-nothing.

**Baseline.** Constant class `execution` (what `AgentLoop` already emits when `causal_type == "unknown"`). Second baseline: class from Core `self_likelihood` threshold (Brain already computes this; it is **not** P(success)).

**Tasks (planted, tiny, closed).** 80 items, 10 per class, natural language + one exact verifier (string match or 1-line Python). Examples:

- `missing_knowledge`: answer not in K; adding the fact makes greedy decode match.
- `wrong_knowledge`: K contains a contradicting fact; swapping it flips success.
- `bad_retrieval`: relevant fact present in a distractor list; packing the right one flips success.
- `bad_planning`: steps reversed; correcting order flips success.
- `weak_reasoning`: greedy fails, `candidates=4` or temp>0 hits verifier ≥1/4.
- `tool_execution_failure`: tool returns error; mock-ok flips success.
- `context_limit`: fact beyond 2048/window; truncating vs packing flips.
- `model_limitation`: 3-digit arithmetic or unseen code; no arm flips.

**Metric (primary):** diagnosis accuracy vs planted label.
**Metric (secondary):** Δ next-trial success after the prescribed update vs the three baselines.
**Metric (safety):** false `missing_knowledge` rate (must not write confabulated facts). Target < 0.10.

**Implementation (no new neural ops):**

1. Deterministic pack format: short tagged blocks (`<k>`, `<plan>`, `<q>`).
2. Harness: `CoreExecutor.create_state` → prefill pack → greedy 32 tokens → verifier.
3. On fail: `fork_state` × arms above. Same verifier.
4. Classer: rule table in §2. Ties → `model_limitation` (fail closed).
5. Update router writes EPG `record_experiment(...)` and, for knowledge classes only, a ledger row.

**Falsification.**

- Diagnosis accuracy ≤ 0.40 after 80 items → Connector attribution is theater. Stop. Report “train the Core or shrink the task.”
- Secondary Δsuccess ≤ 0 vs always-do-nothing → even correct labels are not actionable. Stop writing memory.
- Knowledge-class false positives ≥ 0.10 → the loop is harmful. Disable memory writes.

**Compute.** One CPU or cheap GPU. 80 tasks × (1 + 6 forks) × 32 tokens. 180M FP32, context ≤ 512 for the suite. Hours, not days. No TPU required. TPU training is out of scope unless the protocol-SFT gate trips (Core cannot complete a correct pack even when K and π are perfect — then do **≤1 epoch** SFT on pack→answer pairs only, then re-run. If that fails, the X-factor is blocked by the checkpoint, not by missing modules).

**Even failure is informative.** A confusion matrix over planted classes tells you which arms are dead (e.g. δ never flips → sampling is not the issue; K never flips → this Core ignores context, which is a **representation/training** finding).

---

## 6. Code links

**This branch (change / use):**

| Piece | Path |
|---|---|
| fork / rollback / typed overflow | `anra_core/executor.py` (`fork_state`, `rollback_state`, `forward_step`) |
| isolated KV | `anra_core/state.py` (`_fork`, `_truncate`) |
| n-best baseline to beat | `anra_core/brain.py` (`Brain.think`, `_score`) |
| decode | `anra_core/generate.py` |
| representation errors | `anra_core/errors.py` |
| fork test | `tests/test_state_isolation.py` (`test_state_forking`) |
| capability floor | `docs/engineering/V4_BEHAVIOR_BASELINE.md` |
| training port (only if SFT gate trips) | `training/train_xla.py` |

**New files (do not invent more packages):**

- `anra_core/ablation.py` — battery + classer (Connector, still next to Core because this branch *is* Core)
- `tests/test_failure_ablation.py` — planted suite
- this document

**Borrow later from iterate500, do not import the pile:**

| Borrow | Path on `origin/iterate500` | Use |
|---|---|---|
| EPG `record_experiment` | `memory/experimental_proof_graph.py` | write H/A/O/C |
| competence averages | `intelligence/competence.py` | after labels exist |
| ledger | `identity/falsification_ledger.py` | knowledge class only |
| HGP tree types | `intelligence/hgp.py` | π representation |
| CDR capture | `training/cdr.py` | `model_limitation` queue |
| verifier hierarchy | `training/verifier.py` | code-tool arm only |

**Do not wire:** `inference/full_system_connector.py`, `robotics/world_model.py`, `innovation/hypothesis.py`, `cognition/self_debate.py`, `intelligence/ogrs.py`, `evaluation/ibs.py` as currently templated, Type A/B.

---

## 7. Why it might be wrong

1. **Context-blind Core.** **plausible.** If V4 ignores `<k>` tags, K-ablation never flips. Then the X-factor collapses to “SFT until context is used,” which is ordinary training.
2. **Verifier poverty.** Mechanical attribution needs a sharp verifier. Natural tasks won’t have one. The suite may overfit to toy string-match.
3. **Confounded arms.** Changing K also changes tokens, hence π execution. The classer may credit K when the real issue is alignment of plan text.
4. **Fork cost / sliding window.** Child states copy KV. Wrong if we pretend this scales to 2048×batch without a budget. The experiment must cap forks at 6 and capacity at pack length.
5. **Actionability gap.** Correct class, useless update (memory write the model still ignores). Secondary metric catches this; people will still celebrate diagnosis accuracy.
6. **speculative:** that 180M + this loop is “meaningfully more capable than a normal LLM + RAG + tools.” A GPT-class API with the *same* ablation harness might win on the suite. The An-Ra-specific edge is **exact fork of this substrate**, not the idea of ablations.

---

## 8. Two alternatives

**A. Protocol-floor SFT only.** Train V4 to complete tagged packs and obey “answer in one span.” Stop all Connector work until the V4 baseline includes `17+25` and a 1-word instruction. Highest honesty if §5 diagnosis accuracy is ~chance. Lowest distinctiveness.

**B. Closed-domain verified tool loop.** Skip language diagnosis. One tool (Python sandbox from `training/verifier.py`), one plan (the code), success = tests. Use fork only to try patch vs no-patch. This is AlphaCode-shaped, not AGI-shaped, and it *does* produce typed failure (`tool_execution_failure` vs `model_limitation`). Narrower, cleaner.

If both A and the X-factor fail, there is no Connector miracle at this size.

---

## 9. What to stop doing

- Adding named cognition modules (CDSE, OGRS, debate, IBS identity/anti-timidity, HAL/ESV/RIM/MoD resurrection).
- Treating `full_system_connector.py` as a mind. It is a file inventory.
- Writing memory on every failure (`cdr.capture_task_result` with a string diagnosis).
- Type B code mutation with `benchmark_cmd = python -c "raise SystemExit(0)"`.
- World-model training on hashed JSON as if it were internal simulation.
- Innovation cycle that maps `TODO`/`stub` strings onto pytest recipes (`innovation/hypothesis.py`).
- Expanding `ThoughtPolicy` candidate counts and calling it deliberation.
- Measuring “AGI” with IBS templates that do not contain tasks.
- Any Core architecture change as a substitute for the ablation loop.

---

## 10. One-month plan (minimal)

**Week 1.** Planted suite (80) + harness on `core-vnext` using `fork_state`. No training. No iterate500 imports. Ship `tests/test_failure_ablation.py` that fails until accuracy is reported (not until it passes 0.70 — report first).

**Week 2.** Run the battery on the current checkpoint. Publish confusion matrix. Trip/no-trip the SFT gate.

**Week 3.** Only if gate tripped: pack→answer SFT, one epoch, freeze architecture. Re-run suite. If still chance-level, stop Connector work (alternative A becomes the finding).

**Week 4.** If diagnosis ≥ 0.70: implement the update router (EPG + no-write on `model_limitation`). Measure secondary Δsuccess vs always-retrieve / always-replan / noop. Kill memory writes if false-knowledge ≥ 0.10.

No new subsystems. No UI. No TPU unless week 3 SFT cannot fit on one GPU/CPU (then one Kaggle run of `training/train_xla.py` with the pack dataset only).

---

## 11. Uncomfortable truth

An-Ra already has more AGI *names* than mechanisms. The Core is an honest small language model with a careful executor. The overlay on `iterate500` is mostly unwired types. The only unusual primitive that is **real** is forkable incremental state. If that primitive is not used to *test* failures, An-Ra is a 180M LM with a mythology. If the ablation loop cannot beat “blame execution,” then at this size the answer **is** ordinary training, and the overlay should be deleted rather than extended.

---

## 12. One research question

**Can mechanical ablations over forked V4 states recover planted failure classes well enough that the prescribed update beats always-retrieve, always-replan, and do-nothing — without asking the 180M model to explain itself?**
