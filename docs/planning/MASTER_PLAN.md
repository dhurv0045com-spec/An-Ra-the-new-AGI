# AN-RA Master Plan — From Recovery to a Mind That Earns Its Claims

*Written 2026-07-04 by Claude (Fable 5) after a full working session inside this
codebase: 454 non-GPU tests passing, wiring audit completed, seven subsystems
revived into the live path with evidence. This plan is derived from the code as
it actually is, the hardware this project actually has, and the arithmetic of
what makes language models speak.*

---

## Part 0 — The Diagnosis (read this before anything else)

**Why the trained checkpoint cannot talk: it is undertrained by 20–100×.
Nothing else in this repository matters until that is fixed.**

The arithmetic:

| Quantity | Value |
| --- | ---: |
| Model parameters | 499,167,047 |
| Chinchilla-optimal training tokens (~20 tokens/param) | ~10B |
| Tokens seen by the best small models that talk (SmolLM-360M, Qwen2.5-0.5B) | 600B – 12T |
| Realistic tokens per 3-hour T4 session (fp16, batch-limited) | ~30–60M |
| Sessions needed to reach even 2B tokens on T4 alone | ~40–70 |
| Tokens per hour on one rented A100 (500M model, packed 1024 ctx) | ~200–300M |
| **Cost to train 10B tokens on a rented A100 (~$1.50–2/hr)** | **~$60–120** |

Three corollaries that shape everything below:

1. **Do not grow the architecture yet.** The V3 upgrade (1536–3072 hidden,
   36–40 layers) multiplies parameter count 3–15×, which multiplies the token
   requirement 3–15×, on the same compute. Growth comes *after* the 500M
   lineage demonstrably speaks (Stage 5), via function-preserving expansion so
   nothing already earned is lost (CSII).
2. **The tokenizer is a hidden 30–50% compute tax.** 8,209 tokens yields
   ~1.5–2.0 tokens per English word versus ~1.3 for a 32k BPE. Every training
   step and every context window pays it. The append-only V4 migration
   (supported; canonical IDs 0–8208 immutable) should be sized and executed
   *before* the big token campaign, not after.
3. **A 499M model will never be Jarvis by itself — the system will be.**
   Small models become useful through what surrounds them: verifiers that
   check, memory that persists, tools that act, and gates that decide. That
   is exactly the architecture this repo already believes in. The plan
   invests in the model until it speaks, and in the system so that a small
   honest model becomes a large honest capability.

**The two laws (unchanged, absolute):**

1. The weight lineage, tokenizer identity, and canonical token IDs 0–8208 are
   never silently touched. Growth only via named, versioned, verified
   migrations.
2. Nothing is called "improved" without ablation evidence (Gate 5), and
   nothing acts autonomously without clearing the sovereignty gate. Running
   is not helping; acting is not the same as being allowed to.

---

## Part I — What Is Already Done (this session, verified)

Foundation work completed 2026-07-03/04, all with tests, all in the
engineering log:

- **Evidence integrity:** deterministic seeded evaluation (replays exactly);
  vacuous coherence/repetition metrics replaced with measured fallbacks;
  `/sovereignty/status` runs real health checks instead of hardcoding "ok";
  capability graph probes real imports + health instead of filename substrings;
  feature flags fail closed on unknown names; unmeasured telemetry reports
  None, never fake zeros.
- **Subsystems revived into the live path:** ghost memory (durable per-session
  stores, native hash embedder, measured 0.30 retrieval threshold, isolation
  proven); identity injector (real `clean_response` cleanup live); CIV
  (per-session identity vector persisted across restarts, measured
  `router_civ_similarity` instead of constant 1.0); cognition in full-system
  chat (CRE classification + epistemic outcome recording per request); HAL fed
  real quality signal instead of a 0.60 constant.
- **DFC falsification pass:** on math/logic messages the 45Q symbolic bridge
  independently derives the answer (sympy) and scores the model's output;
  the score feeds HAL/CIV truthfulness. Correct answers score 1.0, wrong 0.0,
  live.
- **KV-cache parity gate strengthened to distribution level** (entropy +
  max-prob curves, 1e-3), with fault-injection tests proving the gate can
  actually fail. Cache stays off until the gate passes on the real checkpoint.

Baseline discipline for everything below: every slice ships with tests, the
full non-GPU suite green, and an engineering-log entry. `PROGRESS.md` records
cross-session state so any session can resume with "continue".

---

## Part II — The Plan

### Stage 1 — Make It Speak *(the critical path; everything else waits on this)*

**Goal:** a 500M checkpoint that passes the 200-prompt recovery gate at ≥80%
coherence, then the Gate 6 bars (≥90% coherence, ≥85% format compliance,
<1% repetition/EOS failures over 1,000+ generations).

**1.1 Forensics on the existing checkpoint (Gates 0–2, one session)**
- Freeze baseline: SHA-256 of checkpoint + tokenizer, config, corpus manifest,
  malformed-output samples. Never overwrite.
- Run `check_frontier_checkpoint.py` (exact tensor accounting) and the 500
  tokenizer probes.
- Deterministic greedy outputs, seed 0, cache off; run the 200-prompt gate.
- Decision per IMPROVEMENT.md: exact load + coherence <80% ⇒ classify as
  undertraining ⇒ continuation training (expected outcome given Part 0).
- Extract what the checkpoint DID learn: per-source validation loss,
  token-frequency analysis of outputs, n-gram novelty. This tells us whether
  the data pipeline fed it anything learnable at all.

**1.2 Tokenizer V4 (one session + one validation session)**
- Measure fertility of the current 8,209-token vocab on held-out samples of
  each corpus source (English prose, code, math). Expected: ≥1.5 tok/word.
- If confirmed, execute the append-only V4 migration to 16,384 (path already
  exists: `build_tokenizer_recovery.py`, `native_append_v4` backend, tests in
  place). Deterministic init of new rows; IDs 0–8208 untouched (Law 1).
- Gate: fertility improvement ≥20% measured on held-out text; decode/encode
  round-trip on the 500-probe suite; old checkpoints still load under the
  migration.

**1.3 The token campaign (the heart of the plan)**

Compute strategy, in order of preference:
1. **Rented A100/H100 burst (recommended, ~$60–120 for 10B tokens).** The
   single highest-leverage purchase this project can make. `build_brain.py`
   already takes shard manifests and continuation phases; it needs a
   multi-hour headless mode (checkpoint every 500 optimizer steps, resume on
   preemption — mostly exists).
2. **Colab TPU v2-8 fleet (free).** `build_brain_tpu.py` exists. ~20–40k
   tok/s ⇒ 1B tokens in 8–14 hours ⇒ Phase A in 3–5 sessions. Multiple
   accounts = experiment farm, one writable artifact per worker (rule already
   documented).
3. **T4 sessions (fallback).** Only for ablations, drafts, and eval — never
   the main campaign. The draft-proof pipeline (`train_draft_recovery.py`)
   stays the fast end-to-end pipeline validator (~30M tokens, 6-layer model)
   run before every campaign phase to catch data/tokenizer bugs cheaply.

Data (Gate 4 curriculum, budgets from IMPROVEMENT.md, quality upgrades):
- Phase A/B: 2B raw foundation tokens (FineWeb-Edu 55%), immutable shards,
  source-hash train/val splits, unique-token accounting (all implemented —
  verify with the draft pipeline first).
- Phase C: 200M code/math/science + verified DFC tokens. DFC labels come from
  verifiers (the symbolic bridge now wired at inference doubles as the
  training-data verifier), never model-invented.
- Phase D: 100M conversation/instruction tokens with **answer-token loss
  weighting** (exists — verify it triggers), format diversity, and the 2%
  identity/replay mix so the model knows who it is.
- Phase E: 10M verifier-replay/tool-call tokens in the exact tool-call format
  Stage 4's orchestrator will use at inference (train the format you serve).

Training-loop hardening (mostly exists; verify each with a draft run):
- WSD or cosine LR with warmup; z-loss; grad clipping (exists).
- Sequence packing for raw phases (exists: `raw_causal_shards_v1`);
  fp16-safe attention on T4 via the logit-bounded attention already in
  `anra_brain.py`.
- Eval every 250 optimizer steps against the *deterministic* compact suite
  (fixed this session), checkpoint every 500 on optimizer boundaries, select
  by held-out capability, not training loss (all per IMPROVEMENT.md).

**Stage 1 exit gate:** 200-prompt recovery gate ≥80% ⇒ continue curriculum;
Gate 6 bars ⇒ Stage 2 unlocked at full depth. Every phase transition logged
with checkpoint hash + eval evidence.

### Stage 2 — Make It Useful (system intelligence around the model)

**Goal:** a full-system mode measurably better than the bare model on the
private 500-task suite, proven by ablation (Gate 5), not asserted.

- **2.1 Verifier-first generation.** Extend the DFC falsification pass beyond
  math/logic: sandboxed code execution verifier (parse + run generated code
  against generated tests in a subprocess jail), unit conversions, date/time
  arithmetic. Verified outcomes feed HAL/CIV truthfulness (channel already
  wired). When a verifier rejects, the runtime retries once with the error
  appended — hypothesis → check → update, at inference time.
- **2.2 One memory, one budget.** Three memory systems exist (memory router,
  ghost store, session history). Unify behind a single context-assembly
  budget: identity / current message / history / retrieved memory each get
  token allocations (the optimizer exists), with ghost snippets as a
  retrieval source. Measure with synthetic continuity evals: plant facts N
  turns back, score recall@k. No memory claim without a recall number.
- **2.3 Prompting as a versioned contract.** The `H:/ANRA:` format, identity
  block, and memory-injection templates are checkpoint-coupled artifacts:
  version them with the tokenizer hash, test round-trip through the real
  tokenizer, and reject serving when template version ≠ checkpoint's trained
  format. A model trained on one format and served another is silent damage.
- **2.4 Tools behind the gate.** Operator tools (file/OS/CAD) already exist
  with narrow dispatch. Widen dispatch to trained tool-calls (Phase E format)
  but route *every* call through `sovereignty_gate.py` (Stage 3.2) with
  signed audit records.

**Stage 2 exit gate:** three-seed, same-checkpoint ablations show positive
deltas for memory, verifiers, and each native subsystem (MoD/RIM/DSTP/ESV/HAL)
with bounded latency and <2% protected-validation regression. Anything
negative gets fixed or turned off — the registry claims only what helps.

### Stage 3 — Make It Improve Itself (GEPA + AIE, honestly)

**Goal:** a self-improvement loop that really proposes, really tests, and
really accepts/rejects its own changes — with the falsification discipline
applied to itself.

- **3.1 GEPA loop made real.** `training/gepa.py` + `run_self_improvement.py`
  become: propose (bounded change: prompt template, sampling params, memory
  thresholds, data-mix weights — *not* weights or code at first) → predict
  (expected metric delta, written down before the run) → evaluate (the
  deterministic compact suite + targeted private categories, three seeds) →
  accept/reject against the prediction → log to the experimental proof graph.
  An accepted change without a pre-registered prediction is a bug, not a win.
- **3.2 The sovereignty gate as the single door.**
  `self_modification/sovereignty_gate.py` becomes the one choke point for:
  self-modification proposals, autonomous tool/device calls, and outside-world
  actions. Deny-by-default, capability-scoped allowlists, signed append-only
  audit log, owner-revocable. The epistemic tracker's live calibration (wired
  this session) feeds the gate: a system that knows when it's been wrong
  recently gets narrower permissions. Capability and gate ship together —
  same feature, not two.
- **3.3 Self-training proposals.** Once 3.1 is trusted on serving-level
  changes, extend proposals to training: "phase D needs more format-diverse
  data", backed by per-category eval evidence, emitted as a job manifest for
  the human (or later, the gate) to approve. The model proposes experiments
  on itself; the gates decide.

**Stage 3 exit gate:** ≥10 completed GEPA cycles with pre-registered
predictions; acceptance rate and calibration reported; zero gate bypasses in
the audit log; a demonstrated *rejected* proposal (a loop that never rejects
is not evaluating).

### Stage 4 — Make It Act (agents, and then the world)

**Goal:** goal-directed multi-step action beyond the chat window, every step
gated and traced.

- **4.1 Orchestrator loop:** goal → plan tree (`intelligence/hgp.py` exists) →
  tool acts (through the gate) → verify outcomes (Stage 2 verifiers) → commit
  to memory → replan. Traces expose every step in the Matrix UI.
- **4.2 Sandbox first:** the agent workspace (`agent_workspace/`) is the only
  writable surface until N gated sessions complete with zero violations.
  Then, deliberately: local device control behind per-capability grants.
- **4.3 Robotics/embodiment** (`robotics/sim_to_real`, `world_model`): starts
  only after the language+tool core passes Stage 2 gates — an embodied system
  that can't reason about consequences shouldn't be given actuators. First
  milestone: world-model rollouts predicting sim outcomes better than a
  no-model baseline (measured), sovereign-gated sim actions.

**Stage 4 exit gate:** end-to-end goal completion rate on a fixed 50-goal
suite; 100% of actions carry gate-audit records; rollback drill passes.

### Stage 5 — Let It Grow Without Dying (CSII → V3)

**Goal:** the earned 500M mind carries forward into a larger body.

- **5.1 Function-preserving growth:** implement Net2Net-style width/depth
  expansion (1280→1536 first, not 3072): new tensors initialized so the
  grown model's logits match the source within tolerance on a fixed probe
  set — verified before any further training. QK-Norm and RoPE-base 500k
  enter here as part of the V3 migration, both behind load-compat tests.
- **5.2 Continue, don't restart:** the grown model resumes the Gate 4
  curriculum. Success = it never scores below the pre-growth checkpoint on
  the protected validation set at any point in continuation (that is what
  "growing without dying" means, measurably).
- **5.3 Context growth** 1024→1536→2048 only after short-context and
  retrieval gates pass (per MASTER_GOALS.md).

**Stage 5 exit gate:** logit-parity report at growth time; no protected
regression during continuation; all Gate 6 bars re-passed at the new size.

### Stage 6 — Prove It (the standing obligation, not a phase)

- Private 500-task suite × 3 modes × 3 seeds + subsystem ablations on every
  promotion candidate (machinery exists; keep it deterministic).
- Signed release bundles binding checkpoint + tokenizer + corpus manifests +
  eval evidence; rollback drill before every promotion.
- Blinded human review for open-ended coherence (the one thing automation
  can't attest).
- `PROGRESS.md` + engineering log after every slice, so any session resumes
  exactly where the last one stopped.

---

## Part III — Priority Order and Effort Map

| Order | Work | Effort (sessions) | Unlocks |
| ---: | --- | ---: | --- |
| 1 | Stage 1.1 forensics on real checkpoint | 1 | The undertraining verdict with evidence |
| 2 | Stage 1.2 tokenizer V4 + fertility gate | 1–2 | 20–40% cheaper tokens forever |
| 3 | Draft-pipeline validation run | 1 | Confidence the campaign won't waste compute |
| 4 | Stage 1.3 Phase A–B campaign (TPU or rented GPU) | 3–10 | A model that produces language |
| 5 | Stage 1.3 Phase C–E curriculum | 3–6 | A model that converses and follows format |
| 6 | Stage 2 ablation-proven full system | 3–5 | Usefulness beyond the bare model |
| 7 | Stage 3 GEPA + sovereignty gate | 3–5 | Honest self-improvement |
| 8 | Stage 4 agents/actions | 4–8 | Acting past the chat window |
| 9 | Stage 5 growth to V3 | 4–8 | Scale without losing the lineage |

Do not reorder 1–5. Everything after 5 can interleave.

## Part IV — What This Plan Refuses To Do

- No pretrained external weights, ever (Law 1). The lineage stays An-Ra's.
- No architecture growth before the current model speaks — growth without
  tokens is a bigger silence.
- No "improved" without pre-registered predictions and ablation evidence.
- No autonomous action without the sovereignty gate — capability and consent
  ship together.
- No AGI claims. The repo's own words stand: promote only what survives the
  evidence.
