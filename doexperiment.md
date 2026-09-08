# doexperiment.md — the CYMEK experiment queue

Written 2026-09-09 by the Cymek research/engineering agent, at operator
request: "whatever experiments you think should be done, no matter how
big — I will run them later."

This file is a PLANNING QUEUE, not a preregistration. Every experiment
below must get its own hash-bound preregistration (PLAN / THREATS /
DESIGN_REASONING / PREEXECUTION_AUDIT / PREREGISTRATION.json /
RUN_READINESS.json, two-commit freeze) before anything executes. The
discipline is never waived for size, urgency, or enthusiasm:

- matched controls, shared-parent/shared-tail forks where the question
  is causal;
- candidate-free evaluation (generated answers + valid stops only);
- actual-token accounting (steps never define exposure);
- split firewalls (DEV_CONTROLLER / DEV_MEASUREMENT / SEALED_RESERVED;
  sealed never steers);
- hardware-only resolvers; one absolute campaign deadline per run;
- TPU claims require TPU hardware evidence — CPU oracles and GPU runs
  never substitute;
- every experiment states, before execution, what result would change
  our mind and what would kill the hypothesis;
- no experiment runs on the local machine beyond unit tests, math
  oracles, and tiny fixtures. The big ones run on Colab GPU (research
  laboratory) or TPU (production engine).

Compute boundaries: LOCAL = tests only. COLAB GPU = 1–3 h/session
research (multi-session allowed with Drive-mirrored checkpoints and
hash-verified resume). TPU = production certification and the 500M
campaign only.

---

## The queue at a glance

| # | Experiment | Where | Cost | Unlocks |
|---|------------|-------|------|---------|
| X0 | Execute CYR-GPU-005 (FROZEN — run as-is) | Colab GPU | 2–3 h | The retention-policy decision |
| X0b | Return-bundle audit + exact-SHA audit + Citadel audit | local/agent | hours | Right to build on 005 |
| X1 | CYR-GPU-006 — freezing vs consolidation (displacement-matched LR) | Colab GPU | 2–3 h | The 500M LR/decay recipe |
| X2 | CYR-GPU-007 — plasticity cost of new skills | Colab GPU | 2–3 h | 500M mixture/curriculum design |
| X3 | CYR-GPU-008 — state-threshold law sweep | Colab GPU | 2–3 h | Controller: keep, tune, or drop |
| X4 | CYR-GPU-009 — P35-scale replication of the winner | Colab GPU (multi-session) or TPU | 4–10 h | Scale-transfer of the effect |
| X5 | PRE500M TPU certification | TPU only | 1–2 days | Right to call anything TPU-proven |
| X6 | Data-pipeline rehearsal at 1/1000 scale | Colab GPU / high-RAM | 2–4 h | E1/E3 machinery de-risked |
| E1–E6 | The launch gates (as defined in `blueprint/EXECUTION.md`) | Colab + TPU | the real program | `main_training_authorized=true` |
| LAUNCH | The 500M campaign per the frozen spec | TPU | days | The actual model |

Order is deliberate: X0 first (it is frozen and paid for), then the
mechanism trio X1–X3 (they decide the recipe the big gates will test),
then scale/certification, then the gates, then launch. Skipping ahead
burns 5B tokens on an untested recipe.

---

## X0 — Execute CYR-GPU-005 (do this first; it is already frozen)

- Status: READY_FOR_OPERATOR_COLAB_GPU_RUN; RUN_READINESS=true; pushed
  at `6b36dca`.
- Notebook:
  https://colab.research.google.com/github/dhurv0045com-spec/An-Ra-the-new-AGI/blob/cymek-500m-readiness/notebooks/cymek_colab_gpu_research_v5.ipynb
- Runbook: CELL 0 must print `CYR-GPU-005 PREEXECUTION GATE: PASS`
  (it checks out the frozen executable and verifies every hash), then
  CELL 1, then CELL 2; return `CYMEK_GPU_RESEARCH_V5_RESULTS.zip`.
- X0b (mandatory, same cycle): audit the returned bundle — canonical
  receipt hashes, timebox/INCONCLUSIVE honesty, red-team ledgers — then
  the exact-SHA audit of what actually ran, then the independent
  Citadel audit. Nothing builds on 005 before these three pass.

## X1 — CYR-GPU-006: freezing vs consolidation (displacement-matched LR)

- Question: is LOW's protection (ARK-007R: 0/12 vs 9/12 collapse) a
  consolidation mechanism, or just parameters barely moving? CYR-GPU-005
  measures the displacement ledger; this experiment MANIPULATES it.
- Design: reuse the 005 shared-parent fork contract unchanged. Arms from
  the same G90 parents: LOW (1e-5), HIGH (1e-3), and INTERMEDIATE — an
  LR (and/or shortened HIGH exposure) resolved pre-training to match
  LOW's relative parameter displacement over the continuation window,
  with the match verified from the ledger. Retention compared at
  matched displacement.
- Decision rule: LOW still wins at matched displacement →
  consolidation-like mechanism (optimizer-state dynamics matter;
  controller worth carrying into 500M). Advantage vanishes →
  near-freezing explains everything; the 500M recipe wants schedule-
  based decay, not a state controller.
- Red team: displacement/moment ledger per evaluation (already built);
  negative control — an arm with displacement ≈ 0 must show ≈ 0 update
  norms or the instrumentation is broken.
- Budget: one Colab session, same dose floors as 005.

## X2 — CYR-GPU-007: plasticity cost of new skills

- Question: what does acquiring a NEW capability cost the OLD one under
  each policy — and does brief scheduled HIGH rehearsal buy new-skill
  learning without old-skill collapse? (Arkenstone left new-skill cost
  explicitly unresolved; the 500M cognition mixture depends on it.)
- Design: after T2 G90 parents (same contract), arms: (a) HIGH on the
  new binding/registry family, (b) LOW hold + no new learning (control),
  (c) LOW hold + scheduled HIGH bursts on the new family (rehearsal
  schedule preregistered in ACTUAL tokens), (d) HYST governing the
  bursts. Both families scored candidate-free throughout; old-capability
  retention and new-capability acquisition both primary.
- Decision rule: if (c) ≈ (a) on new learning while ≈ (b) on retention,
  interleaved rehearsal earns its place in the 500M curriculum; if (c)
  loses on both, families need isolation (separate phases), which
  reshapes the mixture schedule.
- Non-negotiable: the registry task keeps the query-only variant design
  (no ARK-009 order confound) and reports NOT_INFORMATIVE at zero event
  rate rather than manufacturing a failure.

## X3 — CYR-GPU-008: the state-threshold law

- Question: is one hysteretic controller calibration valid across seeds
  and tasks, or does the threshold law need per-task tuning? Arkenstone
  called the exact state-threshold law unclear; nobody has swept it.
- Design: preregistered factorial over (enter_retention ∈ {0.80, 0.90,
  0.95}) × (reenter_plasticity ∈ {0.30, 0.50, 0.70}) × (confirmations ∈
  {2, 3}) at MICRO scale on qualified 005/006 parents, same fork
  contract, retention area as the response; include a no-controller
  fixed-time arm as anchor.
- Decision rule: a broad plateau → one preregistered calibration for
  500M; sharp optima or seed instability → the controller is a research
  object only, and 500M runs a fixed schedule.

## X4 — CYR-GPU-009: P35-scale replication

- Question: does the winning policy survive a 10× scale jump (8.6M →
  35.4M parameters)?
- Design: the 005 winner (post X1–X3 adjudication) replicated at the
  P35 finalist (35,411,328 params, verified) under the same fork
  contract. Multi-session Colab with Drive-mirrored checkpoints and
  hash-verified resume — or fold into the TPU PRE500M window if a
  single session cannot afford the dose floor. The resolver refuses
  starved runs; do not shrink the science to fit the hardware.

## X5 — PRE500M TPU certification (hardware truth or nothing)

- Scope (already scoped in the adapter + blueprint): real-path XLA on
  the declared topology, the accumulation oracle re-run ON HARDWARE,
  exact model/update equivalence, collective receipts, throughput
  curve, failure injection, checkpoint upload/redownload/restore
  canaries, topology receipt binding.
- Rule: every status currently reading
  IMPLEMENTED_PENDING_PRE500M_TPU stays there until THIS runs. CPU
  oracles were necessary; they are not sufficient. No 500M token is
  spent before this passes.

## X6 — data-pipeline rehearsal at 1/1000 scale

- Question: does the production entry actually run a 50M-token
  rehearsal end to end (manifest → pack → mixture schedule → demand
  planning → milestones → mirror → exact resume) without human
  intervention?
- Design: synthetic corpus at 1/1000 of production scale, generated by
  the frozen generators, run through the production trainer on Colab
  (high-RAM if needed). Every receipt the 500M campaign will require
  must appear here first. Production corpus decision untouched —
  DATA_NOT_READY stays until E1.

---

## Stage 3 — the launch gates E1–E6 (per `blueprint/EXECUTION.md`)

These are the program's own gates; the doc does not redefine them, it
sequences them. Each needs its runner implemented, representative data,
and its own preregistration:

1. **E1 — tokenizer and corpus identity**: matched P35 16k/24k/32k
   tournament on the declared, hash-bound corpus; promote from
   raw-byte/FLOP-matched evidence only. (`next_action` in the readiness
   receipt points here.)
2. **E2 — P35 architecture**: consistent 2:1-GQA shape/context
   comparison; replicate the top two learned arms.
3. **E3 — cognition data and objective**: 5/15/30% verified-cognition
   mixtures under CE only; query-swap stays a separate matched-compute
   experiment — it may not silently enter the launch objective.
4. **E4 — optimization and curriculum**: LR/batch/WSD stability at the
   P35 winner; exact resume across a checkpoint boundary; stable FP32
   moments; no Tier-1 worst-family collapse. X1–X3 outcomes feed the
   schedule decision here.
5. **E5 — M102 transfer**: winning recipe vs strong CE control at
   600M–1B tokens, two winning-recipe seeds, fresh natural transfer,
   exact checkpoint restoration.
6. **E6 — target and custody**: on the declared TPU/XLA topology,
   preflight, collective, throughput, failure-injection, and
   upload/redownload/restore canaries; bind every external identity.
7. Then: fill every external identity, regenerate the readiness
   receipt → `READY_FOR_FREEZE_REVIEW` → independent review validates
   experiments and custody, not just hashes → launch.

## LAUNCH — the 500M campaign

Per the frozen spec: 250,216,960-parameter dense 26×896 decoder, 14Q/7KV
2:1 GQA, 24,576 byte-BPE, CE-only, 5B real tokens, 65/20/15
natural/code-cognition mixture. The controller/fixed-decay decision,
the curriculum, and the mixture come from X1–X3 + E3/E4; the sealed
evaluation battery scores once at the end; promotion and abort rules
are the frozen ones. Any deviation mid-campaign is a new preregistered
experiment, never a silent edit.

---

## What would change this queue

- A CYR-GPU-005 verdict of INCONCLUSIVE (timebox/starve/fork failure)
  sends us back to dose engineering before X1–X3.
- New Arkenstone/BRAMASTRA results fetched at each preregistration
  supersede the trajectory claims here if their receipts verify.
- If X1 shows pure near-freezing AND X3 shows no robust controller, X4
  shrinks to a plain schedule check and the program saves a scale run.
- Nothing in this queue overrides the compute boundary or the freeze
  protocol. Bigger is not an excuse: it is the reason the discipline
  exists.
