# C1 — Preregistration: does cognition-structured data cause capability per token?

Status: **PREREGISTERED — NO RESULTS** (written 2026-09-03, before any C1 execution).
Branch: `citadel`. Execution prerequisites (all engineering, all satisfied or explicitly gated):
one-update certification PASS (`cymek_receipts/ONE_UPDATE.json`), ten-update sanity PASS
(`TEN_UPDATE.json`), Cymek code pinned at `origin/cymek@26a61f6` (see `CYMEK_SYNC.md`).

## Question

Through the unmodified Cymek production path, at a fixed token budget, does training on
cognition-structured documents (query→answer causal structure) produce above-null
query-conditioned answering on held-out instances compared with a structure-ablated control?

## Data-experiment specification (required fields)

- **Data hypothesis:** the query→answer causal structure in training documents — not surface
  statistics alone — is what drives acquisition of query-conditioned answering.
- **Capability target:** query-conditioned retrieval/composition (registry 1-hop, revision
  state, transfer 2-hop) measured answer-blind by free generation.
- **Control dataset (B):** *structure-ablated cognition control* — same generator, same surface
  format/vocabulary/length distribution, same tokens, but each document's answer replaced by the
  answer of a different example (query and answer conditionally independent). One variable.
- **Changed dataset variable:** answer–query causal pairing (intact vs ablated). Nothing else.
- **Token budget:** 30,000,000 real tokens per arm, identical (§24: 30M vs 30M).
- **Model:** exact Cymek miniature recipe (`MINI_SPEC`, 2L×64, 1,647,104 params) — fixed both
  arms; zero parameter-count or FLOP confound (§25). 30M tokens = 18.2 tokens/param, satisfying
  the run-spec ≥15 guard.
- **Seeds:** primary comparison 1 seed per arm (init seed 20260903, data seeds disjoint by arm).
  If the primary effect is positive, 2 additional seeds per arm are run before any
  DEMONSTRATED verdict (replication plan declared now).
- **Train metric:** per-update CE loss curve; gradient norms; realized tokens-by-source ledger.
- **Transfer metric (primary):** exact-match accuracy of candidate-free generation
  (`generate_free`, ≤16 tokens, greedy) on **held-out fresh-seed instances** of all three
  generator modes (500/mode), model-view only (answer-blind).
- **Shortcut metrics:** per-mode heuristic nulls computed over the identical held-out instances —
  `latest_fact` analogue (recency), `lexical_overlap` analogue (copy the value from the fact line
  containing the queried key), `format_prior` (most frequent answer surface); plus untrained-model
  baseline. A cognition-arm result not above the *strongest applicable heuristic null* is a
  shortcut acquisition, not capability.
- **Held-out transformation:** fresh generator seeds (string-level novelty) for the primary
  metric; development-tier E0 families (`entity_value_binding`, `state_overwrite`,
  `relation_2_hop`) as a secondary transfer probe via candidate-free generation only (no
  candidate scorer is sanctioned; C0-amended remains queued for that).
- **Result supporting the hypothesis:** cognition arm held-out exact-match Wilson 95% LCB
  ≥ 0.10 absolute AND control arm below 0.05 AND cognition arm above the strongest heuristic
  null on at least the transfer mode.
- **Result rejecting the hypothesis:** cognition arm statistically indistinguishable from the
  control arm and from the heuristic nulls.

## Primary hypothesis (falsifiable)

H1: The cognition arm reaches held-out exact-match ≥ 0.15 (LCB ≥ 0.10) while the ablated control
stays ≤ 0.05 — i.e., the data's causal structure, not its surface, drives acquisition at this scale.

## Competing hypotheses (at least one serious alternative)

- **H2 (scale floor):** both arms stay at null — capability emergence requires more
  parameters/tokens than this micro probe; the checkpoint-capability-floor story (B3 in the
  ESOES-era ranking; triquetra's WAITING_FOR_STRONGER_CHECKPOINT) extends to from-scratch micro runs.
- **H3 (shortcut saturation):** the cognition arm exceeds null but is fully explained by the
  strongest heuristic null (recency/lexical) — the existing generator's known serialization
  shortcut (CYMEK_DATA_SHORTCUTS A1/A2) is what was learned. This is a *contained* positive:
  it would demonstrate data→behavior causality at micro scale while capping the interpretation
  at shortcut level and prioritizing the E1 generator fixes.

## Why this experiment

It is the first causal data→capability measurement in the project's history: every prior
cognition claim is observational on V4 checkpoints or infrastructure certification. It requires
no new architecture, no scorer (primary metric is candidate-free), no external corpus, no sealed
fixtures, and ~7 GPU/CPU-hours total. It simultaneously exercises the highest-value engineering
fix (cognition data through the full production path) and discriminates H1/H2/H3, which demand
different next moves (scale up vs fix generators vs fix objective/curriculum).

## Independent variable

Exactly one: **query→answer pairing in the training documents** (intact vs ablated). Same
generator, formats, token budget, model, init, optimizer, schedule, packing, evaluation.

## Fixed variables

Model spec + init seed; tokenizer (frozen 24,576 artifact, hash-verified); optimizer (AdamW
0.9/0.95/1e-8, wd 0.1); schedule (`bounded_warmup_schedule`, peak 3e-4 — the schedule every
executed Cymek run uses; deviation from the canonical WSD is recorded); tokens/update 4,096;
30,000,000-token budget per arm; packing (`pack_documents`, bucket-512 cognition corpus);
manifest pipeline (dedup, cluster split, contamination scan vs eval prompts); evaluation
protocol and thresholds (declared here, before execution); training-text format
`context\nquery answer` (declared here — Cymek defines no cognition document renderer, so
Citadel declares one now, identical for both arms).

## Models/checkpoints

`MINI_SPEC` from `anra_v5/miniature_run.py` at cymek `26a61f6` (spec SHA recorded at run);
init seed 20260903; checkpoints content-addressed per arm; parameter/moment hashes at start and end.

## Data

- Arm A: `build_training_examples` (e0-train/0.2.0) across ~560 sequential seeds, 30M real
  tokens of rendered documents (context + query + answer).
- Arm B: identical pipeline; each document's answer replaced by the answer from a different
  example (fixed deterministic derangement; derangement table hash recorded).
- Both arms: manifest audit, dedup statistics, contamination scan results, and per-family token
  ledgers recorded in receipts. Known, declared limitation: the generator's serialization
  shortcut (A1/A2) is present in Arm A by design of the existing generator — H3 exists to
  contain exactly this.

## Controls

- Structure-ablated control arm (above).
- Untrained-model baseline on the identical eval battery.
- Three heuristic nulls + format prior over the identical instances.
- Implementation validation: the C1 driver reuses the certified one-update path verbatim
  (`cymek_receipts/one_update_cert.py` machinery); per-update receipts (loss, grad norms,
  ledger) are emitted mechanically and checked non-degenerate.

## Metrics

Primary first: held-out exact-match accuracy (candidate-free, answer-blind) with Wilson 95%
intervals, per mode and pooled, per arm. Secondary: dev-family transfer probe; training-loss
curves; realized mixture ledger; capability-per-token (accuracy / 30M tokens) per §26.

## Statistical treatment

Wilson 95% intervals on all rates; two-proportion comparison of arm A vs arm B (exact paired
design is unavailable across independent runs — unpaired exact test on instance-level outcomes
plus interval non-overlap as the descriptive gate); Holm correction across the three modes
before any per-mode claim.

## Success threshold (declared before execution)

Arm A pooled held-out exact-match LCB ≥ 0.10 AND arm A above its strongest heuristic null with
interval separation AND arm B pooled ≤ 0.05. Per-mode claims additionally require the mode-level
gate after Holm correction.

## Failure threshold (declared before execution)

Arm A pooled held-out exact-match ≤ 0.05 (near-null) after 30M tokens ⇒ micro-scale floor or
format failure; classify via training-set fit before interpreting (see possible outcomes).

## Confound checks

1. Token-budget equality enforced by the state machine's exact per-update ledger (mechanical).
2. No parameter/FLOP difference between arms (same spec, same init seed).
3. Eval answer-blindness: generation never sees candidates or gold; prompts are model-view only.
4. Train/eval instance disjointness: eval seeds disjoint from training seeds; template namespace
   disjointness asserted by the Cymek generator itself.
5. Dedup: manifest exact-cluster statistics reported; identical renderings across cycles rejected.
6. Memorization vs learning discrimination: held-out accuracy is compared against accuracy on a
   sample of *training* instances (declared: 500 resampled) — separating rule extraction from
   instance memorization.
7. Schedule deviation (bounded-warmup vs canonical WSD) is identical across arms — it cannot
   explain an arm difference, only cap external validity (recorded).

## Compute budget

≤ 10 hours wall total on the certified CPU path (measured 2,360 tok/s ⇒ ≈3.5 h/arm);
CUDA permissible only after a 5-update clip-gate stability pre-check passes (given the measured
CPU fp32 tolerance nondeterminism and the CUDA calibration of that gate). Storage: two
checkpoints + receipts; checkpoints are artifacts, not committed.

## Stop condition

Both arms run exactly once to 30,000,000 tokens; evaluation battery executes once; receipts are
written and the verdict recorded. No thresholds, arms, or metrics may change after any result is
seen; amendments require a superseding file.

## Possible outcomes

1. **H1 (structure causes capability).** Belief change: the production path turns cognition-
   structured data into query-conditioned behavior at micro scale; next step is E1-grade
   generators (shortcut-free, all 9 families) and a scale ladder. If replication seeds hold,
   this is the project's first DEMONSTRATED training-result.
2. **H3 (shortcut acquisition only).** Belief change: data→behavior causality holds but the
   current generator teaches serialization shortcuts; E1 generator hardening becomes the
   blocking scientific dependency, with a measured baseline to compare against.
3. **H2 (scale floor).** Belief change: capability emergence requires scale beyond the micro
   probe; the triquetra checkpoint-floor story extends to from-scratch training; next
   discriminator is a parameter/token ladder, not data work.
4. **Arm A high on training instances, null on held-out.** Belief change: the model memorizes
   instances but does not extract the rule — a generalization failure at fixed templates;
   template-diversity (C3) becomes the target variable.
5. **Both arms degenerate (loss non-decreasing, ledgers inconsistent).** IMPLEMENTATION_FAILURE;
   repair before any interpretation; the certified path is re-validated first.
