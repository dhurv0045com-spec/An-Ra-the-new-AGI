# T1B — Preregistration: batched multi-scale calculator run (single session)

Status: **PREREGISTERED — NO RESULTS**. Branch `citadel`. New experiment (not
an amendment: new arms, new gate family, new questions). T1 (`FAIL`: loss
10.10→2.85, dev CE 10.08→2.91, exact 0/500 throughout) motivates it.
Prerequisites: T0 receipt PASS + T1 receipt present (context arm A0).
Cymek runtime pin: `298c91a` (unchanged; T1-relevant surface re-verified).

## Question

At fixed data, fixed init, and fixed code path, does held-out generation
exact-match move with training budget — or does only train accuracy rise
(memorization), or does nothing move (scale floor)?

## Why batched, why now

Sequential dribbles (one tiny run → peek → repeat) burn sessions and invite
TEST peeking. One session executes all arms with frozen per-arm contracts;
every arm's TEST observation is preregistered below and all arms are reported
win or lose. The T1 FAIL showed loss-learning without exact-match movement;
the scale response (learning vs memorization vs floor) is the highest-value
discriminator and it is cheapest measured in one session.

## Arms (exactly one variable: total budget)

Same MINI_SPEC, same init seed 20260904, same data order, same optimizer/LR,
same runtime. Batch 32 rows × L32 = 1024 capacity-tokens/update.

```text
A0    = T1 receipt as executed (200 updates, ~0.2M cap) — reference, no recompute
A1k   = ladder (1000,),  early_stop=False  (~1.0M cap)
A2.5k = ladder (2500,),  early_stop=False  (~2.6M cap)
A5k   = ladder (5000,),  early_stop=False  (~5.1M cap)
A10k  = ladder (10000,), early_stop=False  (~10.2M cap)
A20k  = ladder (20000,), early_stop=False  (~20.5M cap)
```

Total ≈ 39M new capacity tokens ≈ tens of MB text-equivalent in one session.
Estimate from measured 8710 tok/s: ≈75 min training + eval overhead; ceiling
<2 TPU-h. If the session dies mid-list, executed arms stand as reported (order
A1k→A20k; no arm may be skipped to reach a larger one).

## Repetition disclosure (binding)

UNIQUE_PHYSICAL_TOKENS = 4000 canary rows (fixed). SCHEDULED_CONSUMED_TOKENS
= arm budget. Max arm replays the corpus 160 epochs. This is deliberate: the
experiment tests whether more optimization on the same canary extracts the
rule or only memorizes. No claim of a larger physical corpus may be derived.

## Per-arm contract (identical for A1k–A20k)

Fresh init per arm; own untrained DEV+TEST baseline (re-measured, never
reused across sessions); DEV eval at endpoint (informational — no escalation,
budget is fixed); TEST generation eval EXACTLY ONCE; train-sample exact (200
rows, memorization lens); heuristic nulls recomputed; stop histogram +
5 samples + answer-CE untrained/trained; save/destroy/reload with prediction-
hash gate; full receipt (`TPU_SCALE_A{arm}.json`) with the T1B gate below.

## Gate per arm (same 5 rules as T1, own baseline)

Rules 1–5 of AMENDMENT_001 §A10 evaluated per arm. Multiplicity: per-arm PASS
claims form a family of 5 → Holm correction on the trained-vs-null comparison
(rule 2) before any single arm may be called PASS on that rule; rules 1,3–5
are descriptive gates reported as-is. Memorization flag
(train_sample_exact − test_exact ≥ 0.30) is informational: PASS + flag =
PASS-with-memorization-caveat, and triggers the corpus-diversity follow-up
rather than a scale-up.

## Possible outcomes

1. **Scaling:** test exact rises monotonically (or jumps) with budget while
   train≈test → rule extraction is budget-limited; next is a finer ladder +
   harder (held-out-range) transfer probe.
2. **Memorization:** train exact rises, test flat, flag trips → current
   templates teach instances, not rules; next is template-diversity work (C3
   direction), NOT more epochs.
3. **Floor:** neither moves (loss-only, as T1) → capacity/budget floor or
   objective-shape problem at this scale; next is the answer-loss-weighting
   vs scale discriminator, not blind scaling.
4. **Format break:** stop histogram dominated by NON_ALPHABET/empty with
   near-zero decodable answers → generation-format defect, not a learning
   result; fix format, re-preregister, do not scale.

## TEST accounting (explicit)

T1 observed TEST twice (untrained + trained final). T1B observes TEST exactly
twice per arm (own untrained baseline + own trained final) × 5 arms = 10 new
frozen observations. No TEST observation drives any decision (budgets fixed).
All 10 + T1's 2 are reported. Total TEST exposure after T1B: 12 observations.

## Checkpoints

Binary `.pt` exported (operator transfer) ONLY for the smallest-PASS arm and
the final executed arm (MINI_SPEC ≈ megabytes — cheap, preserves substrate
for diagnosis; the T1 lesson: ephemeral checkpoints destroy diagnosability).
SHA in every receipt; binaries never committed to git.

## Stop conditions

Per-arm infra failure (non-finite, no mutation, reload state mismatch,
overlap violation) → IMPLEMENTATION_FAILURE for that arm, 10-minute diagnose,
continue to next arm. ≥2 infra failures → abort session, report. Any single
arm exceeding 60 min wall → abort that arm, continue.

## Success for the session

All executed arms receipted with complete diagnostics and honest verdicts,
regardless of PASS/FAIL — information per TPU-minute, not a PASS.
