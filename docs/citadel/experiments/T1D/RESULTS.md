# T1D RESULTS — EXECUTED / ARCHIVED (authoritative record)

Status: **EXECUTED / ARCHIVED — DO NOT RERUN**. All six arms
SCIENTIFIC_FAIL; cross-arm **INCONCLUSIVE**. Evidence: the operator's
returned `CITADEL_T1D_RESULTS.zip` (19 members,
zip valid, every JSON
parses; per-member SHA-256 in RESULTS.json).

- Citadel SHA: `abec6bd2d503cff7669067a2cc120023c0006306`
- Cymek SHA: `28bf57a0d299a2c13a99fe0046616c00a1b8530c` (production authority)
- TPU environment: torch/XLA 2.9.0, calibration **(512,
  64)** @ ~6977.9 tok/s
- Budgets: A/B/C 8M, D/E 4M, F 2M cap tokens (auto-scale did not fire)

## Headline result

Answer-CE arithmetic learning does not lift off at 2–8M tokens for
3.7–7.4M models under ANY of the five variations. TEST exact 0–6.6%
everywhere, below the 22.5% copy-first-operand null; TRAIN exact 0–11%
(no memorization either); dev curves flat at every checkpoint. Verdict
INCONCLUSIVE per the frozen rules — and the suite ELIMINATES curriculum,
teacher, 2× scale, output masking, and self-knowledge as *sufficient*
for lift-off at these budgets.

## Per-arm record

| Arm | Variation | Loss | TEST exact t0–t4 | TRAIN t0–t2 | Reload | Status |
|---|---|---|---|---|---|---|
| A | flat control | 10.04→1.36 | 0.000/0.058/0.000/0.000/0.000 | 0.000/0.095/0.000 | True | SCIENTIFIC_FAIL |
| B | curriculum | 10.05→1.42 | 0.000/0.066/0.000/0.000/0.000 | 0.000/0.110/0.000 | True | SCIENTIFIC_FAIL |
| C | teacher | 10.04→0.77 | 0.000/0.064/0.000/0.000/0.000 | 0.000/0.095/0.000 | True | SCIENTIFIC_FAIL |
| D | SCALE2 | 10.04→1.52 | 0.000/0.062/0.000/0.000/0.000 | 0.000/0.105/0.000 | True | SCIENTIFIC_FAIL |
| E | masked softmax | 3.85→1.53 | 0.000/0.062/0.000/0.000/0.000 | 0.000/0.110/0.000 | True | SCIENTIFIC_FAIL |
| F | self-knowledge | 10.06→5.19 | 0.000/0.000/0.000/0.000/0.000 | 0.000/0.000/0.000 | True | SCIENTIFIC_FAIL |

## POSTMORTEM 1 — termination contract (major design limitation)

**15,000/15,000
generation records ended MAX_TOKENS (100%)**.
Code audit: training rows encode to literal characters only; the
eligible mask covers answer characters only; EOS_ID is never appended.
Generation stops on EOS/PAD/newline/non-alphabet else MAX_TOKENS.

> THE MODEL WAS NEVER SUPERVISED TO EMIT THE EOS TERMINATION TOKEN USED BY
> GENERATION. T1D exact-generation results cannot cleanly distinguish
> arithmetic-content failure from answer-termination failure.

T1D is NOT rescinded and NOT retroactively rescored — the limitation is
recorded and corrected in T1E (PLAN.md: EOS supervised,
MAX_GENERATION_STEPS = MAX_CONTENT_TOKENS + 1, TERMINATION_FAILURE vs
CONTENT_FAILURE split).

## POSTMORTEM 2 — content-only forensics (POST_HOC_NOT_PREREGISTERED)

From the stored per-arm samples (20/arm — small n): content exact at
target length ≈ 5% in every arm. The arithmetic characters themselves
are mostly wrong — this is NOT merely a stop failure. Both failures
coexist; T1E measures them separately.

## POSTMORTEM 3 — teacher primitives (the positive finding)

- Arm C held-out teacher microtask accuracy: **0.515
  (n=200)** — vs ~0 everywhere else.
- Classification: **PRIMITIVE_LEARNING_WITHOUT_COMPOSITIONAL_TRANSFER**
  (interpretation, not an AGI claim): primitive microtasks are learnable
  while full T2+ arithmetic composition stays near zero.
- TEACHER_DIVERSITY_LIMIT: unique pools of {'digadd': 12, 'digsub': 13, 'divmicro': 13, 'singlemul': 10} rows
  were placed 32618× each per kind —
  replay factors {'digadd': 2718.2, 'digsub': 2509.1, 'divmicro': 2509.1, 'singlemul': 3261.8}. T1E expands the pools;
  it does not simply repeat more.

## POSTMORTEM 4 — self-knowledge probe contract

- 57/96 probe targets
  exceed MAX_ANSWER_TOKENS=8 (examples: ['the operator', 'learn from my mistakes', 'the operator', 'try again']).
- **SELF_KNOWLEDGE_EVAL_CONTRACT_INVALID** — Arm F's negative is NOT a
  clean result. Official receipt unchanged (SCIENTIFIC_FAIL). Feasible-
  only scoring requires the full per-probe predictions (not serialized);
  T1E-family self-knowledge gets a corrected contract preregistered
  separately.

## POSTMORTEM 5 — budget confound

- B budget 8,000,000 vs D/E 4,000,000: D-vs-B changes
  model size AND budget; E-vs-B changes output space AND budget.
- Do NOT conclude 'scale does not help' or 'masking does not help'.
  T1E token-matches these contrasts on LOSS_BEARING_TOKENS.

## Data volume

- Available unique: 6,416,000 rows
  (~96.8 MB). Total placements across
  all arms: 1,869,821. Consumable unique
  fraction ≈ 35.1%.
- Verdict: **DATA_VOLUME_NOT_CURRENT_BOTTLENECK** — expansion must target information
  diversity, not unused bytes.

## Tier interpretation

- T0/T1: memorization/basic-fit probes (finite spaces; overlap
  possible/expected). T2/T3/T4: structural held-out surfaces. Even the
  T1 memorization probes failed (train ≤ 11%) — below-fit-floor regime.

## PRE50M

The smoke failed deterministically: the smoke state's token budget
funded `updates` but the resume-proof publishes `updates+1` (Cymek:
sold at "a completed run cannot advance"). **Cymek behavior is
correct; this was a Citadel PRE50M smoke bug** — fixed this cycle
(token_budget=(updates+1)*tokens_per_update, reserved-final-update
regression against the real Cymek contract incl. the negative control).
The session status propagation bug (PRE50M labeled PASS from file
existence) is also fixed. PRE50M certifies on the next TPU contact via
notebooks/citadel_colab_pre50m.ipynb (~minutes).

## REPLICATION — second independent full session (2026-09-06, later same day)

A complete fresh T1D rerun (Citadel `1cc36bf`, fresh Colab TPU, calibration
(512, 64) @ **7145 tok/s**) reproduced the entire result:

- All six arms again **SCIENTIFIC_FAIL**; cross-arm **INCONCLUSIVE**.
- Losses: A 10.04→1.36, B 10.05→1.42, C 10.04→**0.77**, D 10.04→1.52,
  E 3.85→1.53, F 10.06→5.19 — same shape as session 1.
- TEST exact: A 0/5.8/0/0/0, B 0/6.6/0/0/0, C 0/6.4/0/0/0, D 0/6.2/0/0/0,
  E 0/6.2/0/0/0, F 0/0/0/0/0 — all below the 22.5% null, all reloads
  identical. Arm F self-probe again 0.0.
- **PRE50M: PASS — `ready_for_50m_training: true`, zero blocking reasons.**
  The smoke's reserved-final-update resume worked exactly as designed
  (cumulative 10,240 → 12,288 tokens, moments preserved, continued update
  OK, writer fence rejected-as-required, token accounting consistent,
  checkpoint compat verified). Session PRE50M status agrees with the
  decision (status-propagation fix verified in production).
- Scale2 measured smoke rate: 414.4 tok/s (bf16, CPU-relative interval
  fields recorded); session throughput 7145 tok/s (MID, calibrated).
- Bundle: `CITADEL_T1D_RESULTS (1).zip`, sha256 prefix c3f6643bf8aa88ff,
  19 members, zip valid.

The T1D null is now a **confirmed replication across two independent
sessions** — the finding is stable, not a fluke of one run.

## Receipts

- Normalized machine record: RESULTS.json (member hashes, forensics).
- No checkpoint binaries committed; receipts only, per repo policy.
