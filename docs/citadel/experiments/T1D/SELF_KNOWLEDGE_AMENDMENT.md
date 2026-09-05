# T1D — SELF-KNOWLEDGE AMENDMENT 002 (preregistered, implementation additive)

Status: **PREREGISTERED — NO RESULTS**. Amends T1D by ADDING Arm F and
self-knowledge probes; changes NO existing arm, budget, threshold, seed,
dataset tier, curriculum, or teacher ratio. Written before any T1D result
exists. Authorized by the operator's standing direction (2026-09-06): the
long-term An-Ra goal is a small system that learns like a child — little
data, little compute, fails and improves fast — and the operator's specific
hypothesis is that a learner which **knows what it is** (identity, body,
infrastructure, purpose, motivation, abilities, limits) will accept and use
self-knowledge from its training stream. Main-training-scale payoff is
expected later; this amendment tests the mechanism cheaply now.

## The operator's hypothesis, stated falsifiably

H_SELF_ACQ: a model trained with a small fraction of self-knowledge rows in
its stream acquires held-out self-knowledge question-answering above the
untrained baseline and above the strongest trivial null, while its
preregistered arithmetic learning (loss decrease) is preserved.

H_SELF_TRANSFER (secondary, explicitly budget-capped and DESCRIPTIVE ONLY):
Arm F's arithmetic tiers look like a 2M-token curriculum arm's arithmetic.
Because Arm B runs 8M tokens, B-vs-F arithmetic differences are confounded
by budget and support NO causal claim. The causal "self-knowledge accelerates
learning" question requires matched budgets and belongs to the main-training
stage; this amendment does not claim it.

## Arm F — SELF (added; everything else frozen)

```text
ARM F — SELF: MID 3.7M | answer CE | frozen curriculum schedule (same as B),
              but every 7th drawn row (row-count fraction ~1/7) is a
              self-knowledge training row | 2,000,000 cap tokens
```

- Same spec (MID), same init seed pattern, same optimizer/schedule/packing/
  calibration shape/time-box as the other arms.
- Self rows REPLACE curriculum draws (they do not extend the budget).
- The exact self-row stream, its fraction, and per-arm consumption are
  receipted (feeder ledger gains a `self:*` source key).
- Session ceiling: F adds ~10–15 min inside the existing <2 TPU-h budget
  (D and E already run at 2M).

## Self-knowledge data contract (`citadel_tpu/self_knowledge.py`)

- Rows use the frozen calculator row grammar: `<prompt> = <answer>`, all
  lowercase, alphabet-safe, answer <= MAX_ANSWER_TOKENS tokens, row <= 64
  chars, so production `split_prompt_target` / `answer_spans` / generation
  work UNCHANGED.
- Domains (the operator's list): identity, body, infrastructure, purpose,
  motivation, abilities, limits, mission.
- Deterministic `self_row(i)` (hash-streamed like tiered_data), train cursor
  semantics identical to teacher rows; a fixed held-out probe set
  (`SELF_PROBE_N` rows) uses DIFFERENT question forms from every training
  form (form-level disjointness asserted by construction table).
- Scoring: TEXT exact-match after casefold + whitespace collapse + trailing
  punctuation strip. The arithmetic integer normalizer is NOT used (it maps
  non-numeric text to a shared None); the self scorer is a separate pure
  function with Wilson intervals. Untrained baseline + `most_common_answer`
  null are computed on the identical probe rows.

## Evaluation additions (same session, no extra arms)

- Per-arm: self-probe accuracy (arms without self rows are the natural
  no-self controls on identical probes).
- Receipt blocks: `untrained_self`, `trained_self`, per-arm
  `diagnostics.self_knowledge` {accuracy, lcb, n, per-domain breakdown}.
- Machine rule added to cross-arm classification:
  `SELF_KNOWLEDGE_ACQUIRED`: arm F probe LCB >= untrained LCB + 0.10 AND
  > most-common-answer null + 0.10. Diagnostic for other arms: any arm above
  the same bar without self data would indicate probe leakage -> the probe
  disjointness gate fails the arm loudly instead of celebrating.
- TEST accounting unchanged: self probes are a DEV-tier diagnostic family;
  the preregistered one-shot TEST observations are untouched.

## What would change our mind

- F probe LCB clearly above untrained + null: self-knowledge is learnable
  from a small stream fraction by a 3.7M model — supports carrying
  self-knowledge into main-training curriculum design.
- F probe at/below untrained: self-knowledge as drafted is not learnable at
  this scale/fraction — the mechanism needs a different form (e.g., higher
  fraction, structured identity tokens, or longer exposure) before
  main-training invests in it.
- F arithmetic catastrophically below expectation: self rows interfere —
  recorded as a real cost of the identity curriculum.
- Leakage gate trips: the probe design is defective — no claim is drawn.
