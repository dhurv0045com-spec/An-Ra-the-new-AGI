# T1 AMENDMENT_001 — executable contract (supersedes vague wording, not the question)

Status: **PREREGISTERED — NO RESULTS**. Amends `PLAN.md`, which is preserved
unchanged. The question, hypotheses (H1/H2/H3), MINI_SPEC model, CE objective,
and single-device scope are unchanged. No T1 results exist anywhere; no
threshold was tuned on data. Written on `citadel` before any T1 execution.

## A1. Cymek pin update

```text
OLD CYMEK PIN (PLAN.md line 5):
4abeaeb (origin/cymek at time of writing)

ACTUAL T0-CERTIFIED CYMEK RUNTIME:
298c91ac04f756f0833a7edcf63e73af3d5af688 (= origin/cymek HEAD, verified)

REASON FOR AMENDMENT:
runtime/bootstrap development moved to a newer audited Cymek runtime before
T1 execution; no T1 results exist yet.
```

`origin/cymek` has not moved since T0 (still `298c91a`; T1-relevant surface
re-verified identical). T1 exercises the exact certified runtime: same
`MINI_SPEC`, same model construction, same CE objective, same optimizer
semantics, same checkpoint path as the passing T0 receipt.

## A2. T0 applicability classification

T0 (`TPU_ONE_UPDATE.json`, PASS) REMAINS APPLICABLE. Since the T0 run
(`citadel_sha 525a357`), no T0-critical semantics changed: `xla_backend`
optimizer-stepping/mark-step/rendezvous behavior identical on the executed
path (later edits were additive helpers + fail-closed guards never triggered
on a present API); `one_update.py` untouched; `environment.py` gains only an
additive `pjrt_device_env` provenance field (§28); model/objective/optimizer/
checkpoint code paths byte-identical via the pinned runtime. No T0
re-certification required before T1.

## A3. Task definition (frozen)

Generator: `calculator-canary/1.1` (`citadel_tpu/calculator_data.py`).
For each row (e.g. `3 + 4 = 7`):

```text
PROMPT = text before "=" plus "="   →  "3 + 4 ="
TARGET = text after "=" stripped     →  "7"
```

Whitespace convention: single spaces as rendered by the generator; the split
is `row.rsplit("=", 1)`. Frozen; no reformatting after results.

## A4. Answer normalization (frozen, strict integer equality)

`normalize_answer(s)`: strip leading/trailing whitespace; cut at the first
newline; the remainder must match `^-?\d+$` exactly — else INCORRECT (no
commentary, no units, no second number allowed). Compare by INTEGER equality:
`int(pred) == int(target)`. Rationale recorded: leading zeros (`007` vs `7`)
accepted, everything else rejected; a deliberately documented choice, not
fuzziness. `normalize_answer` is deterministic and unit-tested.

## A5. Generation protocol (frozen)

Greedy decoding only. No beam, no sampling, no temperature, no post-hoc
decoding tricks. Fixed static shapes every step: `[B=8, L=32]`, same dtype,
same model, same mask shape; generated tokens written into the fixed buffer
via index scatter (one compiled graph). Partial final batch: pad rows run
through the same compute with `valid=False` and are EXCLUDED from the metric
(deterministic). Per-row stop (first hit wins):

```text
PAD id 0 / EOS id 3 / any id outside DECODABLE → stop (reasons PAD/EOS/NON_ALPHABET)
newline id 12 → stop, answer is text so far (reason NEWLINE)
MAX_ANSWER_TOKENS = 8 → stop (reason TRUNCATED; counted INCORRECT)
```

8 covers all valid answers in the frozen ranges (longest: 5 digits + sign).
Decodable ids: space 34, `+` 45, `-` 47, `*` 44, `/` 49, `=` 63, digits 50–59,
newline 12. Each evaluated row records: prediction, target, correct,
stop_reason, generated_token_count.

## A6. Encoding (frozen, audited)

`ENCODING_VERSION = "char-byte-offset/1.0"`: `id = (ord(c) % 250) + 2`.
Calculator alphabet `0-9 + - * / = space \n` maps to ids ≥ 12 — provably no
collision with PAD 0 / UNK 1 / BOS 2 / EOS 3 (minimum content ord is 10 →
id 12). `encode`/`decode` round-trip is mechanically proven for the full
alphabet by unit test. Non-alphabet ids emitted by the model stop generation
(A5); they are never silently mapped.

## A7. Objective semantics (recorded, deliberate)

Training = whole-row autoregressive CE over `context+query+answer` rows.
PAD targets excluded from the loss (verified: `targets != pad_id` mask);
prompt tokens ARE supervised (deliberate canary choice — the capability
metric in A8 still evaluates prompt→answer generation, so reconstruction and
answering are not confused). BOS-exclusion never triggers (no BOS prepended;
content ids never equal 2). Per-batch receipt records real vs padding vs
supervised token counts (first batch audited).

## A8. Data splits and receipt (frozen)

TRAIN 4000 (seed 71001, operands 0–49) / DEV 500 (71002, 50–79) / TEST 500
(71003, 80–119); ranges structurally disjoint per operand. Machine-verified
before training: exact overlap TRAIN∩DEV/TRAIN∩TEST/DEV∩TEST = 0;
commutative-key overlap across splits = 0; generalization slices
(50 commuted 80–89 pairs + 100 range 120–199 rows) drop exact TEST duplicates
(count recorded). Data receipt records: generator version + code hash, counts,
seeds, ranges, op distribution, split hashes, all overlap counts, encoding
version.

## A9. Baselines and nulls (frozen)

Untrained exact-match on the identical frozen TEST set (measured once, with
Wilson interval — never inferred). Four mechanical heuristic nulls on the
same TEST set: always-`0`, copy-first-operand, copy-second-operand,
most-common-training-answer. `STRONGEST_HEURISTIC_NULL` = max of the four
(computed from data only, before any trained result is inspected).

## A10. Numeric success gate (frozen; justifications inline)

`T1 PASS` requires ALL (machine-evaluated from the receipt):

```text
1. trained_acc > untrained_acc AND trained_LCB > untrained_UCB
   (non-overlapping Wilson 95% intervals: signal beyond noise)
2. trained_LCB > strongest_heuristic_acc
   (beats every mechanical shortcut, not just the untrained net)
3. trained_acc - untrained_acc >= 0.10
   (n=500 → Wilson half-width ≤ ~0.045; 0.10 exceeds 2× sampling noise)
4. final_loss < first_loss AND trained_test_CE < untrained_test_CE
   (optimization moved and generalized by the loss lens too)
5. pre_reload_prediction_sha256 == post_reload_prediction_sha256
   (capability survives save/destroy/reload bit-identically)
```

`FAIL` = training executed but any rule fails (classify H2 vs H3 by
train-fit-vs-held-out pattern). `IMPLEMENTATION_FAILURE` = plumbing broke
(non-finite loss/grads, no mutation, reload mismatch of state, overlap
violation, budget overrun) — never a scientific verdict.

## A11. Budget ladder (frozen; TEST protected)

Cumulative update rungs: `[5, 20, 100, 200]` (batch 32 rows × L32 → max
204,800 tokens; ceiling < 2 TPU-h, expected minutes). TRAIN supplies updates;
DEV drives every escalation; TEST is observed exactly twice total (untrained
baseline once + trained final once) and never drives a decision:

```text
R5:   plumbing only (finite loss, params changed). Always continue (cheap).
R20:  DEV exact-match + DEV CE. Continue iff dev_CE_20 < dev_CE_untrained.
      Else STOP → FAIL (no learning signal), TEST trained-eval still runs once
      at endpoint for an honest null record.
R100: Continue iff dev_exact_100 > dev_exact_20 OR dev_CE_100 < dev_CE_20 - 1e-4.
      Else endpoint = R100.
R200: final endpoint.
```

No threshold, rung, or metric changes after any result is seen.

## A12. Reload gate (frozen)

Save → record SHA → destroy/recreate model → reload → rerun the identical
frozen TEST generation set. Require identical ordered prediction vectors
(`pre_reload_prediction_sha256 == post_reload_prediction_sha256`); metric
equality alone is insufficient.

## A13. Interpretation (binding)

PASS means: the tiny Cymek checkpoint learned this held-out calculator canary
under this run (learning-system certification). It does NOT mean general
arithmetic reasoning. No AGI-progress language in any receipt or report.
