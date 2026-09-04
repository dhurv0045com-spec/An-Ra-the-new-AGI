# AGENT.md — Citadel handover (machine-readable status for the operator's agent)

> Convention (binding): rewritten at the END of every Citadel work cycle,
> committed to the `citadel` branch ONLY, then `git push origin citadel`.
> Other branches are read-only audit inputs — never modified, never pushed.
> A CPU/CUDA run is NEVER a TPU result. No fabricated device results.
> Preregistration and results never share a commit. Download ceiling <10 GB;
> this cycle: 0 bytes.
>
> STANDING OPERATOR POLICY: prefer batched preregistered experiment suites
> over repeated round-trips. When multiple known hypotheses fit one compute
> session, test them together. Tiny experiments are automated internal
> gates/preflight only, never repeated operator-facing work. Bigger only when
> information value justifies it.

## STATUS

Citadel SHA: `fa3df6e` (local; T1C receipts + brief to be pushed this cycle)
Pinned Cymek runtime SHA: `298c91ac04f756f0833a7edcf63e73af3d5af688` (unchanged)

**T1C EXECUTED — full 4-arm session returned.** Operator ran
`notebooks/citadel_colab_t1c.ipynb` end to end at `7f81efc` (auto-scale halved
budgets 8M→4M at measured 3782 tok/s, as preregistered) and returned
`CITADEL_T1C_RESULTS.zip` (31 MB, incl. all 4 checkpoints). Receipts recorded
verbatim under `docs/citadel/tpu_receipts/t1c_session/` (binaries excluded);
every arm gate recomputed locally from raw numbers — all verdicts follow.
New brief: `docs/citadel/EXPERIMENTS_BRIEF.md` (all Citadel experiments +
results, for future reference).

T1C result: **all 4 arms FAIL, cross-arm INCONCLUSIVE** (no contrast rule
fired — correct, nothing moved anywhere). Headline: loss falls on every arm
(10.0→1.3–1.9) while exact-match stays 0/500 everywhere incl. train samples;
generations are valid integers with wrong digits (` 1110000`-style mode
collapse, 100% MAX_TOKENS stops) — format works, computation absent, and no
memorization either. Objective, data, and 2.3× scale all moved nothing at 4M
tokens. Prior hotfix history (buffer geometry + parser defects) preserved below.

T1C has now hit two deterministic implementation defects on the real Colab TPU
before any valid arm completed. Neither failure is a scientific T1C result.
Calibration remains reusable (shape 256×32, ~3.9k tok/s; preregistered auto-scale
halves budgets) and the 108.9 MB data manifest remains reusable.

### Operator failure #1 — evaluator buffer geometry

Observed: `prompt too long for fixed buffer: 'subtract 1353 from 1269 ='`.
Root cause: greedy generation reused training L=32 even though word-template
prompts need prompt + MAX_ANSWER_TOKENS writable headroom. Fixed by dedicated
static eval L=64, complete DEV/TEST geometry preflight, explicit answer-CE
length guards, and exact regression coverage.

### Operator failure #2 — legacy canonical-only parser

After the L=64 hotfix, T1C progressed further and failed in both arm A and B
with:

```text
malformed calculator row (no '='): '263 - 791 -> -528'
```

Root cause: `calculator_eval.heuristic_nulls()` consumes a deterministic sample
of rich TRAIN rows, but its legacy `parse_row()` only understood the original
T1 canonical `a op b = c` surface. T1C intentionally contains canonical,
compact, arrow, and word templates, so arrow/word rows reached a helper that had
never been upgraded to the richer corpus contract.

Hotfix `f23bbee` updates `calculator_eval.parse_row()` itself — not a notebook
monkeypatch — to support all frozen T1C forms while preserving the historical
`(a, op, b, target_text)` return contract:

```text
canonical:  12 + 9 = 21
compact:    12+9=21
arrow:      263 - 791 -> -528
words/add:  add 12 and 9 = 21
words/sub:  subtract 9 from 12 = 3
words/mul:  multiply 12 by 9 = 108
words/div:  divide 108 by 9 = 12
```

The evaluator selftest now exercises every form plus a mixed-template
`heuristic_nulls()` call, which is the exact helper path that failed in Colab.
Because `t1c_preflight` already runs `calculator_eval.selftest()`, this class of
mixed-template semantic mismatch is now a pre-arm gate. Cell D also reloads the
hotfixed evaluator/runner after Git refresh, preventing stale module-cache use.

No model architecture, optimizer, T1C arm definitions, data splits, objective
semantics, budgets, success gates, or scientific thresholds changed. T1C
preregistration remains applicable; both observed failures are implementation
repairs only.

## T0 / T1 (history)

```text
T0: PASS (unchanged)
T1: FAIL (loss-learned, exact-flat; historical result unchanged)
```

## T1B

```text
SUPERSEDED_BY_T1C (preserved, unexecuted)
```

## T1C

```text
EXECUTED 2026-09-04 — all arms FAIL, cross-arm INCONCLUSIVE (see brief)
```

Arm table: A MINI/whole/rich, B MINI/answer/rich, C MINI/answer/narrow,
D MID/answer/rich — 488 updates each, loss↓ everywhere, test exact 0/500
everywhere, reloads identical, strongest null 2.7% unbeaten. Diagnosis:
valid-integer mode collapse (not format failure), no memorization,
no movement from objective/data/scale at 4M tokens. Next design must answer
the brief's 3 open questions (floor vs teacher vs curriculum).

## DOWNLOADS

```text
ITEM | SOURCE/PURPOSE | BYTES | CUMULATIVE BYTES
(none)
TOTAL_DOWNLOADED_GB = 0.0
```

## QUESTIONS FOR OPERATOR

```text
NONE
```

## BIGGEST BLOCKER

No current blocker — T1C answered its session. The open problem is the next
experimental design (capacity/budget floor vs answer-teacher vs curriculum),
not infrastructure.

## NEXT ACTION

No operator run needed. Next Citadel work: design the follow-up that
discriminates floor vs teacher vs curriculum, preregistered before execution.

## Latest hotfix commits

```text
f23bbee fix(citadel): parse every T1C arithmetic template in evaluator
2a123f2 test(citadel): preflight all T1C eval buffer invariants
3ee6597 fix(citadel): give T1C generation safe fixed-buffer headroom
```
