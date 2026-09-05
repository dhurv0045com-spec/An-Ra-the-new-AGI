# UNIFIED EVIDENCE MAP (Arkenstone)

Rule: execution artifacts beat prose. Each entry carries source branch, evidence
type, and novelty class for Arkenstone.

## DEMONSTRATED (executed, receipts trustworthy)

| Claim | Branch / evidence | Notes for Arkenstone |
|---|---|---|
| V5 stack trains on TPU end-to-end | citadel T0, `docs/citadel/tpu_receipts/`, MINI_SPEC 1.6M, loss 10.12 finite, checkpoint reload identical | plumbing certified; T1D runtime-pinned to cymek `28bf57a` |
| Loss falls but exact-match stays 0 across objectives | citadel T1 (whole-row CE, 200 upd, loss 10.10→2.85, exact 0/500) and T1C arms A-D (loss →1.3-1.9, test exact 0/500 all) | the central anomaly; NOT a format failure (valid integers generated) |
| 2.3x scale alone moved nothing | citadel T1C arm D (MID 3.7M vs MINI 1.6M, both exact 0/500 at 4M tokens) | naive scaling below ~4M tokens is not the lever |
| Train exact ≈ 0 even on seen rows | citadel T1C (all arms; best: narrow arm C 6/500) | the model fits loss statistics, not instances — repetition dose unmeasured |
| Strongest cheap heuristic null = 2.7% | citadel T1C (copy-first-operand) | tasks are not trivially shortcut at the answer level |
| Certified production training path (CPU+CUDA), exact resume, mutation certification, cursor-authoritative stream | cymek receipts under `artifacts/cymek/` (V5A/P35 canaries, miniature) | engineering substrate Arkenstone inherits |
| V5-A exact 250,216,960-param center constructs + trains (bounded) on local CUDA | cymek `artifacts/cymek/v5a_bounded_canary.json` | available if a mechanism earns scale |
| Binding generator FAILED cheap-heuristic qualification (preserved) | cymek generator qualification receipt (fb989ae) | capability data can be shortcut-solvable; qualification machinery exists |

## STRONG EVIDENCE

| Claim | Source | Notes |
|---|---|---|
| T1C cross-arm verdict INCONCLUSIVE is honest | citadel T1C: no contrast rule fired because nothing moved anywhere | arms were not compared on a moving target |
| Query-swap challenger infrastructure exists, fail-closed, never default | cymek `v5_objectives/query_swap.py`, training spec λ=0 | available as an arm, untested at any scale |

## TENTATIVE

| Claim | Source | Notes |
|---|---|---|
| "Loss-learning without rule learning" is a capacity/budget floor | citadel open questions (prose) | explicitly unproven; T1D designed to discriminate |

## FAILED (executed, negative)

| Attempt | Evidence | What it rules out |
|---|---|---|
| Whole-row CE vs answer-only CE | citadel T1 vs T1C A/B: both exact 0 | objective masking choice is not the lever at 4M tokens |
| Rich (6.5M) vs narrow (4k) corpus | citadel T1C A/B vs C: narrow raised train exact 0→6/500 only | data narrowness alone does not lift off |
| 1.6M → 3.7M params | citadel T1C D vs A/B | modest scale alone does not lift off |

## NOT TESTED (implemented/planned, never executed)

- **T1D** (citadel, PREREGISTERED — NO RESULTS): floor vs curriculum vs teacher vs representation on a tiered arithmetic ladder, 5 arms, 3.7M/7.4M models, 8M tokens, XLA-pinned. THE decisive designed-but-unexecuted experiment.
- **PRE50M** systems certification + NEXT_50M_DECISION gate (citadel, staged "final pre-run handover", unexecuted).
- Query-swap λ ∈ {0.05, 0.15} matched-compute arms (esoes/cymek plans).
- P35-A control/treatment cognition-mixture experiment (cymek, dataset pair frozen, NOT authorized).

## MISSING (needed, does not exist anywhere)

- Any measured **lift-off threshold** (dose of exposures/steps/tokens at which train exact first rises from 0) on ANY task.
- **Per-position digit accuracy** decomposition of the exact-match failures (citadel recorded only exact + loss).
- Tokenizer/vocabulary contrast for symbolic learning (H_REPR was a diagnostic arm in T1D, never run; vocab size never varied).
- Mechanistic probes on a model that CAN do the task (nothing to probe yet — no branch has a model that exhibits the capability).

## BLOCKED

- P35-A execution: requires certified external data corpus + XLA target (cymek readiness gates).
- 5B-token natural corpus: external blocker.
- TPU/XLA runs from this environment: no TPU access; T1D/PRE50M await their designed runtime.

## NOVELTY ANCHORS (what Arkenstone can add that no branch contains)

1. Measured lift-off dose-response on trivial→easy symbolic tasks at micro scale (citadel always ran mixed corpora at ≥4M tokens; never the simplest case).
2. Per-position accuracy decomposition of exact-match failure.
3. Vocabulary/representation contrast at matched decoder scale.
4. Independent red-team of cymek's binding-generator v2 qualification.
