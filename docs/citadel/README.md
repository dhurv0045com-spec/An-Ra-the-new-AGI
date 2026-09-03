# CITADEL — Independent V5 Research Layer

Citadel is the controlled-experiment research branch of the An-Ra / MISSION AGI project.
Its job is not to build components. Its job is to determine, through falsifiable
experiments and reproducible evidence, **what actually improves An-Ra**.

Operating loop: **Build → Measure → Understand → Improve**. The goal is information, not activity.

## Central question

> What is currently preventing this small An-Ra Core from acquiring stronger transferable
> internal cognition per parameter and per training token?

Followed always by:

> What is the cheapest controlled experiment that distinguishes the leading explanations?

## Bootstrap facts (repository record)

| Fact | Value |
|---|---|
| Branch | `citadel` |
| Created from | `origin/esoes` tip `85f44b7b449f2ee39a0e80203a2d7df04614983b` |
| Merge base with esoes | `85f44b7b449f2ee39a0e80203a2d7df04614983b` |
| Worktree | `C:\Users\ankit\.zcode\workspace\default\An-Ra-citadel` (main clone stays on `cymek`) |
| Bootstrap date | 2026-09-03 |
| Ancestry rule | Citadel descends **only** from ESOES. Triquetra and Cymek are audit inputs, never merged. |

## Documents

| File | Purpose |
|---|---|
| [BRANCH_MAP.md](BRANCH_MAP.md) | What each branch is, tip SHAs, systems, conclusions, unresolved work. FACT vs CITADEL INTERPRETATION separated. |
| [EVIDENCE_LEDGER.md](EVIDENCE_LEDGER.md) | Every substantive existing claim, audited and verdict-scored. |
| [NEGATIVE_RESULTS.md](NEGATIVE_RESULTS.md) | Failures, false greens, and withdrawn claims — preserved, with repeat decisions. |
| [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) | Unresolved, experimentally actionable questions. |
| [BOTTLENECK_RANKING.md](BOTTLENECK_RANKING.md) | Candidate bottlenecks ranked by expected information gain / cost. |
| [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md) | Citadel's binding rules: verdicts, controls, compute ladder, sealed-eval discipline. |
| [experiments/C0/PLAN.md](experiments/C0/PLAN.md) | Preregistration of Citadel Experiment Zero. |

## Current position (post-audit summary)

- **No V5 model exists.** ESOES supplies contracts, a hardened cognitive benchmark, and local
  canaries; Cymek supplies production contracts. Nothing has ever been trained on the cognition data.
- **The measurement instrument for candidate selection is broken.** Every likelihood-based scoring
  policy failed a preregistered bias screen; `production_scoring_mode = null`. All learned
  cognition comparisons are blocked on this.
- **All real-model cognition evidence comes from weak V4 checkpoints** that Triquetra's calibrated
  readiness gate classifies as `INSUFFICIENT / NOT_IDENTIFIABLE / NOT_READY`. Triquetra is formally
  `WAITING_FOR_STRONGER_CHECKPOINT`.
- Citadel C0 therefore targets the scorer blocker first: it is the cheapest experiment that gates
  every subsequent measurement (see BOTTLENECK_RANKING).
