# ARK-020 — DESIGN REASONING AND RED TEAM (written before execution)

## Why four skills shaped like this

The mission requires structurally distinct, independently evaluable skills that a ~21M
model can actually form under 12-slot doses within ~2000 updates, using the proven
single-next-token machinery (V3/V4 lineage). Alternatives considered and rejected:

- **Multi-token answers (e.g., two-digit arithmetic)**: breaks the final-position CE
  machinery, adds segmentation confounds, and single-digit addition saturates (45 cases).
  Rejected for this campaign.
- **Two-hop composition**: with single-token words, 12 tokens yield only 6 chains — far
  too few factsets for a 400-train split with clean holdouts. Rejected.
- **Four binding copies with different tokens**: exactly the "fake diversity" the mission
  forbids. Rejected.

Adopted set: A and B are the *known-profile* binding pair (direct continuity with V4,
whose interference behavior is already characterized); **C (successor-cycle rule with
never-trained sealed keys) is genuinely structural novelty** — no context list, requires
induction; **D (value→key inverse retrieval) is a different retrieval computation**
(value-indexed lookup vs key-indexed). If the Guardian protects across the A→B transition
but fails across B→C or C→D, the registry-vs-family structure of the failure is itself
the finding.

## Why the predictive signal is what it is

Candidate signals were ranked by (computable at CONTROL cadence, mechanistically
motivated by R2's "constrain capability-relevant directions" picture, preregistrable
without tuning):

1. **robust-min margin < 0.95** (V4 already used this as EARLY_MARGIN_WARNING — it is
   the simplest "losing headroom" signal and it is one threshold, frozen).
2. **degradation_rate ≤ −0.03 per 100 updates over last 3 CONTROL observations** (slope:
   catching fast erosion before the absolute threshold crosses).
3. Rejected for now: gradient-conflict online estimates (expensive at 25-update cadence),
   hidden-state drift probes (needs a frozen probe set + extra forward passes; deferred
   to the mechanistic diagnostics at 100-update cadence), Fisher proxies (diagonal
   approximations are weak at this scale).

The comparison REACTIVE vs PREDICTIVE is the point: if PREDICTIVE prevents failures
that REACTIVE only recovers from, at similar cost, prevention is real. If both behave
the same, the warning signal adds nothing — that is a reportable negative.

## Replay allocation: why risk-based with max 2 slots

Static 1/32 spends ~5000 slots per arm regardless of need. The efficiency question is
whether the Guardian can match protection with fewer. Two slots/update cap (1/16 of
batch) keeps worst-case Guardian cost equal to the V4 primary replay dose and guarantees
real-text slots ≥ 18 never collapses (fail-closed in `mixed_update`). Risk order = lowest
robust-min first with deterministic id tiebreak — simple, measurable, and it makes the
"protection goes where needed" claim falsifiable from logged allocation traces.

## Red team (self-attack, before outcomes)

- **Task leakage**: C's sealed keys are never trained, never in control sets; the cycle
  is a single fixed permutation; distractor mode uses only trained keys. D reuses the
  factset 5-way split machinery with its own seed. A/B use V4's exact 5-way splits.
  Checked by test (`test_split_integrity`).
- **Replay stealing exposure**: replay replaces real-text slots, not task slots; task
  slots per phase are constant across arms (12 for C/D; selected dose for B). Every arm
  gets identical task exposure by construction; logged in counters.
- **Compute confound**: replay updates run extra forward rows; cap arm runs shadow
  backups. The PLASTIC_HIGH baseline therefore has the *least* compute per update. This
  favors PLASTIC_HIGH on speed and hurts it on stability — conservative in the right
  direction (if Guardian matches plastic acquisition speed despite fewer fresh rows,
  the plasticity claim is stronger). Science NLL per set guards the substrate.
- **Controller reading SEALED**: structurally impossible in `ark020_core` — controller
  functions accept only CONTROL metric dicts; tests pin the signatures.
- **Evaluation-noise false recovery**: de-escalation requires a 4-observation healthy
  streak (same as V4); recovery-within-400 uses sustained healthy streaks, not single
  evals. Formal recovery is distinct from PREVENTION in `decide`.
- **Task-order confound**: phase order is fixed B→C→D by design (skills accumulate);
  the order *seeds* vary presentation streams, not skill order. Skill-order effects are
  therefore out of scope and stated as a claim boundary.
- **Model-init confound**: all arms of a matched set restore from the same parent
  snapshot with byte-identical optimizer/scaler state (V4 `restore`), identity-checked.
- **Horizon bias**: fixed phase lengths, no outcome-based early stopping anywhere;
  formation gates use streaks with frozen deadlines (B keeps V4's pilot deadline logic
  only in the dose stage).
- **Duplicate telemetry**: `dedupe_trajectory` applied at completion (V3.1 lesson).
- **The 48-token gate**: if fewer than 48 eligible single-token words exist, the runner
  fails at the entry gate with an explicit error (fail-closed) — no silent fallback to
  overlapping token sets. Overlap between skill vocabularies would confound interference
  attribution.
- **Parent availability**: parents are V4's Drive cache when identity-matched; otherwise
  rebuilt with the identical V4 procedure. Either path is receipted.

## Compute honesty

Per matched set per arm: 5000 continuation updates + batteries. Full campaign: 4 sets ×
7 arms × 5000 = 140,000 updates plus parents/pilots/dose stage. This is ~3× V4. The
runner is exact-resumable; runtime calibration estimates sessions from measured
update/eval seconds and may not change the protocol. If calibration says the campaign is
too large for practical operator time, the frozen fallback is the same campaign at 2
matched sets (2 parents × 1 order seed) declared in a **prospective addendum before
execution** — never a silent weakening after outcomes.
