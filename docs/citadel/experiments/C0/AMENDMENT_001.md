# AMENDMENT_001 — C0: coverage/selection-bias confound

Status: **AMENDMENT — PREREGISTERED, NO RESULTS** (written 2026-09-03, before any C0 model
execution). The original `PLAN.md` is preserved unmodified below this amendment; in any
conflict, this amendment governs. Thresholds in the original plan are unchanged unless
explicitly amended here.

## Reason for amendment

Review identified a confound in the preregistered `generation_match` policy semantics: the
original plan treats a no-match generation as an incorrect selection when computing surface-axis
bias. That conflates two different quantities:

1. **Generation coverage failure** — the policy produces text that matches no candidate at all
   (expected for an *untrained* P35, whose generations are near-garbage), and
2. **Candidate-selection bias** — among generations that DO match a candidate, systematic
   preference for axis-favored candidates (fewest-token, shortest-byte, marked-prefix, position).

With low coverage, the unconditional bias statistic is dominated by the coverage failure: if
nearly every generation matches nothing, every axis's unconditional selection rate collapses
toward 0 and B ≈ p0 trivially — the screen would report a spurious **PASS for the wrong reason**
(a false green of exactly the kind this project has twice produced and twice caught). The
original plan's failure semantics also mislabeled this outcome: it could have recorded
"H2 supported" (policy family biased) when the correct verdict is that the quantity of interest
was not identifiable at all.

## Amended definitions (govern all C0 analysis)

- **Valid selection:** a generation whose normalized text exactly matches exactly one candidate
  after the preregistered normalization (casefold, whitespace collapse). Multi-match generations
  follow the preregistered fixture-order tie-break and count as one valid selection.
- **Coverage** (per tokenizer, pooled over the 5 development seeds; also reported per seed):
  `coverage = valid selections / total generations`. No-match generations are excluded from every
  selection-rate denominator and never counted as selections.
- **Conditional selection bias (PRIMARY):** for each decoy axis a, `B_a = |Acc_a − p0_a|` computed
  **only over valid selections**, with `Acc_a` the rate at which the axis-favored candidate is
  selected among valid selections and `p0_a` the calibrated null for that axis.
- **Unconditional axis rates (INFORMATIONAL ONLY):** no-match-as-incorrect rates over all
  generations are still computed and reported, explicitly labeled as confounded by coverage and
  excluded from every verdict.

## Amended minimum-coverage requirement (preregistered before model execution)

**Minimum identifiable coverage: 0.10 per tokenizer, pooled over the 5 development seeds**
(n_eff ≥ 128 valid selections out of 1,280 generations per tokenizer).

Justification (declared before execution):

- PASS identification at the 0.10 gate: the two-sided 95% interval half-width on a rate near
  p0 = 0.25 is ≈ 1.96·√(0.25·0.75/n_eff) = 0.849/√n_eff. Requiring half-width ≤ 0.10 needs
  n_eff ≥ 72; the 0.10 floor (n_eff ≥ 128) provides margin and stable Wilson intervals.
- FAIL-detection power: with n_eff = 128, a true conditional bias of B ≥ 0.175 is detected with
  95% confidence (half-width ≤ 0.075). Biases between 0.10 and 0.175 may pass undetected at this
  floor — recorded as a stated power limit, not silently ignored.
- Per-tokenizer reporting: coverage is evaluated per tokenizer (16,384 / 24,576 / 32,768); a
  tokenizer below the floor is NOT_IDENTIFIABLE for that tokenizer even if others pass.

## Amended verdict logic

| Condition | Verdict |
|---|---|
| Coverage < 0.10 for a tokenizer | **NOT_IDENTIFIABLE** (for that tokenizer) — no claim about H1 or H2 is drawn |
| All tokenizers NOT_IDENTIFIABLE | Experiment verdict NOT_IDENTIFIABLE; the untrained-weight substrate cannot support the screen; next action is a substrate with generation capability, not a policy conclusion |
| Coverage ≥ 0.10 and every axis B_a ≤ 0.10 (point estimate) with Wilson intervals inside the gate | **PASS (H1 supported)** for that tokenizer |
| Coverage ≥ 0.10 and any axis B_a > 0.10 with Wilson LCB of B_a > 0.10 | **FAIL (H2 supported)** for that tokenizer |
| Coverage ≥ 0.10, mixed across tokenizers/axes | MIXED; scope recorded; no promotion |

The five likelihood negative controls and the constructed-oracle positive control are unaffected
by this amendment (they do not generate); their reproduce-the-recorded-failure and power criteria
stand unchanged. The screen-validity and sensitivity checks continue to gate every verdict.

## Is C0 still the highest-value next experiment?

**No — execution deferred; preregistration preserved.** The Cymek delta audit (2026-09-03,
`26a61f6`) changed the picture:

1. The production training chain is now executed and mechanically certified at bounded scale
   (real updates on real data, CUDA canaries), so the "instrument everything before any training"
   premise is weakened: cheap *candidate-free* capability metrics (generation exact-match, which
   need no candidate scorer) can measure the first real learning experiments.
2. The first broken edge moved: the binding constraint is that **no cognition data has ever
   flowed through the production path** (0 cognition tokens in every committed receipt; no
   generator wiring; mixture unenforced in the consumed path). That is an engineering/data
   bottleneck, not a measurement one.
3. The preregistered C1 uses candidate-free generation metrics as its primary measure, so the
   amended C0 is off C1's critical path.

C0 remains preregistered, amended, and queued: it becomes the gate for any **candidate-based
selection** claim (the assisted-scoring metric family, any E3-style selection comparison), and it
is still the cheapest way to establish whether *any* answer-blind selection scorer can be
sanctioned. It is simply no longer the first experiment Citadel runs.
