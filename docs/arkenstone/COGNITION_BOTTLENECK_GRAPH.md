# COGNITION BOTTLENECK GRAPH (Arkenstone)

DATA → REPRESENTATION → OBJECTIVE → OPTIMIZATION → INTERNAL STATE →
QUERY CONDITIONING → COMPUTATION → MEMORY → ANSWER GENERATION →
GENERALIZATION → TRANSFER → EVALUATION → NOVELTY VERIFICATION

## Node status and the single sharpest unknown

| Node | Evidence | Failure modes consistent with T1/T1C | Cheapest distinguishing experiment |
|---|---|---|---|
| DATA | mixed corpora 6.5M/4k rows tried | dilution of trivial tiers; repetition dose uncontrolled | dose-response on ONE trivial family (ARK-001) |
| REPRESENTATION | vocab 24576 assumed everywhere; never varied | dead-vocab embedding wastes capacity; digit tokenization awkward | compact vs byte-level vocab at matched decoder (ARK-001) |
| OBJECTIVE | whole-row vs answer-only tried | answer-only CE still distributional; per-position marginals may suffice for loss | per-position accuracy decomposition (ARK-001) |
| OPTIMIZATION | AdamW standard; loss always falls | optimization reaches marginal mode, not instances (train exact 0) | measure lift-off dose on simplest task (ARK-001) |
| INTERNAL STATE | unmeasured (no model exhibits capability) | state never forms | only probe-able after lift-off exists anywhere |
| QUERY CONDITIONING | esoes probes designed | never exercised on a capable model | after lift-off |
| COMPUTATION | single pass | iterative depth untested | after lift-off |
| ANSWER GENERATION | valid integers, wrong digits | mode-digit repetition observed in T1C generations | per-position decomposition (ARK-001) |
| GENERALIZATION/TRANSFER | never reached (nothing to transfer) | — | after lift-off |
| EVALUATION | exact-match + Wilson; heuristics nulled | per-position detail missing | ARK-001 metric set |
| NOVELTY VERIFICATION | this register | — | continuous |

## Ranked bottlenecks (expected information gain / cost)

1. **Lift-off existence + dose at micro scale** (ARK-001): if even single-digit
   arithmetic never lifts off, curriculum/teacher/scale arms are premature — the
   pathology is upstream (optimization/representation). If it lifts off in
   minutes, the threshold is measured and every downstream arm gains a target.
   Cost: minutes of CPU. Information: discriminates H1/H2/H9 vs H3/H4/H6 as the
   NEXT bottleneck group.
2. **Per-position decomposition** of failures (rides on ARK-001 arms): distinguishes
   "distributional marginal fit" from "instance fit".
3. **Vocabulary contrast** (rides on ARK-001): H_REPR without a TPU session.
4. (post-lift-off) query conditioning, composition, transfer.

## Hypothesis groups mapped to ARK-001 outcomes

- H1 capacity floor / H2 budget floor / H9 optimization pathology → predict
  lift-off FAILURE even on single-digit at micro scale (or absurd dose).
- H3 curriculum / H4 teacher / H6 objective → predict single-digit lifts off
  easily with flat CE (so the multi-digit boundary is where these act);
  T1D's B/C arms then target the measured boundary.
- H5 representation → predicts compact-vocab arm lifts off at far lower dose
  than byte-level arm at matched decoder.
