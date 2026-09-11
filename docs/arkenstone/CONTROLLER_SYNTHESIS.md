# TWO CONTROLLERS — cross-branch synthesis (mission §13)

**Question:** is the Guardian alone enough for continual cognition?

## The evidence split

**FORMATION** (can a capability come to exist at all?) is governed by representation and
objective geometry. Cymek's CYR-GPU-011 → 012-R1 → 013-R1B line showed that the *declared
tied vocabulary/class-space size* can decide whether held-out structural capability forms
— with identical active tokens and identical data, V24576 sat at ~0% held-out while
intermediate class spaces reached 50–100% in matched seeds, seed-sensitively. No Guardian
intervention can protect a capability that never formed; ARK-019 V3.1's formation
bottleneck demonstrated exactly this failure mode in a Guardian context.

**PRESERVATION** (can an existing capability survive further learning?) is governed by
task-support signal and update geometry. The Arkenstone laws (R2: cap OR sparse replay,
each independently sufficient; V3.1: Guardians reconstructed SKILL_A at 0.963 vs 0.005
unprotected) are all preservation-side results.

These are different control problems with different actuators:

| | Formation Controller | Preservation Guardian |
|---|---|---|
| acts on | representation/output geometry, objective, curriculum | replay allocation, update caps, phase-matched LR |
| measured by | formation rate/time-to-qualification | retention, duty, recovery time, prevention rate |
| evidence | Cymek R1/R1B/R1C (seed-sensitive, mechanism open — R1C pending) | R2 + V3.1 + (pending) ARK-020 |

## What ARK-020 contributes to the synthesis

ARK-020 does not build a Formation Controller — that would outrun the evidence. It does
three things the synthesis needs:

1. **Formation gates as first-class campaign stages.** The plastic-reference formation
   gate per phase (≥3/4 sets) is a mini formation-controller measurement: if C or D fails
   to form under identical exposure across arms, the campaign reports
   `INCONCLUSIVE_FORMATION_INSTABILITY` rather than blaming protection. This keeps
   formation failure from being misattributed to preservation machinery — the exact
   V3.1 lesson, now structural.
2. **Cross-family preservation.** Skills A/B (binding family), C (rule induction), D
   (inverse retrieval) let us measure whether preservation transfers across capability
   *families* — the property a combined system needs.
3. **Plasticity accounting.** Per-phase confirmation speed vs PLASTIC_HIGH bounds how
   much protection costs formation of the *next* capability (the ARK-018 heavy-Birth
   plasticity-slowdown risk, now measured per arm).

## Integration proposal (only where evidence supports)

- **Now (SUPPORTED):** keep formation and preservation as separate measured stages with
  explicit gates between them — formation gates precede protection responsibility for a
  capability (ARK-020's registry does exactly this: a capability enters the Guardian's
  protection set only at qualification).
- **After R1C + ARK-020 (SPECULATIVE):** a joint scheduler that treats class-space/
  representation choice as a formation actuator and replay/cap as preservation actuators.
  No implementation is justified until both evidence lines exist on the same substrate.
- **Not proposed:** a unified "cognition controller" architecture. Two experiments do not
  authorize an architecture.

## Bottom line

Guardian alone is **not** enough — it presupposes formation. Formation alone is **not**
enough — R2 shows acquired structure is actively destroyed by naive continuation. A
credible continual-cognition system needs both, and the honest current state is: one
preservation controller candidate undergoing generalization testing (ARK-020), one
formation mechanism under causal dissection (CYR-GPU-014-R1C), and a measured, preregistered
interface between them (this document + ARK-020's gates).
