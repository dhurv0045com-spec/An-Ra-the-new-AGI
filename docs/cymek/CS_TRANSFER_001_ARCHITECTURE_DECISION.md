# CS-TRANSFER-001 architecture decision

Status: **ACTIVE ARCHITECTURE CONSTRAINT**

Evidence source: completed CS-TRANSFER-001 Amendment-2 result, scientific executable `f8582808b6e2be0753cb2689d9d6d4aeb4d57aeb`.

## Decision

Keep the working production/Core vocabulary at **24,576**. Do not promote a physical V=4,096 tied embedding/output matrix based on CS-TRANSFER-001.

The experiment's official verdict is `PARTIAL_OR_INTERACTION`:

- mean paired development identity formation-AUC gap (`4096 - 24576`): `-0.034765625`
- mean paired development identity endpoint gap: `-0.06875`
- mean sealed identity endpoint gap: `-0.1395833333333333`
- sealed pair gaps: `[-0.1833333333333333, +0.10833333333333331, -0.42499999999999993, -0.058333333333333334]`
- 3 of 4 sealed matched pairs favor V=24,576, but one pair reverses, so the result is interaction/seed-sensitive rather than a clean reverse-effect proof.

## Architecture implication

The identity/copy bottleneck is not robustly explained by output class count alone. Completed R1C already showed that inactive-softmax/denominator competition alone was insufficient; CS-TRANSFER-001 now shows that physically shrinking the tied class space itself is also not a robust remedy.

Therefore:

- retain Candidate A / full-softmax V=24,576 as the conservative working architecture;
- do not change the production tokenizer or vocabulary from this evidence;
- do not revive masked-output Candidate B as a promotion path;
- do not perform a broad Core redesign around the vocabulary hypothesis;
- do not authorize PRE500M, 250M, or 500M scale from this result.

## Next architecture gate

Run one narrow causal mechanism dissection that separates:

1. tied-row geometry / presence of extra tied rows;
2. denominator competition; and
3. optimizer / weight-decay effects on those rows.

The objective is to determine whether the residual behavior comes from an interaction in optimization/parameterization rather than from vocabulary size itself.

Official next action from the final artifact:

`run a narrow mechanism dissection separating tied-row geometry from denominator competition/weight-decay effects; do not change production vocab`

## Provenance

- Raw Drive result: `MyDrive/CYMEK/CS_TRANSFER_001_A2/receipts/FINAL_RESULT.json`
- Uploaded raw artifact SHA256: `37e4bf741f9e55fc942fe4ade62d2d1c9619e4992e4a42ab507730809c12b150`
- Task-1 evidence record: `docs/research/CS_TRANSFER_001_FINAL_EVIDENCE.md` on `research/evidence-consolidation-2026-09-13`
- Claim ceiling: controlled development-scale causal evidence only; no production tokenizer/corpus/scale/cognition/AGI authorization.
