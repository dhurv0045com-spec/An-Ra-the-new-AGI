# CS-TRANSFER-001 final evidence

Status: **COMPLETE**

This note records the final post-sealed result of CS-TRANSFER-001 Amendment 2. The raw campaign artifact remains in Google Drive; this branch stores the compact evidence/decision record.

## Provenance

- Scientific executable: `f8582808b6e2be0753cb2689d9d6d4aeb4d57aeb`
- Result schema: `anra-cs-transfer-001-final-result/v1`
- Preregistration SHA256: `419c5e049fd9b46db596deb28485eb208712c0961dfe9fd9c6a4d5efb401f5d5`
- Raw Drive artifact: `MyDrive/CYMEK/CS_TRANSFER_001_A2/receipts/FINAL_RESULT.json`
- Uploaded raw artifact SHA256: `37e4bf741f9e55fc942fe4ade62d2d1c9619e4992e4a42ab507730809c12b150`
- Treatment gap convention: `PHYS_4096 - PHYS_24576`.

## Official result

**Verdict: `PARTIAL_OR_INTERACTION`**

Development aggregate:

- control formation floor: PASS
- mean paired identity formation-AUC gap: `-0.034765625`
- mean paired identity endpoint gap: `-0.06875`
- positive AUC pairs: `1 / 4`
- pairs within absolute 0.10 AUC gap: `2 / 4`
- sealed remained unseen during development aggregation: `true`

Sealed identity endpoint:

- pair 0 gap: `-0.1833333333333333`
- pair 1 gap: `+0.10833333333333331`
- pair 2 gap: `-0.42499999999999993`
- pair 3 gap: `-0.058333333333333334`
- mean sealed identity endpoint gap: `-0.1395833333333333`

Three of four sealed matched pairs therefore favor the physical V=24,576 arm on identity endpoint. Pair 1 reverses direction, so the result is not a clean reverse-effect claim; it is seed-sensitive / interaction-dominated under this development-scale protocol.

Sealed identity exact+valid-EOS by pair:

| pair | PHYS_24576 | PHYS_4096 | 4096 - 24576 |
|---:|---:|---:|---:|
| 0 | 0.608333 | 0.425000 | -0.183333 |
| 1 | 0.241667 | 0.350000 | +0.108333 |
| 2 | 0.633333 | 0.208333 | -0.425000 |
| 3 | 0.116667 | 0.058333 | -0.058333 |

## Evidence update

CS-TRANSFER-001 does **not** support the simple hypothesis that physically shrinking the tied vocabulary/output matrix from 24,576 to 4,096 is a robust fix for identity/copy formation. The mean development and sealed effects are negative in the preregistered `4096 - 24576` direction, while one seed pair reverses strongly enough to require an interaction/seed-sensitivity interpretation.

Combined with completed R1C (`SOFTMAX_COMPETITION_NOT_SUFFICIENT`), the evidence now weakens both simple explanations:

1. inactive output-class / denominator competition alone is not sufficient; and
2. physical class-space reduction alone is not a robust causal remedy.

The remaining search should focus on mechanisms that can interact with tied-row geometry, optimizer dynamics, and weight decay rather than changing the production vocabulary on this evidence.

## Frozen next action

`run a narrow mechanism dissection separating tied-row geometry from denominator competition/weight-decay effects; do not change production vocab`

## Claim ceiling

`Controlled development-scale causal evidence about physical tied class-space effects on formation under a shared low-ID production-tokenizer surface. No production tokenizer, corpus, PRE500M, 250M, 500M, cognition, or AGI authorization.`

No production-vocabulary change and no PRE500M/250M/500M authorization follows from this experiment.
