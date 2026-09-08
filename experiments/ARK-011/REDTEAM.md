# ARK-011 RED TEAM

## Main threats checked

1. **Controller leakage into the primary endpoint** — blocked by design. OOD_CONTROL alone triggers acquisition/collapse/recovery; OOD_SEALED is measurement-only.
2. **Unequal future data** — blocked by matched continuation indices from the exact same recovery snapshot.
3. **Post-hoc seed selection** — blocked for ARK-011: fresh acquisition seeds 1313/1414/1515 and orders 5701..5704 were frozen before execution.
4. **Pseudo-replication** — six forks are not six independent models. The independent acquisition count is three.
5. **Permanent-forgetting language** — rejected. The endpoint is recurrent post-recovery instability; HIGH arms can later recover.
6. **Low LR merely freezing parameters** — still a live alternative explanation for broad usefulness. ARK-011 establishes protection, not plasticity.
7. **Conditional event selection** — the decisive analysis necessarily conditions on orders that both collapsed and recovered. This is appropriate for the stated question (what to do after recovery) but does not estimate population event incidence.

## Integrity audit

Uploaded Discovery V6 bundle SHA256: `1ec3224075de49b080e229111e4c4f897430c3eca888c487116aff07a65c8d15`.

ARK-011 source receipt SHA256: `0b677b7ab07d0064466c6080d6ae5c876956341bd5f787d7a07f0b5d33be4558`.

The uploaded bundle's canonical receipt-hash algorithm independently revalidated this receipt. Runner commit was `2238f5519c3225f44010e74ba6d1af9f31d8524f`; CUDA / torch `2.11.0+cu128`.

## Red-team verdict

**The preregistered Micro T2 adaptive-protection claim survives this audit.** Promotion beyond the micro arithmetic domain is not justified until transfer and plasticity gates succeed.
