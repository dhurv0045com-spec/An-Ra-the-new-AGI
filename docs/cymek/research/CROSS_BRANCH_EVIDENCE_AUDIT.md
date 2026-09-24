# Cross-branch evidence audit — phase 3

**Audit date:** 2026-09-24
**Authority:** [`../../research/EVIDENCE_SOURCE_MANIFEST_2026-09-24.json`](../../research/EVIDENCE_SOURCE_MANIFEST_2026-09-24.json), SHA-256 `c164c735ec4e628a6311fe4c52117c8b58370321df61b00d4976c11769f0215f`
**Evidence ledger:** [`../../research/EXPERIMENT_EVIDENCE_LEDGER.json`](../../research/EXPERIMENT_EVIDENCE_LEDGER.json), 90 experiments
**Consolidation state:** standalone commit on the target branch; `90f77b7f` is the pre-consolidation parent.

## Scope and topology correction

All current origin branches and relevant archive tags were reviewed through 2026-09-24. The review selected **71 byte-identical evidence/provenance files**; the source manifest records exact source refs, commits, blob identities, exclusions, and external-only artifacts. A no-import classification means the reviewed branch added no unique selected bytes, not that it was ignored.

The old history warning was false for the current branch topology. The target branch has merge-base `010798094a43ea1ce2343abd79017212b873ec35` with `origin/main`; `90f77b7f` has parent `0f293e8aea655e4c0809d3186e486627fbe98d63`; and `28bf57a` has parent `1460741c94379bc9b51b2dc1640ef3a4ccbe31a3`. The current target is not a parentless squash and is not correctly described as two disconnected current shards. Unreachable historical objects remain a separate custody risk.

## Post-cutoff evidence matrix

| Evidence | Current status | Exact boundary | Audit consequence |
|---|---|---|---|
| `CYR-GPU-014-R1C` | Complete 24/24; `SOFTMAX_COMPETITION_NOT_SUFFICIENT`; K01 **FIRED** | Mean `MASK_4096 - FULL_24576` gap `-0.11038062283737024`; fixed physical V24576 controlled development mechanism scope. | Inactive-softmax competition is not a sufficient tested mechanism. Do not rerun as a mask-only campaign and do not claim V24576 optimality. |
| `CS-TRANSFER-001` | Complete `PARTIAL_OR_INTERACTION` | Physical V4096 is not a robust remedy; controlled development-scale surface; raw Drive result remains external with SHA-256 `37e4bf741f9e55fc942fe4ade62d2d1c9619e4992e4a42ab507730809c12b150`. | Physical reduction and simple transfer story are not settled in favor of V4096. No natural-language or scale claim. |
| `CYMEK-V5.1-CANARY-V2` | Complete 360 updates / 1,474,560 tokens; mechanical PASS, identity formation FAIL | One frozen narrow canary; family-specific identity/copy endpoint. | Mechanical completion is not cognition or production readiness. |
| `FORMATION-MUX-001-S5-V8` | Complete 24/24 and sealed; `INCONCLUSIVE` | Formal NULLs at a near-zero identity baseline; `M3-M2` exploratory only. | Formal NULLs remain authoritative, but no mechanism is exonerated. |
| `FORMATION-MUX-001-V12-FRONTIER-PARTIAL` | `IN_PROGRESS`; 2/24; recovery engineering CI passed but original Output absent | No sealed evaluation, final result, or checkpoint payload; no recovery execution. | Later frontier cannot be merged with S5 or used for promotion. |
| `ROLE-TRANSFER-001` | `NOT_TESTED`; preregistered execution-blocked | Four-arm design, 12 fresh blocks, full-preclip norm matching, clipping/manipulation gates, clean-room endpoint; no trainer, official arm, sealed evaluation, or result. | Design of record supersedes the old tied-row placeholder but authorizes no science or execution. |
| `HORM-003` / `HORM-004` | Separate five-seed miniature `NOT_SUPPORTED` results | HORM-004 success fraction is 0.0 on every seed; rich dynamic appraisal was not measured. | Do not infer that hormonal modulation or rich appraisal is broadly disproved. HORM-001/002 custody is blocked. |
| `BRAMASTRA-K8-20260922` | Completed engineering-partial `INCONCLUSIVE` | E0/E1 executed; E2 zero qualified receipts and not positive cognition; E3 blocked; E4/E5 not run. | Engineering execution cannot be relabeled cognition, tool learning, AGI, or RSI. |
| `GANDIVA-TPU-100M-PREFLIGHT` | `NOT_TESTED` / engineering not run | No TPU run, optimizer update, checkpoint result, or qualification receipt. | No TPU qualification or training authorization. |
| `ARK-020-V4` | `DO_NOT_RUN/NOT_EXECUTED` | Resume/identity defects confirmed; all authorization flags false. | No ARK-020 or Guardian result exists in the consolidation. Guardian V4 raw bundle is missing. |
| `CITADEL-DATA-001` / `CITADEL-EVAL-001` | Data/evaluation `NOT_READY` | Leakage, latest-position shortcut, duplication, and supply failures; old surface cannot support future positive lift-off claims. | Regenerate corpus/evaluation before transfer; no production corpus. |
| `ARK-014` / rerun | Narrow local result/imported receipts | Order-robustness binding result; retention screen zero-event. | Preserve narrow claim ceiling; do not inflate to broad capability. |

## External-only and custody boundaries

The source manifest is the complete index. Important external references include:

- R1C operator bundle SHA-256 `2a9359e49f792f962774e42b4e5825b93fb7c91ddcad8f530c1c6f16c6f9bf9e`;
- R1/R1B prior bundle hashes `a22b538396a3d0957a60a27f39b0cf3dd3b20874585b4c15a03207224f613d29` and `7ffebfd49ad0bd8d81035e3cee56b23a5f31f34ba8af0f915408409e31b62792`;
- Canary raw bundle and persistent state/checkpoints, with no full selected bundle hash;
- Formation-Mux S5 bundle SHA-256 `859489d9babc532a4ed1e785af4e4993541b93730f7166dff2946a26cc996bc5`, partial archive SHA-256 `3e3ad68cd7f80bd242733b61153d4bb8f3fedbb1e4fba8fbc0f3ebc5d904423f`, and missing original checkpoint payload. Recovery-preflight engineering passed remotely, but no recovery execution occurred;
- CS raw Drive result SHA-256 `37e4bf741f9e55fc942fe4ade62d2d1c9619e4992e4a42ab507730809c12b150`;
- Citadel T1D raw bundle with no full selected hash;
- K8 result pack SHA-256 `688c8e1838bebb12a2ba3716808bf30ef07d3e0d4bd85f2069a6ff52fbed577c` and prepared data bundle SHA-256 `6fb94b7018406632b0e62dcd23ca777046ae5d881363bb8e8c78d74139785fd6`;
- HORM-001/002 historical manifest hashes `458902c08373a5a2109757f8209f71f3035a8db216125ef7ab0e81198a228ad5` and `77923d08c080ac2d11a4a7c6f756577bf29d7d9a81911cc9fc78e2ad182fbe62`, with custody blocked.

A compact receipt can support its recorded narrow outcome while the raw bundle remains external. A custody audit cannot recover bytes, authenticate a missing checkpoint, rerun an experiment, or create a scientific result.

## Current decision

The first action is `FMUX-CONTROL-METRIC-PREFLIGHT`, followed by exact recovery of the original checkpoint-bearing Output and conditional frozen-frontier completion only if capability, metric, and custody gates pass. Recovery-preflight engineering is qualified but has not recovered state. Preregistered `ROLE-TRANSFER-001` is the blocked successor of record after implementation, evaluator/no-overlap freeze, and independent qualification; it supersedes the old tied-row placeholder and must not be duplicated. Corpus/evaluation regeneration runs in parallel and is required before natural-language or scale transfer. No broad mechanism campaign starts from a floor.

R1C and CS-TRANSFER-001 are complete evidence. The historical full-exposure compact-bridge proposal is not a current next action. No result in this audit authorizes a production vocabulary or tokenizer change, PRE500M, 250M, 500M, cognition, AGI, TPU qualification, tool learning, or RSI.
