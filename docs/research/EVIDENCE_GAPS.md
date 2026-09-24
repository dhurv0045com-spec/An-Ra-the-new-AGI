# EVIDENCE GAPS

**Phase:** 3
**Date:** 2026-09-24
**Authority:** [`EVIDENCE_SOURCE_MANIFEST_2026-09-24.json`](EVIDENCE_SOURCE_MANIFEST_2026-09-24.json), SHA-256 `c164c735ec4e628a6311fe4c52117c8b58370321df61b00d4976c11769f0215f`.

The current ledger has 90 experiments. The consolidation imported 71 byte-identical evidence/provenance files from reviewed current branches and archive tags. Commit `90f77b7f` is the pre-consolidation parent; the current branch contains the consolidated state.

## Current active gaps

| Gap | Current consequence | Cheapest legitimate closure |
|---|---|---|
| Formation-Mux primary identity floor | S5 formal NULLs are authoritative but cannot identify or exonerate a mechanism. | `FMUX-CONTROL-METRIC-PREFLIGHT`; if custody or metric resolution is insufficient, perform read-only custody/metric review. |
| Formation-Mux v12 custody | The later frontier is 2/24 with no sealed evaluation, final result, or checkpoint payload. Recovery-preflight engineering passed remotely, but no state was recovered. | Recover and hash-verify the original saved Output and pass the pinned preflight before any continuation; do not retrain from the evidence-only archive. |
| ROLE-TRANSFER-001 readiness | Protocol bytes are frozen, but no trainer, official arm, sealed evaluation, or result exists; upstream Output/frontier and implementation gates remain open. | Complete FMUX capability/custody gates, conditional frozen-frontier completion, generator/trainer/evaluator/no-overlap implementation, and independent remote qualification before any authorization review. |
| CS-TRANSFER raw result | Compact imported records do not include the raw Drive result or checkpoints. The raw result is recorded externally with SHA-256 `37e4bf741f9e55fc942fe4ade62d2d1c9619e4992e4a42ab507730809c12b150`. | Recover the exact Drive artifact and verify the hash before a stronger custody claim; the recorded result remains valid only at its current development-scale ceiling. |
| Guardian V4 raw bundle | ARK-019-V4 is transcribed/external only; the raw bytes are absent. Guardian validity is unresolved. | Recover and byte-audit the bundle; absent or failed audit means no Guardian promotion and no ARK-020 interpretation. |
| ARK-020 readiness | Resume, identity, missing-checkpoint, controller-coverage, and provenance defects remain. Status is `DO_NOT_RUN/NOT_EXECUTED`. | Repair and independently validate readiness before any authorization decision; do not run a campaign to fill the gap. |
| Citadel corpus | Latest-position shortcut, leakage, duplication, and supply failures make the old surface unusable for future positive lift-off claims. No production corpus is materialized. | `CORPUS-REGEN` followed by contamination, shortcut, leakage, supply, and sealed-fixture screens. |
| Citadel evaluation | T1D is shortcut/leakage compromised; PRE500M was not executed. | Regenerate the evaluation surface and rerun the attack battery before any transfer or promotion claim. |
| TPU execution | The 100M preflight was not run and made no optimizer update. | Run only the dedicated zero-update backward preflight if separately scheduled; a pass would still be engineering-only. |
| K8 data/contract repair | E3 was blocked by insufficient tool-training data; E4/E5 were not run. | Repair and validate data cardinality and xprobe/export contracts before any new K8 allocation. |

## External-only artifact register

The following are not local evidence bytes and must not be described as recovered artifacts:

- R1C operator bundle: SHA-256 `2a9359e49f792f962774e42b4e5825b93fb7c91ddcad8f530c1c6f16c6f9bf9e`; raw ZIP excluded.
- R1 prior bundle: SHA-256 `a22b538396a3d0957a60a27f39b0cf3dd3b20874585b4c15a03207224f613d29`; R1B prior bundle: `7ffebfd49ad0bd8d81035e3cee56b23a5f31f34ba8af0f915408409e31b62792`.
- Canary-v2 raw bundle and persistent state/checkpoints/raw rows: external; no full selected bundle hash.
- Formation-Mux S5 bundle: SHA-256 `859489d9babc532a4ed1e785af4e4993541b93730f7166dff2946a26cc996bc5`; partial archive: `3e3ad68cd7f80bd242733b61153d4bb8f3fedbb1e4fba8fbc0f3ebc5d904423f`; original checkpoint tree is absent.
- CS raw Drive result and checkpoints: external; final-result SHA-256 is recorded above.
- Citadel T1D raw bundles: external; the selected record does not contain a full bundle hash.
- K8 result pack: SHA-256 `688c8e1838bebb12a2ba3716808bf30ef07d3e0d4bd85f2069a6ff52fbed577c`; prepared data bundle: `6fb94b7018406632b0e62dcd23ca777046ae5d881363bb8e8c78d74139785fd6`.
- HORM-001/002 historical result/checkpoint sets: manifest hashes `458902c08373a5a2109757f8209f71f3035a8db216125ef7ab0o81198a228ad5` and `77923d08c080ac2d11a4a7c6f756577bf29d7d9a81911cc9fc78e2ad182fbe62`; custody is blocked.
- ARK-014 checkpoint payloads and the TPU preflight result bundle are not imported.

A compact result receipt can support a narrow outcome while its raw bundle remains external. An audit, custody report, or protocol cannot substitute for missing bytes.

## Historical gaps corrected by phase 3

The former R1C launcher/provenance warning is a historical audit finding, not a current blocker: R1C is complete 24/24 and K01 is fired. The old parentless-squash/two-shard warning is also not current topology. The target branch has a merge-base with `origin/main` at `010798094a43ea1ce2343abd79017212b873ec35`; `90f77b7f` and `28bf57a` both have parents. Unreachable historical objects remain preservation items, but they do not justify reopening completed R1C/CS evidence.

## Standing closures

The next sequence is control/metric preflight, exact original-Output recovery, conditional frozen-frontier completion, then implementation/readiness of preregistered execution-blocked `ROLE-TRANSFER-001`; corpus/evaluation regeneration proceeds in parallel before natural-language or scale transfer. The old tied-row placeholder is superseded. No broad mechanism campaign may start from a floor, and no gap closure authorizes production vocabulary/tokenizer change, PRE500M, 250M, 500M, cognition, AGI, TPU qualification, tool learning, or RSI.
