# Evidence Source Manifest — 2026-09-24

Machine-readable authority: `docs/research/EVIDENCE_SOURCE_MANIFEST_2026-09-24.json`, SHA-256 `c164c735ec4e628a6311fe4c52117c8b58370321df61b00d4976c11769f0215f`.

## Scope

- Re-freeze phase: 3
- Target: `research/evidence-consolidation-2026-09-25`
- Merge base: `origin/main` at `010798094a43ea1ce2343abd79017212b873ec35`
- `origin/main` at import: `b620f1c2d084b0db5fa0d24196949f88c92edc1e`
- Pre-consolidation target tip: `90f77b7fa6ffd99f5a982263f03b2908298805ec`; this is the parent of the consolidated head.
- Consolidation: 71 compact evidence/provenance files were selected and restored directly from exact Git refs, then committed with the canonical ledgers, validators, tests, and CI; no merge or cherry-pick.
- Consolidation state: completed as a standalone commit on the target branch; existing target ledgers and narratives remain authoritative context and were not re-imported.


## Evidence precedence

1. Byte-identical source Git blob and ref/commit.
2. Explicit result, receipt, postmortem, or integrity audit, within its recorded scope.
3. Preregistration, protocol, surface, and custody records, which do not create outcomes by themselves.
4. Engineering, preflight, build, no-update, and partial records, which do not imply cognition, AGI, production readiness, or scale.
5. Existing canonical context, preserved but not changed.
6. External-only raw artifacts, retained as references until independently recovered and verified.

## Coverage review

All current origin branches and relevant archive tags were reviewed through 2026-09-24 at the recorded tips or commits. A no-byte-import decision is an evidence-coverage classification, not a claim that branch content was ignored. The historical evidence cutoff remains 2026-09-13 and is distinct from the current review-through date.

## Origin branch matrix

| Ref | Tip | Decision | Coverage | Reason |
|---|---|---|---|---|
| `origin/Arkenstone` | `4ae9e3b6dc806d6ada7b7b2f05d32271d072d543` | Exclude | `historical_evidence_already_represented` | Historical Arkenstone evidence is already represented in the target ledger; the current ARK-020 readiness record was imported from its dedicated `origin/arkenstone-ark020-v4` branch. The branch was reviewed at the recorded tip and contributed no additional unique bytes. |
| `origin/BRAMASTRA` | `63d60b3523937703183f1e578fe066df08a53acd` | Exclude | `superseded_by_descendant` | The reviewed BRAMASTRA K8 line is represented through descendant `origin/Gandiva`, which supplied the retained K8 result and related records; no new unique bytes were imported from this branch. |
| `origin/Gandiva` | `49a3717ae9dc81138f55dc2398c7615f8c80b4b1` | Include | `imported_unique_evidence` | Reviewed at the recorded descendant tip; imported the compact K8 result, TPU-preflight handoff, and FINAL_K8 build/readiness records. |
| `origin/arkenstone-ark020-v4` | `308a9865af519ccedd1fe13759e3b8ccdca85907` | Include | `imported_unique_evidence` | Reviewed at the recorded tip; imported `RUN_READINESS.json` from this dedicated ARK-020 branch as blocked engineering evidence. |
| `origin/arkenstone-astra` | `249e3eb0ac1ebbb461c829fe1541ab307eea5e79` | Exclude | `packaging_duplicate` | Packaging/duplicate implementation only: the reviewed branch packages the BRAMASTRA/research-environment implementation and has no unique primary evidence bytes to import. |
| `origin/citadel` | `e96c9a9a07af1d9126930190cbf3d1c764523e2d` | Exclude | `immutable_archived_descendant_preferred` | The immutable archived descendant `archive/deleted/eval-integrity-001-915e23ff22db` is preferred for the reviewed evaluation-integrity evidence; the live branch was reviewed but supplied no additional unique bytes. |
| `origin/codex/cyhex-integrity-audit` | `67220e2cbf012b4fa6f5a37990bb15e19cbc0d64` | Include | `imported_unique_evidence` | Reviewed at the recorded tip; imported the CYR replay/ledger, Formation-Mux custody, and HORM custody/repair records. |
| `origin/cyhex-hermes` | `e60e3dbc420f35c4abdc9c3de385c4d3e6692a19` | Exclude | `superseded_by_descendant` | Superseded for integrity and custody by descendant `origin/codex/cyhex-integrity-audit`, which supplied the imported audit and custody records; no new unique bytes were imported from this branch. |
| `origin/cymek-500m-readiness` | `dc915d4424103a7d528e519d50bd7fb460a83a9a` | Include | `imported_unique_evidence` | Reviewed at the recorded tip; imported the complete compact CYR-GPU-014-R1C evidence directory. |
| `origin/cymek-beta` | `194b98c8bddbe85d362fd5bf391ad68e3303ec79` | Include | `imported_unique_evidence` | Reviewed after a late tip advance; imported updated Formation-Mux recovery/CI records, blocked ROLE-TRANSFER-001 preregistration/readiness, and HORM compact results. |
| `origin/cymek-cs-transfer-001` | `b52fe453bdc68a558b8db1752758ec89832d53f0` | Include | `imported_unique_evidence` | Reviewed at the recorded tip; imported the compact CS-TRANSFER protocol, amendment, engineering, and qualification records. |
| `origin/cymek-next-core-architecture` | `28e3cd025e67c4922da24e20791d807474759674` | Include | `imported_unique_evidence` | Reviewed at the recorded tip; imported the completed Formation-Mux S5 v8 result, audit, surface preregistration, and Kaggle records. |
| `origin/cymek-v51-canary` | `d87aa519455d6d3ca111454cb1a7c8f21c5f2190` | Exclude | `immutable_archived_descendant_preferred` | Superseded for the completed canary record by immutable archived descendant `archive/deleted/cymek-v51-canary-v2-22d1bd4f05f1`; the live branch's historical readiness is not a separate import. |
| `origin/main` | `b620f1c2d084b0db5fa0d24196949f88c92edc1e` | Exclude as base | `base_context` | Reviewed as the `origin/main` repository-context tip at import (`b620f1c2d084b0db5fa0d24196949f88c92edc1e`); it is not an evidence source, while the target/origin-main merge base is recorded separately. |
| `origin/research/evidence-consolidation-2026-09-25` | `90f77b7fa6ffd99f5a982263f03b2908298805ec` | Exclude as target | `target_reference` | Reviewed as the target branch tip immediately before this consolidation commit (`90f77b7fa6ffd99f5a982263f03b2908298805ec`); it is the parent of the consolidated head, not an evidence source. |
| `origin/triquetra` | `f23f0af42d90847cf1d2c244160c8203d1995b33` | Exclude | `no_new_post_cutoff_unique_evidence` | Pre-cutoff Triquetra evidence is already represented in the target ledger; the reviewed tip contains no new unique post-cutoff result to import. |

`origin/HEAD` was reviewed as a symbolic alias for `origin/main`, not an additional branch tip.

## Archive-tag matrix

| Tag | Commit | Decision | Coverage | Reason |
|---|---|---|---|---|
| `archive/deleted/codex__arkenstone-improvements-516c28060ab4` | `516c28060ab4e06cd990ab2a8bb9fef1fbf30628` | Include | `imported_unique_evidence` | Reviewed; imported five ARK-014 JSON files from this immutable archive tag; no ZIPs or checkpoint payloads were imported. |
| `archive/deleted/core-exp-51124deda678` | `51124deda678f361e34bbeb05dd02633dab366a5` | Exclude | `no_new_post_cutoff_unique_evidence` | Reviewed; this pre-cutoff core/result snapshot has no new post-cutoff unique evidence beyond records already represented in the target ledger. |
| `archive/deleted/core-frozen-v4-f72f1939d10b` | `f72f1939d10bb76beaaf8749ee9436049239a6cb` | Exclude | `no_new_post_cutoff_unique_evidence` | Reviewed; this pre-cutoff standalone-core implementation snapshot is already represented by the target's core context and has no new post-cutoff unique evidence. |
| `archive/deleted/cymek-v51-canary-v2-22d1bd4f05f1` | `22d1bd4f05f1deb62679e6f06a8dbe268edee9c0` | Include | `imported_unique_evidence` | Reviewed; imported all five compact files under `docs/cymek/v51_canary_v2` from this immutable archived v2 tag. |
| `archive/deleted/esoes-85f44b7b449f` | `85f44b7b449f2ee39a0e80203a2d7df04614983b` | Exclude | `no_new_post_cutoff_unique_evidence` | Reviewed; this pre-cutoff launch-readiness and implementation snapshot is engineering context with no new post-cutoff unique evidence. |
| `archive/deleted/eval-integrity-001-915e23ff22db` | `915e23ff22dbd6ab9125bbca3b8c8081795f073b` | Include | `immutable_archived_descendant_preferred` | Reviewed; imported Citadel data, evaluation, and T1D evidence from this immutable archived descendant rather than the live branch. |
| `archive/deleted/iterate500-b43842039efe` | `b43842039efea4ccf41903b46ac6a33ec6d58322` | Exclude | `no_new_post_cutoff_unique_evidence` | Reviewed; this pre-cutoff notebook/runtime snapshot has no new post-cutoff unique evidence and is not a primary result source. |
| `archive/deleted/iterate900-6fbd2c0dc644` | `6fbd2c0dc644be79d098cfd45a1c28a8c2ecebee` | Exclude | `no_new_post_cutoff_unique_evidence` | Reviewed; this pre-cutoff TPU runtime snapshot has no new post-cutoff unique evidence and is not a primary result source. |
| `archive/deleted/noop-test-should-not-create-a916d1c8d263` | `a916d1c8d2637abb86d16b1c78e418c95461f3c7` | Exclude | `superseded_by_descendant` | Reviewed; this runbook snapshot is superseded by the current `origin/cymek-cs-transfer-001` source, whose later compact runbook was imported. |

The non-archive `milestone/0001-honest-loop` tag was reviewed; its historical milestone record is already represented in the target ledger and has no new post-cutoff unique evidence.

## Imported evidence index

The JSON manifest is authoritative for every path's exact blob ID and size.

| Group | Source ref | Commit | Files | Boundary |
|---|---|---|---:|---|
| R1C | `origin/cymek-500m-readiness` | `dc915d4424103a7d528e519d50bd7fb460a83a9a` | 10 | Completed controlled development mechanism evidence. |
| Canary-v2 | `archive/deleted/cymek-v51-canary-v2-22d1bd4f05f1` | `22d1bd4f05f1deb62679e6f06a8dbe268edee9c0` | 5 | Narrow canary formation failure; mechanical gates are not cognition. |
| Formation-Mux S5 v8 | `origin/cymek-next-core-architecture` | `28e3cd025e67c4922da24e20791d807474759674` | 5 | Completed 24-arm S5 v8; formal floor-limited NULLs. |
| CS-TRANSFER | `origin/cymek-cs-transfer-001` | `b52fe453bdc68a558b8db1752758ec89832d53f0` | 11 | Prospective protocol and qualification; no new outcome imported. |
| Citadel | `archive/deleted/eval-integrity-001-915e23ff22db` | `915e23ff22dbd6ab9125bbca3b8c8081795f073b` | 13 | Data/evaluation not ready; T1D archived with confounds. |
| ARK-014 | `archive/deleted/codex__arkenstone-improvements-516c28060ab4` | `516c28060ab4e06cd990ab2a8bb9fef1fbf30628` | 5 | Narrow binding result plus protocol/preflight; no broad claim. |
| Formation-Mux recovery/HORM/Role-Transfer | `origin/cymek-beta` | `194b98c8bddbe85d362fd5bf391ad68e3303ec79` | 8 | Recovery engineering CI plus blocked exact recovery; partial v12/frontier custody; HORM negatives; preregistered but unexecuted Role-Transfer design. |
| CYHEX audit | `origin/codex/cyhex-integrity-audit` | `67220e2cbf012b4fa6f5a37990bb15e19cbc0d64` | 8 | Provenance/custody diagnostics; no recovery or causal inference. |
| Gandiva | `origin/Gandiva` | `49a3717ae9dc81138f55dc2398c7615f8c80b4b1` | 5 | Engineering partial/build/no-update evidence only. |
| ARK-020-V4 | `origin/arkenstone-ark020-v4` | `308a9865af519ccedd1fe13759e3b8ccdca85907` | 1 | Blocked engineering readiness; do not run. |

### Paths

**R1C — 10 files**

```text
docs/cymek/experiments/CYR-GPU-014-R1C/FINAL_RESULT.json
docs/cymek/experiments/CYR-GPU-014-R1C/FINAL_RESULT.md
docs/cymek/experiments/CYR-GPU-014-R1C/PLAN.md
docs/cymek/experiments/CYR-GPU-014-R1C/PREEXECUTION_AUDIT.md
docs/cymek/experiments/CYR-GPU-014-R1C/PREEXECUTION_COMPATIBILITY_AMENDMENT.md
docs/cymek/experiments/CYR-GPU-014-R1C/PREREGISTRATION.json
docs/cymek/experiments/CYR-GPU-014-R1C/RUN_READINESS.json
docs/cymek/experiments/CYR-GPU-014-R1C/RUN_READINESS_V2.json
docs/cymek/experiments/CYR-GPU-014-R1C/RUN_READINESS_V3.json
docs/cymek/experiments/CYR-GPU-014-R1C/RUN_READINESS_V4.json
```

**Canary-v2 — 5 files**

```text
docs/cymek/v51_canary_v2/FINAL_RESULT.json
docs/cymek/v51_canary_v2/FINAL_RESULT.md
docs/cymek/v51_canary_v2/OPERATOR_AMENDMENT_1.md
docs/cymek/v51_canary_v2/OPERATOR_RUNBOOK.md
docs/cymek/v51_canary_v2/README.md
```

**Formation-Mux S5 v8 — 5 files**

```text
docs/cymek/experiments/FORMATION-MUX-001/OBSERVED_RESULT_CURRENT.json
docs/cymek/experiments/FORMATION-MUX-001/RETURNED_BUNDLE_AUDIT.md
docs/cymek/experiments/FORMATION-MUX-001/SURFACE_PREREGISTRATION_V4.json
docs/cymek/experiments/FORMATION-MUX-001/observed/S5_KAGGLE_2026-09-15/ARM_SUMMARY.json
docs/cymek/experiments/FORMATION-MUX-001/observed/S5_KAGGLE_2026-09-15/POSTMORTEM.md
```

**CS-TRANSFER — 11 files**

```text
docs/cymek/cs_transfer_001/ENGINEERING_AUDIT.md
docs/cymek/cs_transfer_001/ENGINEERING_REPAIR_E1.md
docs/cymek/cs_transfer_001/FAILURE_MODES.md
docs/cymek/cs_transfer_001/OPERATOR_AMENDMENT_2.md
docs/cymek/cs_transfer_001/OPERATOR_BINDING.json
docs/cymek/cs_transfer_001/OPERATOR_RUNBOOK.md
docs/cymek/cs_transfer_001/QUALIFICATION_REPAIR_E1.json
docs/cymek/cs_transfer_001/README.md
experiments/CS_TRANSFER_001/AMENDMENT_1.json
experiments/CS_TRANSFER_001/AMENDMENT_2.json
experiments/CS_TRANSFER_001/PREREGISTRATION.json
```

**Citadel — 13 files**

```text
docs/citadel/data/CONTAMINATION_REPORT.md
docs/citadel/data/DATA_READINESS.json
docs/citadel/data/DATA_READINESS_REPORT.md
docs/citadel/data/SHORTCUT_BASELINES.md
docs/citadel/evaluation/CAUSAL_VALIDITY.md
docs/citadel/evaluation/EVALUATION_FIREWALL_AUDIT.md
docs/citadel/evaluation/EVALUATION_INVENTORY.md
docs/citadel/evaluation/EVALUATION_READINESS.json
docs/citadel/evaluation/SHORTCUT_ATTACKS.md
docs/citadel/evaluation/eval_attack_results.json
docs/citadel/experiments/T1D/DEVELOPMENT_CERTIFICATION.json
docs/citadel/experiments/T1D/RESULTS.json
docs/citadel/experiments/T1D/RESULTS.md
```

**ARK-014 — 5 files**

```text
artifacts/arkenstone/ark014/ark014-cuda-2201-03/ARK-014_FROZEN_POLICY.json
artifacts/arkenstone/ark014/ark014-cuda-2201-03/ARK-014_PARTIAL.json
artifacts/arkenstone/ark014/ark014-cuda-2201-03/ARK-014_PREFLIGHT.json
artifacts/arkenstone/ark014/ark014-cuda-2201-03/ARK-014_RESULT.json
artifacts/arkenstone/ark014/ark014-cuda-2201-03/ARK-014_TASK_MANIFEST.json
```

**Formation-Mux recovery, HORM, and Role-Transfer — 8 files**

```text
artifacts/cymek/FORMATION-MUX-001/kaggle-session-20260923/RECOVERY_RUNBOOK.md
artifacts/cymek/FORMATION-MUX-001/kaggle-session-20260923/SESSION_REPORT.md
docs/cymek/experiments/ROLE-TRANSFER-001/DESIGN.md
docs/cymek/experiments/ROLE-TRANSFER-001/PREREGISTRATION_V1.json
docs/cymek/experiments/ROLE-TRANSFER-001/RUN_READINESS_V1.json
experiments/HORM-001/ANALYSIS.md
experiments/HORM-001/RESULT_horm003_prospective.json
experiments/HORM-001/RESULT_horm004_prospective.json
```

**CYHEX integrity/custody — 8 files**

```text
docs/cymek/experiments/CYR-GPU-012/CYR011_VS_CYR012_INITIAL_REPLAY.json
docs/cymek/experiments/CYR-GPU-012/CYR011_VS_CYR012_LEDGER.json
docs/cymek/experiments/FORMATION-MUX-001/CHECKPOINT_CUSTODY.json
docs/cymek/experiments/FORMATION-MUX-001/CHECKPOINT_CUSTODY_RECOVERY.md
docs/cymek/experiments/HORM_CUSTODY_AUDIT.json
docs/cymek/experiments/HORM_CUSTODY_RECOVERY.md
docs/cymek/experiments/HORM_STATE_REPAIR.md
experiments/CYR-GPU-012/REPRODUCTION_NOTE.md
```

**Gandiva — 5 files**

```text
docs/bramastra/K8_20260922_RESULT.md
engineering/reports/GANDIVA_COGNITION_TPU_PREFLIGHT_20260923/HANDOFF.md
engineering/reports/FINAL_K8/gandiva-cognition-final-20260923/BUILD_READINESS.json
engineering/reports/FINAL_K8/gandiva-cognition-final-20260923/HANDOFF.md
engineering/reports/FINAL_K8/gandiva-cognition-tpu-r6-20260923/build_verification.json
```

**ARK-020-V4 — 1 file**

```text
experiments/ARK-020-V4/RUN_READINESS.json
```

## External-only artifacts

These are references, not imports. Full details and any recorded hashes are in the JSON manifest.

- R1C result bundle `CYMEK_R1C_SOFTMAX_MECHANISM_RESULTS.zip`, SHA-256 `2a9359e49f792f962774e42b4e5825b93fb7c91ddcad8f530c1c6f16c6f9bf9e`; partial bundle hash not recorded.
- R1 prior bundle SHA-256 `a22b538396a3d0957a60a27f39b0cf3dd3b20874585b4c15a03207224f613d29`; R1B prior bundle SHA-256 `7ffebfd49ad0bd8d81035e3cee56b23a5f31f34ba8af0f915408409e31b62792`.
- V5.1 Canary-v2 result bundle `CYMEK_V51_CANARY_V2_RESULTS.zip`; no full hash recorded in the selected compact evidence. Persistent V5.1 state/checkpoints/raw rows remain at `/content/drive/MyDrive/CYMEK/V5_1_CANARY_V2`.
- Formation-Mux S5 bundle `FORMATION_MUX_001_RESULTS.zip`, SHA-256 `859489d9babc532a4ed1e785af4e4993541b93730f7166dff2946a26cc996bc5`.
- Formation-Mux partial archive `FORMATION_MUX_001_RESULTS.partial.zip`, SHA-256 `3e3ad68cd7f80bd242733b61153d4bb8f3fedbb1e4fba8fbc0f3ebc5d904423f`; `(1)`/`(2)` filename identity remains unresolved.
- Formation-Mux original Kaggle Output checkpoint tree: 30 declared checkpoint hashes in the imported custody inventory; zero checkpoint payloads imported.
- CS-TRANSFER raw Drive result `MyDrive/CYMEK/CS_TRANSFER_001_A2/receipts/FINAL_RESULT.json`, SHA-256 `37e4bf741f9e55fc942fe4ade62d2d1c9619e4992e4a42ab507730809c12b150`; raw result/checkpoints remain external.
- Citadel T1D bundle `CITADEL_T1D_RESULTS.zip`; no full bundle hash recorded. Replication bundle `CITADEL_T1D_RESULTS (1).zip` has only recorded prefix `c3f6643bf8aa88ff`.
- CYR-GPU-011 bundle SHA-256 `fbec390f66223a19a998db6519f42c046ad8cb0a345c205b168f5bd1a86668e5`; CYR-GPU-012 bundle SHA-256 `d2da6d18847c846bd1865d4069c5afe46309ced339350a0115e1c23f3049f2cc`.
- K8 result pack `k8-1fce891161b1-results-a1787617d513-results-c353d8b1.zip`, SHA-256 `688c8e1838bebb12a2ba3716808bf30ef07d3e0d4bd85f2069a6ff52fbed577c`.
- Gandiva prepared K8 data bundle SHA-256 `6fb94b7018406632b0e62dcd23ca777046ae5d881363bb8e8c78d74139785fd6`; the bundle itself was not imported.
- HORM historical result/checkpoint set: custody is blocked; manifest SHA-256 values are `458902c08373a5a2109757f8209f71f3035a8db216125ef7ab0e81198a228ad5` (HORM-001) and `77923d08c080ac2d11a4a7c6f756577bf29d7d9a81911cc9fc78e2ad182fbe62` (HORM-002), with raw missing/EOL-reconstruction states in the imported audit.
- ARK-014 checkpoint payloads and Gandiva pending TPU preflight ZIP/receipt: no imported payload or completed preflight result.

URLs present in imported evidence are recorded in the JSON manifest, including the V5.1 Colab notebook, Kaggle TPU/notebook documentation, and the Pytest warning URL embedded in captured build output. No URL was used as a substitute for missing bytes.

## Explicit exclusions

- Raw ZIPs, partial ZIPs, operator bundles, checkpoint/model payloads, optimizer/RNG state, raw train/dev/sealed rows, progress snapshots, full traces, and generated output.
- Operator code, training/model implementation, tests, source snapshots, duplicate implementation trees, and notebooks.
- All origin branches and archive tags with no imported bytes are listed in the matrices with evidence-coverage classifications.
- Existing canonical ledgers and narrative files, including the existing CS-TRANSFER final narrative, remain target-branch context and were not re-imported.
- Selected source blobs were imported without a merge or cherry-pick; the canonical consolidation is a standalone commit.

## Claim ceilings and distinctions

- **R1C:** completed controlled development mechanism evidence; it does not prove V24576 optimality or authorize tokenizer, PRE500M, 250M, 500M, cognition, or AGI claims.
- **CS-TRANSFER:** imported files are prospective physical-class-space protocol/qualification evidence. Any existing final interpretation remains controlled development-scale causal evidence; it is distinct from completed R1C and does not authorize production vocabulary or scale.
- **Formation-Mux:** completed 24-arm S5 v8 has formal floor-limited NULLs. The later 2026-09-23 v12/frontier material is only a partial custody snapshot: 24/24 S5 development arms, 2/24 TIE-ROLE frontier arms, no sealed evaluation, and no final result. Recovery-preflight engineering passed remotely, but exact recovery remains blocked on the absent original checkpoint-bearing Output.
- **ROLE-TRANSFER-001:** hash-bound preregistration/readiness only. No trainer, official arm, sealed evaluation, scientific outcome, architecture promotion, capability claim, external benchmark validity, or AGI claim exists. It supersedes the old tied-row placeholder as design of record but must not run beside it.
- **Engineering, partial, canary, null, preflight, and custody records never imply cognition, AGI, recursive self-improvement, production readiness, or scale authorization.**

## Caveats

- The archived V5.1 README retains historical pre-execution wording; the imported FINAL_RESULT record is the completed canary result and remains narrow.
- R1C readiness files are pre-execution snapshots alongside the completed final result; readiness wording must not be read as the current outcome.
- CS-TRANSFER imports do not include a new final result; the existing canonical final narrative was intentionally not changed.
- Formation-Mux recovery-preflight engineering passed in GitHub Actions run `35925460488` after preserved failed run `35923273288`, but execution remains blocked on the original checkpoint-bearing Kaggle saved Output; the evidence-only partial archive is not a substitute.
- ROLE-TRANSFER-001 is preregistered and execution-blocked. Protocol-hash CI binds bytes and is not a scientific result, trainer qualification, official arm, sealed evaluation, or authorization.
- HORM custody reports inventory and raw-byte mismatches but do not authenticate, recover, rerun, or scientifically validate the historical artifacts.
- Pre-existing repository files outside the selected imports and canonical consolidation updates were preserved; the consolidation is committed as a standalone change.

## Verification

- Source blob/byte comparison: 71 expected, 71 passed, 0 failed; 0 unselected files in import roots.
- Imported JSON parsing: 36 expected, 36 parsed, 0 errors.
- Strict manifest JSON parsing: passed.
- Authored patch whitespace check: passed with three byte-identical imported evidence files excluded for source-preserved Markdown hard breaks or EOF blank lines.
- No source file was rewritten through a text conversion; Git blob comparison is byte-level.
