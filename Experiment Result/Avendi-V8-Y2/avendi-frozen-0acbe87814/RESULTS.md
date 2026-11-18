# Avendi V8-Y2 Results Report (`avendi-frozen-0acbe87814`)

- **Phase**: `frozen`
- **Source commit**: `a3b72320aa4847e18bfe0d1fbdb6550230c02054`
- **Design SHA256**: `13f8f46c9065115b9bbdecb2884edba2ff88a3bf9e01cba2afd613739c344fbd`
- **Archive SHA256**: `2bde975cd41cbb99abd98aba9ecc3d00b86191304fb0eaa04a53fad78056e318`

## 1. Six-Row Independent Verification Matrix

| Verification Row | Status | Notes |
| --- | --- | --- |
| `byte_integrity` | `PASS` | Archive manifest and member SHA-256 verification |
| `oracle_fixture_correctness` | `PASS` | Independent tokenized-input parser, destructive erasure, and blind-policy zero ceiling |
| `recorded_action_replay` | `PASS` | Public environment replay of recorded policy actions |
| `checkpoint_restore` | `UNAVAILABLE` | Y1 S3 F00/F11 manifest, model spec, and finite tensor verification |
| `regenerated_per_case_agreement` | `UNAVAILABLE` | Joined by (checkpoint_sha256, rendered_input_sha256, case_id, decoder), FP32 max_abs <= 1e-5 |
| `training_reproduction` | `UNAVAILABLE` | P3 paired margin gradient-sign negative control and family-aligned stream verification |

## 2. Scientific Stage Statuses

| Stage | Status |
| --- | --- |
| `P0_instrument` | `QUALIFIED` |
| `P1_restoration` | `BLOCKED` |
| `P2_frozen_diagnosis` | `BLOCKED` |
| `P3_controlled_acquisition` | `NOT_EXECUTED` |
| `P4_transition_interchange` | `NOT_EXECUTED` |
| `P5_conditional_action` | `NOT_EXECUTED` |
| `P6_bounded_correction` | `NOT_EXECUTED` |

## 3. P0 CPU Instrument & Reference Policies

- **Custody & Disjointness Audit**: `PASS` (canonical 1-hop pool: `10920` families)
- **Planted Target-Only / ID-Changed Collision Control**: `PASS`
- **Destructive Goal-Erasure & Bridge-Masking Checks**: 1-hop `PASS`, 3-hop `PASS`
- **Pilot Seed 1801 Complete Families**: oracle `256/256`, last_object `0/256`, third_timestamp `0/256`, ignore_query `0/256`, ignore_bridge `0/256`

| Reference Policy | Dev 1-Hop Complete Families | Dev 3-Hop Complete Families |
| --- | ---: | ---: |
| `public_traversal` | `256/256` | `256/256` |
| `copied_terminal` | `0/256` | `0/256` |
| `last_row` | `0/256` | `0/256` |
| `rank_h_timestamp` | `0/256` | `0/256` |
| `always_abstain` | `0/256` | `0/256` |
| `empirical_majority` | `0/256` | `0/256` |
| `query_blind` | `0/256` | `0/256` |
| `bridge_blind` | `0/256` | `0/256` |

## 4. P1/P2 Frozen Y1 Diagnosis & Routing Decision

- **Restored Y1 Lineages**: `0/6`
- **Selected Anchor Arm**: `F00`
- **First Failing Dependency**: `P1_CHECKPOINT_RESTORATION`
- **Next Admitted Stage**: `P1_RESTORE_OR_P3_FRESH_ACQUISITION`
- **Missing Lineages**: `Y1-t1_factorial-S3-F00-s19, Y1-t1_factorial-S3-F00-s41, Y1-t1_factorial-S3-F00-s7, Y1-t1_factorial-S3-F11-s19, Y1-t1_factorial-S3-F11-s41, Y1-t1_factorial-S3-F11-s7` (deterministic retraining recipe emitted in `frozen_receipt.json`)

## 5. Permanent Y1 Discovery Register Status (`01`–`25`)

| ID | Stage Target | Discriminating Test |
| --- | --- | --- |
| `01` | `P0/P2` | Permute timestamps within the identical graph. Measure prediction changes and disagreement with the old rule. |
| `02` | `P5` | Hide an interior bridge and move its returned row. Score both terminal outcomes when the bridge is switched. |
| `03` | `P0/P2` | Balance directed endpoint changes; compare old versus corrected distribution on the same weights. |
| `04` | `P0` | Compare prefix bytes at the target boundary, full decision states and parent families. Plant a target-only difference and demand rejection. |
| `05` | `P0` | Oracle passes and answer-copy/query-blind controls fail family qualification on actual rendered inputs. |
| `06` | `P2/P3` | Compare tokenized prompts, signed endpoint odds, raw answers and complete pairs. |
| `07` | `P2/P3` | Test literal copy, one-hop two-key selection and equal-difficulty paired routes separately. |
| `08` | `P2` | Measure raw answer mass, within-answer odds, and exact answer/EOS on identical prompts. |
| `09` | `P3` | A single controlled rescue on new families narrows attribution; failure qualifies only the tested recipe. |
| `10` | `P2` | Keep F00 as frozen anchor; retest against shortcut-resistant family endpoint. |
| `11` | `P5` | Hold direct examples fixed, replace neutral rehearsal with availability examples; compare with a separate direct-dose replacement. Log opportunity costs. |
| `12` | `P5` | Hide an irrelevant edge with the same tool flag as a required missing edge. Score decision and downstream answer. |
| `13` | `P3` | Clone the initial corpus/order and checkpoint, then repeat versus refresh exactly matched strata. |
| `14` | `P1/P2` | Preserve separate historical identities; only a matched intervention can isolate original C. |
| `15` | `DEFERRED_SCALE` | Reopen scale after acquisition; give both scales identical preregistered LR search budgets and common evaluation. |
| `16` | `P3` | Fixed train and unseen diagnostic families at logged milestones; report onset, loss and post-change time. |
| `17` | `P3` | Export raw records for train memorization, new instances and withheld grammar separately. |
| `18` | `P2/P3` | Block matched intervention by seed and corpus; distinguish family CIs from training-replication uncertainty. |
| `19` | `P2/P3/P4/P5/P6` | Follow the failure branch and export a row/status for every registered stage. |
| `20` | `P1` | Match exact final hashes and regenerate behavior from real parents; record missing lineage explicitly. |
| `21` | `P0/P1/P2` | Issue six receipts with independent statuses; corrupt one case while preserving aggregate accuracy. |
| `22` | `P0` | Test all four on Avendi payloads plus scientific shortcut negative controls. |
| `23` | `P5/P6` | Measure the same qualified A checkpoint before/after learning B against equal-work A->A, and independently verify B. |
| `24` | `P2/P3/P4` | Reuse historical public payload and row-order controls; qualify each skill in its own setting before composing it. |
| `25` | `P6` | Randomize feedback and update route, then test transfer to new entities and retention on previously acquired skills. |
