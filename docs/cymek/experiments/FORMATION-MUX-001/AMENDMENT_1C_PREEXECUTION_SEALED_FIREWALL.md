# FORMATION-MUX-001 — Amendment 1C (prospective pre-execution sealed firewall)

Status: **PROSPECTIVE_PREEXECUTION**  
Outcome access before amendment: **NONE**  
Applies to: `CS-MECH-002` and `REP-FORM-003A`  
Supersedes for execution: Science S1/S2/S3 sealed-data custody only; their historical files remain unchanged.

## Pre-execution defect found

The Science-S3 runner passed one manifest containing `training`, `development`, and `sealed` rows to every training worker. Training code did not score the sealed rows, but the rows were nevertheless deserialized into worker memory and the all-JSON results package could expose the manifest during a partial session. That violates the declared firewall that sealed examples must be unavailable to training workers and unavailable for interim inspection.

No FORMATION-MUX scientific arm has been executed and no FORMATION-MUX development or sealed outcome has been observed. This amendment therefore changes custody before outcomes, not in response to results.

## Amendment

1. The official persisted worker surface contains **training + development only**. It contains no sealed rows and no sealed shortcut statistics.
2. Before science starts, the deterministic generator may construct the full surface in the trusted coordinator solely to bind cryptographic commitments. Raw sealed rows are not persisted and are discarded before workers launch.
3. The public manifest records experiment-specific SHA-256 commitments to the sealed split plus the immutable full-surface commitment. These hashes reveal no examples or model outcomes.
4. Training, qualification, calibration, checkpoints, partial-session output, and failure bundles receive only the public manifest. A worker must fail closed if a non-empty `sealed` split is present.
5. Raw sealed rows are regenerated deterministically **only after** the corresponding experiment has a frozen `DEVELOPMENT_COMPLETE` aggregate and immediately after its sealed marker transitions to `STARTED`.
6. Regeneration must reproduce the public training/development rows, full-surface commitment, tokenizer identity, and the experiment-specific sealed commitment exactly before scoring is allowed.
7. Raw sealed rows are never written to the campaign output or results ZIP. Only the commitment and post-consumption aggregate metrics are persisted.
8. `CS-MECH-002` arms, model geometry, optimizer treatments, seeds, update budget, endpoints, thresholds, and causal contrasts are unchanged.
9. `REP-FORM-003A` arms, renderings, seeds, processed-token budget, exposure tolerance, endpoints, thresholds, and contrast are unchanged.

## Claim discipline

This amendment fixes data custody and leakage risk only. It creates no positive evidence for any mechanism, rendering, architecture, tokenizer, scale, or AGI claim.
