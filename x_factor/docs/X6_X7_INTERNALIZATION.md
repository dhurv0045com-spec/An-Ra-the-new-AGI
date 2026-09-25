# X6/X7 Internalization Bridge

## Status

**PREREQUISITE BLOCKED. X6 is not ready to execute. X7 is not ready to execute.**

The prospective package is implemented and software-tested, but the X1-to-X6 scientific entry gate has not passed. No model inference, training, sealed evaluation, GPU/TPU work, checkpoint download, or external campaign was run.

Frozen contract identity:

- Protocol: `x_factor/protocols/x6_x7_v1.json`
- Protocol canonical SHA256: `720d5679e13cd1246d57a8bc13ec3b6dfc90835a71b296002b723496209aafa0`
- Protocol file SHA256: `f7b81fb2d68583338b9369aad7e390ee22a37db0c3651986641d018bc4d5cf71`
- Source-closure SHA256: `fc2b3bace607b32b52cb3c9f46449362ff3216f457517c2641b6d0f6ac6ab3ef`
- Canonical base: `codex/x-factor-real` at `49a3717ae9dc81138f55dc2398c7615f8c80b4b1`
- Gate receipt: `x_factor/receipts/x1_to_x6_entry_audit_v1.json`
- Gate receipt canonical SHA256: `e5a3879aa1385ef838ce95122b12539d714633b9fdce6f768e81bc1c501d359d`
- Gate receipt file SHA256: `b07a79fdbddce19b4587dda8c5f677e9cae2ef68cdea9c6c934ea3e090ebf4b8`
- Interface file SHA256: `9f9f4f28ecbb0d2ba89b1b43dc0aaa3eaab890d165c4e39ef1396242ce3a5c6d`
- Focused test file SHA256: `003313a0afebfaef52c318b99558baf2d6e8ed4e66549bb26aebd8d2201807a2`

## 1. Gate Zero audit

### Authority inspected

1. `x_factor/REAL_MODEL_CAUSAL_SPEC.md`
   - Requires legal interventions, matched controls, pre-outcome commitment, metrics beyond raw accuracy, null tests, and independent replication.
   - Separates observed-only, evaluator, diagnostic, and behavioral evidence.
2. `x_factor/ladder.py`
   - X6 is repair-trajectory SFT with the intervention context removed and the verified target retained.
   - X6 requires raw gain, protected-family retention, and distinguishes dependence from internalization.
   - X7 requires retained raw accuracy and falling intervention lift.
3. `x_factor/strength_ladder.py`
   - Reviewed as historical intervention instrumentation. It contains A4 oracle assistance, which is a ceiling only and cannot be used as a training target or native-capability result.
4. `AN_RA_PROGRAM.md`
   - Explicitly marks X1-REAL-0 invalid for predictive self-modeling.
   - States that stronger-checkpoint readiness and X0/X1 remain blocked.
5. `x_factor/registry/checkpoints.json`
   - File SHA256: `e091c311a8b2f987eef1d5d17ca204ef4cebb632c2dfdbb5762116ad34968c3d`
   - Contains five historical V4 candidates and zero eligible research subjects.
6. Prospective uncommitted X1 worktrees
   - `codex/x-factor-real` has uncommitted X1-REAL-1 files but no accepted X1 result receipt.
   - `codex/x1-prospective-predictor` is explicitly `BLOCKED_ON_PREREQUISITE`.
   - Its handoff file SHA256 is `b6123efcf80cbd175b0f8c4f0393377225022714ae507f14526e0673391eb125` and status is `BLOCKED_ON_PREREQUISITE`.
   - Its current uncommitted protocol file SHA256 is `933e0b76e0a86d9463c36b3c2d4540958d7b04f40ba06e3634247c1a747ec480` and current internal identity is `1e2d45cc592beb2c9a6d66fe8475e62720faab982c41c781b713e7b3e0faa7dc`.
   - The handoff still names an older protocol identity, `c33b3e2e4e72cb9fd28fb1243c2c04955f7d0b94ffee879b69c1c3890ad4b476`; this is an unresolved stale-handoff inconsistency and cannot authorize execution.
   - Its basis gate file SHA256 is `3fe71b054355f567d69bf997246ac18b3fc078cbd3394711bb64a07b1e5ff584` with status `BLOCKED_NO_ELIGIBLE_CHECKPOINT`.

No committed X6/X7 amendment or preregistration newer than the canonical X-factor base was found. The prospective files in this worktree are explicitly marked as proposals, not retroactive authority.

### Historical X1 result

Historical receipt:

- Path: `output/x1_real_receipt.json`
- File SHA256: `36214f01bd137fb7ca8181d993e26f1d288d3dd4abc6ca0843700937bc1996f9`
- Schema: `anra-x1-real/v1`
- Checkpoint string: `checkpoints/anra-v4-20k-sft3-accumulate.pt`
- `parameter_sha256`: `null`
- Reported predictor accuracy: `0.9545`
- Always-negative cell accuracy: `0.9733`
- Oracle repair coverage: `0.0889`
- Self-validating `anra-x1-real-1-analysis/v1` receipt: absent
- Independent replication receipt: absent

The paired IBQ invalidation is:

- Path: `output/ibq_legacy_basis_verdict.json`
- File SHA256: `fcad9229ec5e4697d631b8c5798628f19ab492013f7011c933ae8939a085790e`
- Verdict: `BASIS NOT QUALIFIED`
- Reason: sparse/imbalanced response matrix, low oracle coverage, degenerate interventions, and no structure beyond sparsity.

### Checkpoint qualification and continuation

The historical step-30400 checkpoint is registered as:

- File SHA256: `ccaffdfb62328886df0b4000cba47d8e1b7cb259f4bf6fc7967271d9656dc986`
- Parameter SHA256: `6175bb4691f52d000187ce6511287b650d546a72fbd3a5a15fc2dc79bd5ac89c`
- Role/status: `HISTORICAL_WEAK / NEGATIVE_CONTROL`
- Research subject: `false`
- Qualification: `CALIBRATION_ONLY / GATE_V0_NOT_QUALIFIED`

No eligible subject manifest, qualified checkpoint identity attestation, or X6 continuation contract exists. Consequently, the exact parent checkpoint file, parameter tree, tokenizer, runtime, source release, and resume code are unresolved.

### Exact missing X1 evidence

X6 remains blocked until one receipt supplies all of the following:

1. Schema `anra-x1-real-1-analysis/v1`, phase `ANALYSIS_ACCEPTED`, and a valid canonical self-hash.
2. Decision status `SUPPORTED_SCOPED` with `supported=true`.
3. Passing primary prediction and decision gates.
4. A valid independent replication receipt; new seeds alone are insufficient.
5. A qualified intervention basis and accepted X1 release/custody identity.
6. A subject manifest whose SHA256 exactly matches the X6 parent checkpoint.
7. File-verified checkpoint, parameter, model-config, tokenizer, runtime, source, qualification, and continuation identities.
8. A frozen intervention registry, target intervention, transformations, legality inputs, and cost model.

Checkpoint replication may remain explicitly `PENDING` if the accepted X1 contract requires only primary plus independent-task replication. It is not silently counted as complete.

## 2. Frozen protocol and unresolved authority

The protocol freezes all structural decisions that can be fixed without inventing authority:

- Three canonical arms: unchanged parent, repair internalization, and a proposed dose-matched ordinary supervised/replay control.
- External support removal with a bound transformation receipt.
- Independent verification and full source/world/causal-structure provenance.
- Six disjoint roles: experience, replay, development, retention, X6 confirmation, and X7 follow-up.
- World/structure/content collision rejection across all splits.
- Confirmation freeze and mandatory retirement after any debugging/tuning inspection.
- Fixed final-update or development-only checkpoint selection.
- Exact optimizer, scheduler, RNG, sampler, cursor, token-ledger, and restore identities.
- Cluster-level paired uncertainty, seed outcomes, calibration, costs, retention, and transfer.
- Separate X6 and X7 receipt states and a permanent no-AGI boundary.

The current canonical X6 authority confirms:

- Rehearsal fraction `>= 0.50`.
- Parent-relative absolute accuracy floor `0.10` for every protected capability.

Current authority does **not** specify:

- Minimum raw unaided gain versus parent.
- Minimum raw unaided gain versus the ordinary control.
- Minimum reduction in target-intervention lift.
- Minimum transfer gain by axis.
- Minimum independent worlds.
- Minimum paired seeds.
- X7 dependence-fall effect sizes or unrelated-intervention stability margin.

Those values remain `null` in `decision_thresholds`, and the readiness gate returns `RAW_GAIN_AND_LIFT_REDUCTION_THRESHOLDS_UNRESOLVED`.

A prospective amendment proposes:

- Raw unaided gain `>= 0.05` versus parent and control.
- Target-lift reduction `>= 0.05` versus parent and control.
- Transfer gain `>= 0.03` on every frozen axis with a positive lower paired interval.
- At least three paired seeds and 80 independent worlds, targeting 120.
- 5,000 paired cluster-bootstrap resamples.
- X7 raw retention/gain, target-lift reduction, transfer, and unrelated-lift bounds specified separately.

These values are `PROPOSED_ONLY_NOT_AUTHORIZED`. Development-only simulation-based power analysis must occur before approval. No confirmation outcome may be inspected before approval.

## 3. Arm and exposure contract

### PARENT

- Exact unchanged parent checkpoint.
- No training.
- Evaluated on the same fresh tasks and schedule as both children.

### REPAIR_INTERNALIZATION

- Baseline failures repaired by the frozen target intervention.
- Independent verifier confirms the target.
- The assisted context transformation is removed.
- The record retains hashes for the assisted prompt, assisted output, intervention, legality receipt, verifier, and support-removal receipt.
- The model sees only the unaided prompt and verified target.

### RAW_FAILURE_SFT_CONTROL

- Verified baseline failures not repaired in experience collection.
- Same unaided prompt schema and verified-target format.
- Same replay source.
- Proposed clarification of the canonical no-intervention/raw-failure control.
- Exists to estimate what ordinary supervised exposure and replay achieve without the repair-specific selection signal.

Both trained arms must have exact equality for:

- Optimizer updates and identity.
- Schedule and learning rate.
- Example count and prompt/target token counts.
- Replay examples, replay tokens, and replay fraction.
- Data opportunities.
- Checkpoint cadence.
- Evaluation timing.

All arms start from the same parent state tree within each seed. A dose mismatch invalidates the strong causal comparison and must be reported.

## 4. Provenance-bound dataset contract

Every training record binds:

- Original task, cluster, latent world, causal structure, and eligible source split.
- Intervention ID/version, registry hash, and legality receipt.
- Baseline failure and raw-output hash.
- Assisted prompt/output hashes and verified outcome.
- Independent verifier identity and receipt.
- Verified target and target hash.
- Unaided prompt and prompt hash.
- Support-removal transformation ID and receipt hash.
- Prompt/target token counts.
- Protocol identity and record hash.

The firewall rejects:

- X1 evaluation tasks or outcomes.
- Development, retention, confirmation, or X7 tasks.
- Sealed outcomes or evaluator answer keys.
- Answer-revealing interventions and A4 oracle assistance.
- Missing independent verification.
- Retained assisted context.
- World, structure, source, task, or content collisions.
- Unknown source split or unaudited record hashes.

Model explanations are not accepted as proof of internalization.

## 5. Split, retention, and transfer contract

All derivatives of one latent world or causal structure remain in one split. Cluster, world, structure, source, task, and task-content hashes are pairwise disjoint across roles.

A new random seed is not called structural transfer merely because the generator was rerun. Transfer status comes from the actual split identity and frozen axis.

The X6 confirmation cohort includes separate axes for:

- Known-structure instances.
- Surface rendering.
- Causal structure.
- Task family.

Checkpoint transfer is reported only if a distinct qualified checkpoint exists. A missing checkpoint is `NOT_AVAILABLE`, never replaced with the parent.

Retention is reported per protected capability. A material regression cannot be hidden by an aggregate. The X7 cohort is separately frozen and cannot be introduced after seeing X6 outcomes.

## 6. Implemented interfaces

`x_factor/x6_x7.py` implements:

- Strict JSON and canonical hashing.
- X1 receipt audit bound to the X6 parent subject.
- Parent identity, qualification, file-verification, and continuation checks.
- Six-role split and bundle validation.
- Repair-experience validation and dataset construction.
- Training-policy firewall and support-removal provenance.
- Dataset manifests with token/replay arithmetic.
- Per-seed paired-arm manifests with exact exposure matching.
- Exact continuation manifests.
- Outcome-blind evaluation commitments.
- Clean source-release manifests.
- Aggregate source-bound X6 run manifests.
- Fail-closed X6 readiness.
- X6 result arithmetic and protocol-bound receipt revalidation.
- Conditional repair-choice utility when a frozen X1 policy is supplied.
- Separate X7 readiness, fresh-cohort validation, transfer/memorization checks, and receipt arithmetic.

No model architecture or training implementation was changed.

## 7. X6 measurements and decision

For every arm and seed, the receipt reports:

- Unaided exact-answer accuracy.
- Valid stopping, abstention, invalid output, calibration, and Brier score.
- Assisted accuracy under every frozen intervention.
- Intervention lift by intervention.
- Raw change from the paired parent.
- Raw change from the ordinary matched control.
- Target-lift reduction versus parent and control.
- Repair-choice utility when a frozen X1 policy exists.
- Retention by capability.
- Transfer by axis.
- Per-cluster and per-seed results.
- Training, evaluation, intervention, and checkpoint costs.
- Controller provenance.

Aided improvement alone is `NOT_SUPPORTED` for internalization. Positive internalization requires raw unaided gain, reduced target lift, matched-control separation, and retention. Transfer is a separate gate.

Valid X6 outcomes are:

- `PREREQUISITE_BLOCKED`
- `INCONCLUSIVE`
- `NOT_SUPPORTED`
- `SUPPORTED_SCOPED_NO_TRANSFER`
- `SUPPORTED_SCOPED_WITH_TRANSFER`

## 8. X7 dependence-fall gate

X7 cannot start from a negative, inconclusive, invalid, or prerequisite-blocked X6 receipt.

It requires:

- The accepted X6 child, parent, and ordinary control.
- A fresh `X7_FOLLOWUP` cohort.
- A new outcome-blind commitment.
- No target-intervention support in unaided runs.
- The same frozen legal interventions for reduced-assistance measurements.
- The same protected retention floor.
- Fresh transfer-axis evaluation.

`DEPENDENCE_FELL` requires all of:

- Unaided child gain over parent and ordinary control.
- Target lift reduced versus both.
- Unrelated-intervention lift stable within the authorized margin.
- Retention passes.
- Fresh transfer axes pass.
- No explanation based only on training templates or ordinary supervised fine-tuning.

Repeated scripted training is not called self-improvement. Every controller is fixed, learned, human-supplied, externally assisted, or absent.

## 9. Focused validation

Executed:

```text
python -m py_compile x_factor/x6_x7.py x_factor/tests/test_x6_x7.py
PASS

python -m pytest x_factor/tests/test_x6_x7.py -q
16 passed

python -m x_factor.x6_x7 validate-protocol \
  --protocol x_factor/protocols/x6_x7_v1.json \
  --repo-root . \
  --check-source-closure
valid=true
protocol_sha256=720d5679e13cd1246d57a8bc13ec3b6dfc90835a71b296002b723496209aafa0
```

The tests cover:

- Historical X1 rejection.
- Accepted X1 receipt structure.
- Parent qualification and continuation identity.
- Split and causal-structure collisions.
- Training-label and support-removal leakage.
- Dataset token/replay binding.
- Exact exposure matching.
- Outcome-blind commitment.
- Source-bound run manifest and readiness.
- Raw unaided gain versus parent/control.
- Lift reduction.
- Per-capability retention and per-axis transfer.
- Assisted-only negative result.
- Conditional repair-choice utility.
- Receipt tamper detection with valid self-hashes.
- Independent X7 fresh-cohort and transfer gate.

Not run by design:

- Model inference or training.
- Checkpoint loading.
- Real X1 intervention execution.
- Sealed or confirmation evaluation.
- GPU/TPU/Kaggle/Colab jobs.
- Package installation or model download.
- Unrelated project test suites.
- Scientific power simulation.
- X6 or X7 experiments.

## 10. Google Colab notebooks

The package includes three self-contained Google Colab notebooks:

- `notebooks/x6_colab_readiness.ipynb` — reruns the frozen protocol/source closure, historical X1 gate, and 16 focused software tests; produces a Colab evidence ZIP and the correct `PREREQUISITE_BLOCKED` verdict.
- `notebooks/x6_colab_operator_gate.ipynb` — validates an optional future source-bound X6 operator bundle and refuses missing, invalid, or unapproved artifacts without training.
- `notebooks/x7_colab_gate.ipynb` — independently validates a future supported X6 receipt and fresh X7 follow-up bundle; it never claims dependence fall.

All three notebooks:

- Run on a normal Colab Python runtime without assuming a GPU or TPU.
- Do not use Kaggle, install packages, download checkpoints, mount Drive, clone repositories, train, or perform model inference.
- Embed the exact frozen source bundle and verify its canonical hashes after extraction.
- Treat expected scientific gate failures as structured blockers, not Python tracebacks.
- Set `execution_authorized=false`, `model_execution=false`, and `training=false` in their receipts.
- Compile as valid nbformat 4 notebooks and execute all six code cells successfully in clean local Python processes.

A future scientific training/inference notebook must be generated only after a valid X1 receipt, qualified parent, approved amendment, and source-bound operator artifacts exist. Writing a trainer now would fabricate readiness and would violate the frozen gate.

## 11. State separation

### Implemented

- Prospective protocol.
- X1 entry gate.
- Run/readiness/dataset/continuation/receipt interfaces.
- X6 and X7 software contracts.
- Focused tests.

### Verified

- Protocol canonical self-hash and source closure.
- Gate behavior on the preserved invalid X1 receipt.
- Fail-closed split, provenance, fairness, commitment, and receipt arithmetic.
- Model-free module import.

### Scientifically executed

Nothing.

### Experimentally supported

No internalization, transfer, retention benefit, or X7 dependence fall is experimentally supported by this assignment.

## 12. Bounded future runbook and compute

No wall-time estimate is fabricated without a qualified checkpoint and measured throughput. An authorized operator should:

1. Measure device throughput, optimizer-state size, checkpoint save/load time, and intervention cells/second.
2. Compute total cost from frozen updates, paired seeds, independent worlds, interventions, evaluations, and checkpoint count.
3. Include storage, interruption, fresh-process resume, and transfer custody budgets.
4. Use a fixed save cadence and a last-known-good predecessor.
5. Keep confirmation labels inaccessible until training and checkpoint selection are frozen.
6. Stop and preserve evidence on any identity, split, exposure, intervention, or continuation mismatch.

Likely scale remains a small authorized GPU experiment only after the stronger-checkpoint prerequisite is real. This package itself does not request or consume compute.

## Exact next action

The X-factor owner must review and version the proposed ordinary-control and numeric-threshold amendment, then direct an authorized operator to obtain a valid qualified X1-REAL-1 receipt and matching eligible checkpoint/continuation bundle. Do not train from the current blocked package.
