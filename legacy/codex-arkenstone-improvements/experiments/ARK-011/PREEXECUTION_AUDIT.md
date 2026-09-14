# ARK-011 PRE-EXECUTION AUDIT

Status: **STATICALLY REVIEWED; GPU EXECUTION NOT YET PERFORMED.**

## Bound artifacts

- scientific plan commit: `1d5fc08000614e740b7c93d87e8d233c903bcddf`
- execution addendum commit: `70fa2764fc0df79d4bf4b0b16e6620bc87278aa9`
- pinned runner commit: `baec1687ed43d4c0dc420b4f329df5cf266d8645`
- runner Git blob: `b410ced065f39164010715fc4451468b20f7100c`
- Colab launcher commit: `322e78189deae9245f298823b8a90732cd2b6c57`

The Colab notebook checks out the runner commit in detached-HEAD mode and refuses an unexpected HEAD.

## Plan/implementation correspondence

### Controller/evaluation firewall

The training-state functions are structurally separated:

- `acquire(seed, train, control, ...)` receives CONTROL but no SEALED argument.
- `run_to_threshold(..., train, control, ...)` receives CONTROL but no SEALED argument and is used for both instability detection and recovery detection.
- SEALED first enters execution only after CONTROL recovery has already fired, when the recovery snapshot is evaluated and then forked.
- Post-fork retention has no further adaptive LR changes, so reporting SEALED throughout that phase cannot alter the controller.

This is stronger than relying on a comment saying not to inspect SEALED; the phase interfaces omit it.

### Continuation identity

For every preregistered order seed the runner materializes 16,000 batches once and binds the sequence by SHA256.

Maximum consumption is exactly bounded:

- instability search: <= 6,000 batches;
- recovery search: <= 4,000 additional batches;
- post-recovery retention: 6,000 additional batches.

At each trigger the next phase begins at the first unused batch. Both retention arms use the same recovery snapshot and the same unused indices.

### Fork identity

Snapshots include:

- model state;
- optimizer state;
- CPU RNG;
- all CUDA RNG states.

The LOW arm changes only optimizer LR after loading the shared recovery snapshot. The smoke test restores two equal-LR forks from the same snapshot, applies the same minibatch, and requires equal resulting parameter hashes before long training is allowed.

### Provenance and failure behavior

Every JSON written by the runner adds:

- plan SHA;
- addendum SHA;
- checked-out runner commit SHA;
- runtime runner-file SHA256;
- device;
- torch version;
- canonical receipt SHA256.

Full execution catches exceptions, writes `FAILURE_RECEIPT.json`, and packages all partial JSON evidence in `finally`.

## Known risks / limitations before execution

1. **CUDA exactness smoke risk.** If the T4/PyTorch SDPA path is nondeterministic enough that equal restored forks do not produce equal hashes, the smoke test will fail before scientific training. Do not bypass the assertion after seeing results; any compatibility change must be a new pre-execution source-fix commit and notebook pin.
2. **Smaller OOD halves.** CONTROL/SEALED partitioning reduces sample size and may make the 0.90 threshold noisier than earlier full-holdout experiments. The split is frozen before execution and cannot be tuned afterward.
3. **CONTROL and SEALED can disagree.** A CONTROL recovery is not assumed to imply SEALED recovery. All controller-recovered forks execute; `sealed_at_recovery_fork >= 0.90` is only a preregistered primary-analysis qualification and is reported for every event.
4. **Clustered opportunities.** Twelve order opportunities come from three independent acquisition checkpoints, not twelve independent models.
5. **Low-LR plasticity cost remains open.** ARK-011 tests recurrent stability after recovery, not ability to acquire a new skill while consolidated.
6. **No production claim.** Cymek's WSD/token schedule is not modified. A positive Micro T2 result still requires transfer and plasticity-cost experiments.

## Execution gate

Long training is authorized only after the pinned Colab notebook prints:

`GPU SMOKE TEST PASS — ARK-011 is safe to start`

Until a result receipt exists, M-010 remains `TESTING / PREREGISTERED`, not demonstrated.
