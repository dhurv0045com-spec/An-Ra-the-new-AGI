# Owner Runbook

## Scope

This runbook reproduces DG0 on a standard Python installation. It uses no package download, dataset, pretrained model, GPU, TPU, sealed evaluation, or optimizer update. The repository's BRAMASTRA handoff also prohibits local optimizer updates; DG0 honors that boundary.

## Reproduce DG0

From the isolated worktree:

```text
python -m pytest tests/test_discovery_gym.py -q --tb=short
python -m compileall -q discovery_gym
python -m discovery_gym dg0 --output <new-empty-directory> --seed 1701 --worlds-per-split 6 --budget 8
```

Use a new output directory for every run. The CLI refuses to overwrite an existing directory. The deterministic smoke is expected to produce a PASS audit and 192 episode records across eight policy identities. Preserve the complete output directory, including failed or negative runs.

## Artifacts

- `suite_manifest.json`: evaluator-owned world and split identities, including the public intervention-anchor partition;
- `public_worlds.json`: safe public descriptors;
- `protocol.json`: frozen DG0 protocol and hash;
- `public_transcripts.json`: policy-visible action commitments and consequences;
- `evaluator_run.json`: evaluator-side recomputation and metrics;
- `audit.json`: leakage, legality, split, shortcut, and confirmation-firewall checks;
- `result_receipt.json`: source, generator, evaluator, protocol, suite, run, and audit hashes;
- `checkpoint_identity.json`: explicit `NOT_CREATED_DG0` identity.

Do not treat the public transcript as an evaluator record. Do not give `evaluator_run.json`, hidden world records, or oracle actions to a learner.

## Inspect a result

Check `audit.json` first. Then verify:

1. all critical checks are `PASS`;
2. the oracle is marked `uses_hidden_truth=true` and is reported separately;
3. all policies have the same declared budget;
4. rejected and repeated actions appear in the cost ledger;
5. the intervention anchor's one-variable neighborhood is disjoint from target candidates;
6. the information-gain baseline is compared, not credited to a model;
7. the result receipt hashes match the files being reviewed;
8. `checkpoint_identity.json` says no checkpoint was created.

## DG1 preparation

Before any learner run:

1. inspect live CPU, RAM, GPU, disk, and Python environment;
2. estimate wall time, memory, storage, and accelerator qualification separately;
3. choose one small architecture and one bounded hypothesis;
4. record initialization, tokenizer/encoding, training data, optimizer, step budget, checkpoint, and resume state;
5. freeze the development protocol and baseline suite;
6. run development only, compare to transparent controls, and preserve negative results;
7. keep confirmation and transfer hashes inaccessible until the protocol is frozen.

No hardware claim follows from CPU DG0. No scientific result follows from a future build-only verification.

## Failure handling

A failed audit, missing source hash, invalid episode, cost mismatch, leakage, or protocol drift invalidates the affected run. Preserve the failure artifact and create a new run identity. Never clean, overwrite, or silently repair a result directory.
