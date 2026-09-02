# V5 execution sequence

## 0. Reproduce the frozen package

```powershell
python -m unittest discover -s tests
python -m v5_contracts.import_boundaries
python -m v5_contracts.training_spec --output artifacts/v5/training_spec_v1.json
python -m v5_contracts.launch_readiness --output artifacts/v5/launch_readiness.json
```

Expected current status: `READY_FOR_PRELAUNCH_EXPERIMENTS`, with
`main_training_authorized=false`. This checks the evidence inventory, not a
production trainer. E1–E5 learning runners still require implementation and
the representative data and compute described below.

## 1. E1 — tokenizer and corpus identity

Run the static audit and the matched P35 16k/24k/32k tournament on the declared,
hash-bound corpus. Promote a tokenizer only from raw-byte/FLOP-matched evidence.
Record the immutable receipt path and SHA-256 in `LAUNCH_GATES.json`.

## 2. E2 — P35 architecture

Run the consistent 2:1-GQA P35 shape/context comparison. Preserve the existing
MHA, 3:1-GQA, QK, precision, RoPE, and kernel receipts as mechanism evidence,
not as a learned-quality result. Replicate the top two learned arms.

## 3. E3 — cognition data and objective

Compare 5%, 15%, and 30% verified-cognition mixtures under CE only. Raw-Core,
fresh candidate-free, natural-transfer, shortcut, OOD, and worst-family gates
decide promotion. Query-swap is a separate matched-compute experiment and may
not silently enter the launch objective.

## 4. E4 — optimization and curriculum

Measure LR/batch/WSD stability at the P35 winner. Require exact resume across a
checkpoint boundary, stable FP32 moments, finite updates, substrate retention,
and no Tier-1 worst-family collapse.

## 5. E5 — M102 transfer

Train the winning recipe and a strong CE/general-data control at 600M–1B tokens.
Require two winning-recipe seeds, fresh natural transfer, compatible P35→M102
effect direction, and exact checkpoint restoration.

## 6. E6 — target and custody

On the declared TPU/XLA topology, run target preflight, exact model/update,
collective, throughput, failure-injection, and remote upload/redownload/restore
canaries. Bind tokenizer, source, data, pack, runtime, topology, checkpoint, and
sealed-evaluation SHA-256 identities.

## 7. Freeze and launch

Only after E1–E6 are `PASS`, fill every external identity and regenerate the
readiness receipt. `READY_FOR_FREEZE_REVIEW` means only that the inventory is
complete. Independent review must validate the experiments and custody, not
just hashes. A production trainer and signed-launch verifier remain to be
implemented. This command never grants main-run authorization.
