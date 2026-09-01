# ESOES Status

Updated: 2026-09-01

Phase: **V5-A implementation candidate v1.0 frozen; scientific launch blocked**

Canonical research blueprint: [`V5_MASTER_BLUEPRINT.md`](V5_MASTER_BLUEPRINT.md)

Canonical code-facing specification: [`V5_TRAINING_SPEC_v1.0.md`](V5_TRAINING_SPEC_v1.0.md)
Executable receipt: `artifacts/v5/training_spec_v1.json`

Large V5 training authorized: **NO**

The code-facing candidate now fixes every Core, tokenizer-interface, data-mixture,
cognition-family, packing, objective, optimizer, schedule, TPU-topology,
checkpoint, decoding, promotion, and abort constant. The center is a
250,216,960-parameter dense 26×896 decoder with 14Q/7KV 2:1 GQA, FFN 2368,
native 4k full attention, affine QK norm, a 24,576 byte-BPE interface, CE-only
training, 5B real tokens, and a 65/20/15 natural/code-cognition mixture. Frozen
means implementations may not guess or drift; it does not waive E1–E5 or target
evidence.

The cognition slice is token-complete across identity/copy, binding, state,
interference retrieval, composition, counterfactual sensitivity, held-out rule
induction, missing information, and faithful realization. Families and
difficulty are uniformly interleaved. Query-swap and trace objectives are off
(`lambda=0`) unless a separately frozen, FLOP-matched E3 result earns a new spec.

Scorer status: **FAIL DEVELOPMENT / PRODUCTION MODE NULL**. Fixture v1 was
invalidated before powered execution because surface family revealed hidden
role. Corrected schema 2 then ran all 15 CUDA development cells in 0.07574 GPU
hours. DC-PMI and contextual calibration both selected the fewest-token role
100% for every tokenizer. Fresh fixtures remain untouched; CPU parity was
stopped because it cannot rescue a failed bias gate.

Local P35 update/resume, checkpoint transaction, immutable CAS, rank schema,
runner fencing, BF16 parity, RoPE, QK-norm, and cursor canaries pass their scoped
claims. The complete local contract suite is **140 tests passing** on the
framework-neutral environment (three PyTorch-only optimizer tests skipped there
and separately passed in the CUDA environment); import
boundaries, compile checks, deterministic spec reproduction, and diff checks
also pass. TPU/XLA remains explicitly blocked on the current host.

Remaining launch blockers:

- real E1 tokenizer/corpus identity;
- P35 learned 2:1 architecture and E3 CE-mixture evidence;
- E4 recipe comparison and M102 two-seed fresh replication;
- power-sized family/Wilson evaluator and sealed custody;
- signed source/data/pack manifests;
- target TPU model/optimizer/collective/throughput/resume pass;
- remote upload/redownload/clean-restore receipt.

Next action: implement the v1.0 interfaces and run the smallest learned P35
CE-only mixture test. Do not run the 250M/5B model until every blocker is closed.
