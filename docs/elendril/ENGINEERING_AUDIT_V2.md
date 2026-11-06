# Engineering audit V2

| Claim | Evidence and status |
| --- | --- |
| ELD-DEV-003 recorded results are reproducible from prediction receipts | Verified all 13 receipt/resource hashes and independently rescored all 120 predictions snapshots. Replay used original source `acd4b115d2075340a3db70131cea02d5f1b74c4f`. |
| ELD-DEV-003 hybrid fresh replay | Seed 7 exactly reproduced all raw prediction hashes, all stress panels and step telemetry. Endpoint was 44/48. |
| ELD-DEV-003 dense fresh replay | Seed 7 produced 0/48 versus the recorded 1/48, with identical identities and indices. Initial losses differed by update 1. Residual down-projection initialization used ambient RNG after the seeded fork closed. The repair moves it inside the fork, versions the architecture identity, and tests two ambient RNG states. |
| Local checkpoint publication | An OS-owned lineage lock now spans publication, rotation and milestone writes. A test pauses one writer inside generation 1, refuses an independent generation 2 writer and pruning, then restores generation 1. Existing crash and corruption tests remain in scope. |
| CUDA and remote durability | CUDA branch and notebook canary are implemented. The expanded profile passed a local CPU smoke. CUDA numerical parity, Drive behavior and mounted multi-host locking remain unverified. The mounted mirror requires one active publisher per lineage. |
| ELD-DEV-004 | Controlled development evidence for two-edge composition and first-read mediation. Alias and three-hop stresses failed. See exact source and endpoints in [results](ELD_DEV_004_RESULTS.md). |

## Architecture and numerical scope

The inherited dense core retains tied weights, QK normalization, RoPE resets, packed causal attention, AdamW and answer/EOS loss. The symbolic codec is reversible and remains separate from the production tokenizer. The read path excludes boundary IDs and uses only strictly earlier tokens in the current segment. Duplicate IDs contribute the sum of their attention mass. No-source positions return the dense distribution. The masked softmax uses a finite minimum, then zeroes forbidden slots. The log mixture clamps copy mass at a dtype-dependent positive minimum so log and gradient paths stay finite at unavailable symbols. Tiny clamped mass makes the result approximately normalized within FP32 tolerance.

Each read materializes a B × L × vocabulary distribution, while lexical matching currently materializes B × L² × window² intermediates. At B=16, L=4096, V=24576 one FP32 copy distribution alone takes 6 GiB. The lexical Boolean match tensor at window 3 takes 2.25 GiB before float intermediates. Scaling this operator requires a chunked or sparse probability path, memory accounting and independent FP32/BF16 parity. The notebook stays at symbolic vocabulary and short context. No production tokenizer or architecture promotion is implied.

Historical context still applies. Cymek's denominator and physical-vocabulary studies ruled out simple universal fixes. Arkenstone's later skill failed to form, which limited retention inference. Triquetra and BRAMASTRA showed how recency and query blindness inflate per-answer scores. Citadel showed split leakage, and Gandiva's failed campaign established why coverage, prompt admission and durable payloads must be checked separately. See the [evidence crosswalk](RESEARCH_EVIDENCE_MATRIX.md).

The final focused CPU regression passed 159 tests and seven subtests. The independent two-process two-read restart matched the six state component digests and the prediction digest. The full production campaign suite and live remote CI are outside the local verification.
