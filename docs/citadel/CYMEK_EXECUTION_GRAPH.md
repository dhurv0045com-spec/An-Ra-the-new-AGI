# CYMEK_EXECUTION_GRAPH.md

The real V5 execution path as it exists on `origin/cymek` @ `26a61f6242e9e5c1d1b028b4f8c3c7d26ac0fdc6`
(audited 2026-09-03). Every transition is classified:

```
IMPLEMENTED          code exists
TESTED               unit/integration tests cover it
RECEIPTED            a committed artifact records an execution of it
EXECUTED_END_TO_END  a committed receipt records it running inside a continuous real pipeline
DECLARED_ONLY        documents/specs declare it; code does not exist
MISSING              nothing exists
BLOCKED              cannot run for an external reason recorded by the branch itself
```

Rule applied (central operating rule): **code + reproducible execution wins over documentation.**
Cymek's own `artifacts/cymek/evidence_ledger.json` contains stale numbers from a superseded run
and one claim contradicted by its own code; where they disagree, the committed execution receipts
and the code are treated as truth and the discrepancies are recorded below.

```
raw source / generated examples          [see DATA row]
        ↓ provenance                     IMPLEMENTED + TESTED + RECEIPTED (48-file repo corpus only)
tokenizer                                IMPLEMENTED + TESTED + RECEIPTED (real 24,576 artifact, hash-verified at load)
        ↓ dataset mixture                IMPLEMENTED (allocation math) / NOT ENFORCED in consumed path
sampler                                  IMPLEMENTED + TESTED (v5_data.pack.sampler_order) / NEVER EXECUTED
        ↓ packing                        IMPLEMENTED + TESTED + RECEIPTED (true multi-segment stream-fill)
cursor                                   IMPLEMENTED + TESTED + RECEIPTED (coordinate advance; real-token ledger)
        ↓ batch                          IMPLEMENTED + TESTED (v5_data.batch.microbatch) / NEVER EXECUTED
model                                    IMPLEMENTED + TESTED + RECEIPTED (P35 CUDA canary; V5-A construction canary)
        ↓ objective                      IMPLEMENTED + TESTED + RECEIPTED (CE only; query-swap disabled by design)
backpropagation                          IMPLEMENTED + TESTED + RECEIPTED (autograd inside backend)
        ↓ optimizer                      IMPLEMENTED + TESTED + RECEIPTED (AdamW, tied-embedding ownership)
schedule                                 IMPLEMENTED + TESTED (canonical 5B WSD) / EXECUTED variant only (bounded warmup)
        ↓ checkpoint                     IMPLEMENTED + TESTED + RECEIPTED (content-addressed, single-writer fence)
evaluation                               IMPLEMENTED + TESTED + RECEIPTED (checkpoint-backed adapter, 2 hardcoded tasks)
        ↓ promotion gates                IMPLEMENTED + TESTED (fail-closed repair) / NEVER FED REAL EVIDENCE
remote artifact / accepted model         DECLARED_ONLY (contracts; nothing ever submitted)
```

## Stage-by-stage classification

| # | Stage | Class | Evidence |
|---|---|---|---|
| 1 | Raw data source | PARTIAL / EXECUTED (wrong source) | Only data producer wired into the executed path is `v5_training/miniature.py:_load_corpus` — 48 tracked repo text files, ≤500 KB, ~74K tokens (`miniature_receipt.json`: 48 documents, 73,750 real tokens). The e0 cognition generators exist (`e0_cognition/training_generators.py`, `e0-cognition evaluation_generators.py`) but **nothing in the production path calls them**; `tokens_by_source` in every committed receipt is `{natural, code_math_formal}` only. No loader for any external corpus format exists in `v5_data`. |
| 2 | Provenance | RECEIPTED (with holes) | Manifest binds source_id/authorization/raw_sha256; tokenizer hash-verified against its tournament receipt at load (`miniature.py:77-99`). Holes: `_load_corpus` computes per-file byte hashes then discards them (`del digest`); manifest hashes decoded text, not bytes. Miniature receipt lacks device/wall-time/timestamp and names a `source_commit` whose tree does not contain the committed receipt. |
| 3 | Tokenizer | RECEIPTED | Frozen 24,576-entry byte-BPE artifact (`artifacts/e1/local_tournament/tokenizer-24576.json.gz`, sha `97e12db6…`), identity roundtrip + zero-unknown enforced, used by all three executed runs. Still `PROVISIONAL_CANDIDATE_E1_IDENTITY_REQUIRED` in the spec — adequate for development, not frozen for production. |
| 4 | Mixture | IMPLEMENTED / NOT ENFORCED | `v5_data/mixture.py` allocates 0.65/0.20/0.15 + nine family fractions exactly; `DataMixture.assert_valid` enforces sums. But nothing between sampler and update enforces mixture at consumption time, and no executed run consumed more than one intended family. Cymek ledger's "68.7%/31.3% actually consumed" is a stale number from a superseded run (committed receipt: 49.7%/50.3% natural/code). |
| 5 | Sampler | IMPLEMENTED + TESTED / NEVER EXECUTED | `v5_data/pack.py:sampler_order` (deterministic over run_seed/epoch/shard hashes) + `tests/test_v5_data_pipeline.py`. Referenced by tests only; the miniature and both canaries hand-slice update windows (`miniature_run.py:151-185`). |
| 6 | Packing | RECEIPTED | `pack_documents` true multi-segment stream-fill packing (BOS/EOS chunking, bucket 512–4096, exact per-source ledger, cross-checked triple-way). Miniature: pack efficiency 0.96673, 32 sequences, 64 segments. |
| 7 | Cursor | RECEIPTED | Coordinate cursor with exact real-token ledger; `microbatch` cross-checks consumption against `cursor.advance` internally; persisted `CursorState` validated against `TrainingState`. |
| 8 | Batch assembly | IMPLEMENTED + TESTED / NEVER EXECUTED | `v5_data/batch.py:microbatch` — cursor-addressed consecutive-sequence assembly with triple-checked ledger. Tests only (9 tests); bypassed by every executed run. **This certification gap is what Citadel's one-update cert targets.** |
| 9 | Model | RECEIPTED | P35 CUDA canary: exact ladder recipe (16L×384, 6Q/3KV, FFN 1024), 35,411,328 executable params vs 35,414,400 receipt (−3,072 recorded as negative evidence); V5-A canary: exact 250,216,960 params, single tied embedding, 4.94 GiB peak at 512-token microbatch, ~45 tok/s. |
| 10 | Objective | RECEIPTED | `causal_lm_loss` only: shift-once CE, BOS/PAD excluded, EOS supervised, segment-boundary masking, replica-global mean. Query-swap contrastive implemented, `enabled=False` by design, gated to E3 Phase B — **weight is exactly zero in every executed run**. |
| 11 | Backprop + optimizer | RECEIPTED (mechanically certified) | `ProductionTrainingBackend.step`: backward → clip 1.0 (pre-norm from clip call, post-norm recomputed independently) → AdamW → `certify_real_update` raises on PARAMETERS_UNCHANGED / NO_GRADIENT / OPTIMIZER_NOT_STEPPED / MOMENTS_MISSING / SCHEDULE_MISMATCH / TIED_WEIGHT_BROKEN / stale optimizer ownership. Adversarial tests cover a stale-optimizer attack and exact resume. |
| 12 | Schedule | PARTIAL | Canonical `v5_training/schedule.lr_at` (WSD 50M/4.5B/5B) unit-tested only. Every executed run binds `bounded_warmup_schedule` (constant 3e-4, `warmup_tokens=0`). |
| 13 | Checkpoint / resume | RECEIPTED | Content-addressed store, 7-component manifest, single-writer fence; miniature and both canaries record `resume_parameter_hash_equal: true` with hash-compared before/after states. |
| 14 | Evaluation | RECEIPTED (toy) | `CheckpointBackedV5Adapter` (suffix-only candidate scoring with prefix-verification, greedy generation, constrained selection) + gold firewall (`VisibleTask` structurally cannot carry gold). Executed on the miniature checkpoint with **2 hardcoded synthetic tasks**; one wrong, one correct, free generation = 16 unk tokens (honestly recorded). Fresh split untouched; no sealed fixtures exist. |
| 15 | Promotion gates | IMPLEMENTED + TESTED / NO REAL INPUT | Gates repaired to fail-closed (`REQUIRED_DOSSIER_ENTRIES`, empty-probe rejection, closed inventory, Wilson LCB where specified). All 9 tests use synthetic dossiers. No real evaluation dossier has ever been evaluated; `main_training_authorized` is hardcoded `False`. |
| 16 | Remote jobs | DECLARED_ONLY | Hash-bound envelopes + collection CLI; no submission has ever occurred; credentials/storage out of scope. |
| 17 | CI | DECLARED_ONLY | `.github/workflows/cymek.yml` runs pytest + reproduces three frozen contract receipts, but does not run the miniature/canaries, does not reproduce any `artifacts/cymek/*` execution receipt, and does not verify `test_receipt.json` (a local self-report, tested head `6a2b64b`, 237 passed). |

## Answer to the primary audit question (§5)

> Can Cymek, from an empty run directory, consume a real dataset through its intended data
> contracts, train the intended V5 model for ≥1 genuine optimizer update, save a reproducible
> checkpoint, reload it, evaluate it through the intended interface, and produce a valid
> promotion decision artifact?

**PARTIALLY — demonstrated for a miniature/bounded configuration on the repo's own files;
falsified for the intended configuration.** The full chain minus promotion *has executed
end-to-end once* (miniature receipt) and the update chain is mechanically certified — but the
"real dataset" in that demonstration is Cymek's own source code, the model is a 2-layer toy, the
mixture/sampler/microbatch modules were bypassed, evaluation used 2 hardcoded tasks, and no
promotion decision artifact can be produced (gates never fed real evidence; authorization
hardcoded off). The **first broken edge for the intended system is stage 1**: there is no wired
producer of the intended training data (no cognition-data path, no corpus loader for natural
data). The second broken edge was stage 8 (batch assembly never executed in the pipeline) —
**closed by Citadel's independent certification on 2026-09-03**:

> **ONE-UPDATE CERTIFICATION: PASS — twice, on two devices** (`cymek_receipts/ONE_UPDATE.json`,
> `TEN_UPDATE.json`, `ONE_UPDATE.cpu_mini.json`, `TEN_UPDATE.cpu_mini.json`).
> Real generated cognition documents (`e0-train/0.2.0`, 1,200 examples) → data manifest
> (dedup/cluster-split/contamination scan) → true multi-segment packing → **sampler order →
> cursor-addressed microbatch** (the previously-never-executed edge) → certified production
> update → content-addressed checkpoint → exact restore (`resume_hash_equal: true`).
>
> - **CUDA / exact P35 recipe** (16L×384, 35.4M params, RTX 4050, torch 2.14.0+cu126, bf16
>   autocast): update loss 10.1227, post-clip grad norm 0.99999988 — *within* the 1e-6 gate,
>   confirming the gate is CUDA-calibrated; ten-update loss 10.123 → 7.136 (2,514 tok/s).
> - **CPU / miniature recipe** (2L×64, 1.65M params, fp32): update loss 10.0956, post-clip
>   norm 0.99999934; ten-update loss 10.096 → 9.689 (2,360 tok/s).
> - The **P35-on-CPU fp32** attempt was rejected by the clip gate (post-clip 1.00000286 > 1+1e-6):
>   the gate's tolerance is device-narrow (E3 in the bottleneck ranking). Measured on both sides;
>   not retried to a pass.

## Device/throughput reality (from Cymek's own committed receipts)

- P35: ~1,800 tok/s, 3.9 GiB peak, 3 updates declining loss 10.166→9.536 (RTX 4050 Laptop, bf16 autocast + fp32 master).
- V5-A (250M): ~45 tok/s at 512-token microbatch — a 5B-token run is years on this GPU; 4,096-token microbatch would not fit 6 GiB. Topology blocker (TPU/XLA) is real, not bureaucratic.
- Known CPU finding (Citadel, 2026-09-03): `certify_real_update`'s post-clip norm tolerance (1e-6, calibrated on CUDA bf16) **fails closed** on CPU fp32 when clipping activates (measured post-clip norm 1.00000286 > 1.0+1e-6 from fp32 re-accumulation noise) — a legitimate update is rejected. Gate correctness is good behavior; the tolerance is device-narrow and must be widened or device-calibrated before CPU-based certification/CI can run the backend.
