# Signac qualification and training plan

This plan uses increasing cost only after each preceding measurement works. It treats 100M as an experiment scale, not a presumed capability upgrade. The 100M run is not authorized by this document.

## Phase 0 — contract and CPU integrity

1. Instantiate `signac_100m.MODEL_SPEC` through `v5_model.initialize`.
2. Verify exact count **101,790,080**, one tied embedding, finite forward/backward, causal and packed-segment semantics, deterministic initialization, and parameter mutation under the production backend's update contract.
3. Serialize the model spec and hash into the run manifest. Reject stale model/code/tokenizer/data/evaluation identities.
4. Run `pytest tests/test_signac_100m.py tests/test_v5_model.py` and the parameter calculator.

Exit: every test passes; no model/data/hardware readiness is inferred from this phase.

## Phase 1 — data and evaluation qualification

The corpus must pass the existing dataset lifecycle to `RUNNABLE`, with source/license provenance, raw and transformed hashes, exact and near-dedup identities, source/dedup-cluster-disjoint train/dev/sealed splits, tokenizer identity, pack identity, contamination audit, and a qualified real-token count. Synthetic generated tasks are valid for controlled canaries when truth is executable; they are not a substitute for the qualified production mixture.

Citadel must independently mark the relevant capability evaluator ready. Keep the candidate-free primary and sealed firewall. Before any causal comparison, a multi-seed baseline must land in a prospectively frozen, non-floor/non-ceiling sensitivity band on the primary skill and show valid EOS. S5's formal NULLs stay unchanged; the corrected interpretation remains `INCONCLUSIVE_AT_ZERO_BASELINE`.

Exit: hash-verified runnable data and evaluation receipts; baseline capability gate passes with thresholds frozen before outcomes.

## Phase 2 — Kaggle target qualification (TPU)

The notebook first records Python, PyTorch, `torch_xla`, PJRT device type, process world size, global TPU device count, and per-device identity where exposed. Its bounded canaries measure cold first-update latency and exercise synthetic-token update/checkpoint plumbing for all three candidate geometries; they do not report a reliable peak-memory measurement. On target hardware, separately qualify in this order: BF16/FP32 numerical parity; real Signac forward/backward; one optimizer update and global clipping; 4096-context peak memory at planned microbatch; multi-replica reduction; checkpoint write, restart, restore and identical next update/batch; Kaggle Output durability and quota behavior. Preserve machine-readable receipts and a small run bundle.

Current status is **not qualified**: the V5 XLA adapter says `TPU_EVIDENCE_REQUIRED`; the current Kaggle science operators are GPU-specific. A device-discovery preflight is not a TPU model canary. If TPU qualification cannot be completed, label all GPU results GPU-only and do not infer TPU fit or parity.

Exit: every target canary passes from the branch's pinned commit with durable receipts and exact resume.

## Phase 3 — small capability-dose calibration

Only after Phases 1–2, use a low-cost control-only sweep to calibrate dose, learning rate/clipping, and runtime duration. Multiple fresh seeds must leave floor and ceiling, and candidate-free development behavior must be measurable. Do not change the tokenization or full output-space geometry in this calibration. This phase exists specifically to prevent another S5 floor-limited campaign.

Exit: preregistered sensitivity gate passes, train/dev split identities are audited, and outcome visibility rules are tested.

## Phase 4 — 100M-class research run

Preregister one scientific question at a time. First run the frozen core as a scale-transfer baseline against a matched smaller rung; then test any evidence-backed intervention prospectively. Bind runs to matched token exposure, compute reporting, fresh seeds, candidate-free metrics, orthogonal invariance families, sealed evaluation, and retention/recovery after acquisition. Report per-family outcomes, worst family, uncertainty, NLL, valid termination, FLOPs/tokens, clipping, optimizer state, throughput, and checkpoints.

The 20× token figure (**2,035,801,600** for M102) is only a generic planning prior. Select an actual token budget from qualified data supply, cost, measured throughput, and a preregistered decision that this run can resolve. Do not extend training merely because a generic rule says so.

Exit: independent bundle audit, completed preregistered endpoints, fresh replication for any promotion, and no leakage or resume-integrity failure.

## Promotion rule

Promotion requires candidate-free held-out transfer and sealed confirmation on preregistered metrics, valid answer termination, multiple seeds, and no disqualifying contamination, scorer, or resume issue. Loss reduction and parameter count alone never satisfy the gate. Report null, negative, mixed, floor-limited, and incomplete outcomes as such. No outcome from this plan alone licenses an AGI claim.
