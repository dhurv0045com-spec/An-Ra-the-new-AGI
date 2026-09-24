# Signac qualification and training plan

This plan uses increasing cost only after each preceding measurement works. It treats 100M as an experiment scale, not a presumed capability upgrade. The 100M run is not authorized by this document.

## Phase 0 — contract and CPU integrity

1. Instantiate `signac_100m.MODEL_SPEC` through `v5_model.initialize`.
2. Verify exact count **101,790,080**, one tied embedding, finite forward/backward, causal and packed-segment semantics, deterministic initialization, and parameter mutation under the production backend's update contract.
3. Serialize the model spec and hash into the run manifest. Reject stale model/code/tokenizer/data/evaluation identities.
4. Run `pytest tests/test_signac_100m.py tests/test_v5_model.py` and the parameter calculator.

Exit: every test passes; no model/data/hardware readiness is inferred from this phase.

## Phase 1 — data and evaluation qualification

The corpus must pass the existing dataset lifecycle to `RUNNABLE`, with source/license provenance, raw and transformed hashes, exact and near-dedup identities, source/dedup-cluster-disjoint train/dev/sealed splits, tokenizer identity, pack identity, contamination audit, and a qualified real-token count. Synthetic generated tasks are valid for controlled canaries when truth is executable; they are not a substitute for the qualified production mixture. Keep `corpus_seed` fixed across matched runs; choose `sampler_seed` explicitly; vary the model/training RNG independently. The Phase-One helper now keeps the generated corpus and pack manifest stable across run labels and evaluation-suite seeds, so independent training checkpoints can share exact data identities.

The development generator now covers all nine families in the repository's frozen cognition contract: identity copy, query binding, semantic state, interference retrieval, relational composition, counterfactual sensitivity, held-out rule induction, missing information, and faithful realization. It reads the frozen within-cognition shares, apportions small document counts deterministically with every family represented, and binds each synthetic source to its subfamily in the existing family-segregated packer. With enough examples for the family grids, exercises span the specified binding-cardinality, distractor-dose × realizable-position, state-variable/update/query, path-hop, and rule-demonstration dimensions. The interference schedule decouples dose from position; dose 2 has only quartiles 1–3, while doses 4/8/16/32 cover all four. Semantic-state queries cycle through latest, intermediate, rollback, and same-minute precedence semantics. Natural and semi-natural language surfaces are crossed within each family. Prompts expose only context and query; the target is the answer, while graphs and counterfactual metadata remain outside the model input. Versioned training-surface manifests record the observed cognition axes.

This is a generator and pack-path contract, not evidence of acquired cognition. The minimum-one-document rule distorts document-count percentages in small development surfaces; the frozen token-deficit scheduler, not document counts, enforces training shares. The generated rule task withholds an input example, but does not establish generalization to a structurally held-out rule. The RSI embeds a hash-bound nine-family coverage report: eight families have directly attributable evaluation tasks across seven axes; the evaluator isolates a generated counterfactual-premise/relevant-fact pair but does not match the training family's explicit intervention operation; interference retrieval has a dedicated candidate-free evaluation grid across distractor doses 0/2/4/8/16/32 and realizable context-position quartiles; and faithful realization has a narrow exact payload/revision format task with one relevant-fact pair per generated group. The interference grid is a sparse stress diagnostic rather than powered evidence, and the realization task does not cover broader formats or long-form fidelity. These measurements describe availability, not training effects, so RSI remains diagnostic-only and always abstains from changing the recipe.

Citadel must independently mark the relevant capability evaluator ready. Keep the candidate-free primary and sealed firewall. Before any causal comparison, a multi-seed baseline must land in a prospectively frozen, non-floor/non-ceiling sensitivity band on the primary skill and show valid EOS. S5's formal NULLs stay unchanged; the corrected interpretation remains `INCONCLUSIVE_AT_ZERO_BASELINE`.

Exit: hash-verified runnable data and evaluation receipts; an externally audited, cluster-aware baseline analysis passes with thresholds frozen before outcomes. The current `signac_100m/phase1_eval.py` summaries are development diagnostics and cannot satisfy this exit condition.

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
