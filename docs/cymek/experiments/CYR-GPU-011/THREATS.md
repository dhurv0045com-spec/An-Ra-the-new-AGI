# CYR-GPU-011 — THREATS TO VALIDITY

## Scientific threats

### Semantic underexposure
The central historical confound. V9/V10 did not approach ARK-002B's 1,152,000 semantic row presentations. V11 targets that same row box at any selected batch. If wall time truncates a stage, the verdict must use actual exposure. A production null below compact-G90 exposure cannot be called representation divergence.

### Seed variance / grokking variance
ARK-002B showed large seed variance: one seed sustained G90 around 12k updates while another was still climbing at 18k. V11 compact and production-primary are single-seed developmental subjects; only an opportunistic second production seed can support a replicated production label. No single-seed transition-time law is allowed.

### Compact bridge is not exact ARK reproduction
Cymek retains V5 GQA/QK norm, initialization, optimizer grouping, backend and CUDA precision behavior. Compact vocabulary/data/answer-prefix semantics/exposure are brought close to ARK-002B, but a null cannot identify which remaining difference caused divergence.

### Representation bridge changes more than vocabulary size
PRODUCTION_BRIDGE changes vocabulary/tokenization and returns to normal Cymek answer-sequence semantics; embedding parameter count also grows substantially. A compact-positive/production-negative result therefore implicates the production representation burden broadly, not tokenizer vocabulary size alone.

### Evaluation-set adaptation
DEV_CONTROLLER is used for transition control. To limit control-set overfit, final qualified G90 additionally requires >=.90 on the larger DEV_MEASUREMENT STANDARD set. SEALED_RESERVED never controls optimization or stage selection.

### Structural diagnostics are not broad reasoning
COMMUTED, LOCALITY, CARRY, TRIPLE_ADD, THREE_DIGIT and VERBAL probe narrow controlled transformations. They may reveal structural transfer or brittleness, but do not establish general reasoning, language understanding or AGI.

### Diagnostic multiple comparisons
Many structural metrics are reported. None may retrospectively become the primary endpoint. Threshold flags are descriptive/preregistered; no post-hoc winner selection across diagnostics.

### Answer-frequency / shortcut structure
The exact ARK-002B manifest preserves its original distribution. Result interpretation should compare train/heldout trajectories and per-digit behavior rather than assume all shortcuts are absent. Canonical pair overlap is zero, but other statistical regularities can remain.

### Commutation interpretation
Addition is mathematically commutative. COMMUTED performance measures invariance to operand order, not independent task generalization.

### Locality metric usability
Counterfactual relation consistency is meaningful only when both generated answers are numeric. V11 reports numeric-usable fraction alongside relation consistency so nonnumeric outputs cannot silently inflate the metric.

### Early stopping on G90
A stage stops after sustained controller G90, so different subjects can receive different final exposure. G90 timing is itself an endpoint; comparisons must use exposure-at-G90 and final measurement support rather than raw final exposure alone.

## Engineering threats

### CUDA/precision difference from Arkenstone
Arkenstone micro studies and Cymek CUDA production-backend semantics are not identical precision environments. This is an intentional bridge difference and must remain in limitations.

### Calibration optimism
Very short calibration may not perfectly predict long-run training/evaluation cost. Resolver projections are advisory, not a feasibility gate. The monotonic hard deadline is authoritative.

### Evaluation overhead
Candidate-free generation and structural batteries can consume meaningful wall time. Calibration includes candidate-free generation; resolver reserves additional overhead. Actual wall receipts remain authoritative.

### Small-batch update count
Batch32/16 may require 36k/72k updates to target the same semantic exposure. This changes optimizer-update count even when row exposure matches. Therefore batch-size effects and update-count effects remain entangled if batch64 is not feasible.

### Scoped monkeypatch leakage
The research entry temporarily adapts compact rendering, optimizer-call compatibility, semantic max updates and final decision. Context managers restore original functions. Tests must verify restoration. Production code itself is not modified for these research accommodations.

### Compact BOS semantics drift
ARK-002B trains a supervised answer-prefix BOS. V11 compact bridge reproduces it mechanically and tests exact token/eligibility layout. Production bridge deliberately does not.

### Drive/runtime interruption
The runner writes progress/checkpoints and packages failures, but CYR-GPU-011 is primarily a one-shot session rather than a fully transactional campaign resume system. A Colab termination can still truncate a stage. Drive evidence already written remains useful; any missing stage is incomplete, not negative evidence.

### Bundle/receipt mismatch
Cell 2 verifies the bundle SHA256 stored in `campaign_receipt.json`. Any mismatch blocks handoff.

## Interpretation prohibitions

Do not infer from CYR-GPU-011 alone:

- that Cymek broadly reasons;
- that arithmetic grokking transfers to language or agents;
- that compact-vocabulary behavior predicts production-scale behavior;
- that a production null is caused specifically by tokenizer vocabulary size;
- that GPU performance predicts TPU performance;
- that any result authorizes PRE500M or 500M training;
- that ARK-017 or ARK-018 mechanisms are validated before their own raw results exist.
