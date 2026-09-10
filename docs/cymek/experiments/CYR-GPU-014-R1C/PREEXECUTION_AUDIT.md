# CYR-GPU-014 / R1C — PREEXECUTION AUDIT

**Audit state:** STATIC REVIEW COMPLETE / OPERATOR CUDA PREFLIGHT STILL REQUIRED  
**Scientific result:** NOT EXECUTED  
**Frozen scientific executable:** `2a71cea10ebb7a231834b6b112c49e268e9631a5`

## Question audited

R1C is intended to move beyond the R1/R1B vocabulary response map and test a mechanism: with the **physical V24576 tied matrix fixed**, does changing only training-time inactive-class participation / softmax competition alter structural and functional capability formation?

## Static design findings

The final pre-execution design closes several important confounds before any R1C scientific outcome exists:

1. **Physical model is fixed.** Every arm is a 4.130688M-parameter V24576 Cymek V5 RESEARCH_SMALL model. The treatment changes logits only during training.
2. **Matched initialization is explicit.** Within a seed, all six arms must have identical full model bytes; the CUDA preflight constructs and hashes them.
3. **Evaluation is unrestricted by default.** Full-vocabulary candidate-free generation remains the functional endpoint. Training masks cannot make inference easier silently.
4. **Structural vs output-calibration failure is separated prospectively.** The same DEV_MEASUREMENT set is also evaluated with an explicitly labelled active-only candidate diagnostic. The active-only score cannot authorize functional rescue.
5. **Primary arm is not selected from R1C outcomes.** MASK_4096 is frozen because R1/R1B made it the most stable prior intermediate candidate. MASK_8192/MASK_16384 are secondary dose levels and cannot rescue a failed primary pair post hoc.
6. **Aggregate mass gets its own control.** OFFSET_EQ4096 keeps all 24,576 rows participating but reduces inactive logits by the analytically frozen equal-logit offset. This distinguishes a simple partition-mass explanation from literal row/candidate competition.
7. **Trajectory, not endpoint alone, is primary evidence.** Formation AUC and sustained G50 are frozen because R1B showed emergence/collapse dynamics.
8. **Mechanism probes are read-only by contract.** Probability mass, margins, entropy, hidden norm, gradient partitions and counterfactual core-gradient cosines are measured at frozen checkpoints. The runner compares model+optimizer fingerprints before/after each diagnostic.
9. **Large campaign is not weakened for one runtime.** 4 seeds × 6 arms × 3000 updates = 72,000 updates. Runtime calibration estimates session count only. The fixed campaign can span multiple 330-minute T4 sessions through exact Drive checkpoints.
10. **Exact-resume has an executable CUDA gate.** Before scientific updates, 10 uninterrupted MASK4096 updates must byte-match 5 updates + save/load + 5 updates in model/optimizer hashes and real-token counter.

## Static code review

Frozen code paths:

- mechanism core: `v5_experiments/cyr_gpu014_r1c.py`
- CUDA runner: `anra_v5/cyr_gpu014_r1c_run.py`
- pure tests: `tests/test_v5_cyr_gpu014_r1c.py`
- plan: `docs/cymek/experiments/CYR-GPU-014-R1C/PLAN.md`

The launcher also rechecks inherited R1/R1B contracts and exact Git blob identities before it permits the CUDA preflight.

The pure test suite covers:

- exact hard-mask and offset definitions;
- zero gradient to excluded hard-mask logits;
- retained inactive-row gradient under OFFSET_EQ4096;
- fixed-endpoint formation metrics and sustained-G50 semantics;
- prevention of post-hoc rescue by secondary mask levels;
- separate structural-only versus functional+structural verdicts;
- runtime resolver invariance: slower hardware may increase estimated session count but may not remove arms/seeds or lower exposure.

Existing repository CI on the frozen executable commit independently reports success for the inherited CYR-GPU-006/007/008/009/010/011 contract workflows and ESOES contracts. R1C itself is intentionally gated by the launcher pytest + real CUDA smoke rather than being called executed from static CI.

## Important treatment boundary

Hard masking does **not** isolate a mathematically pure denominator scalar. Because the output projection is tied to the embedding table, hard masking removes output-loss gradient from excluded inactive rows (while ordinary AdamW parameter semantics remain intact). Therefore a positive hard-mask result means **training-time inactive-class competition / output-row participation is sufficient**, not that denominator size alone is proven causal.

OFFSET_EQ4096 is what tests the narrower aggregate probability-mass hypothesis while retaining all inactive rows in gradient flow.

## Runtime expectation before hardware calibration

Historical R1B T4 calibration measured the V24576 arm at about `2.3727 updates/s`. A naive 72,000-update full-matrix extrapolation is already ~506 minutes of training before R1C's extra diagnostics, safety factor and finalization. Therefore R1C is deliberately designed as a multi-session campaign. The actual preflight calibration is authoritative only for **time/session estimates**, never for changing scientific design.

## Gate verdict

**READY FOR OPERATOR COLAB CUDA PREEXECUTION GATE.**

This is not a claim that the GPU smoke has passed and not a scientific result. Scientific execution begins only after the pinned notebook verifies code identities, pytest passes, CUDA is present, matched initialization passes, the diagnostic is non-mutating, and exact resume passes.

No production tokenizer change, PRE500M, 500M, broad-reasoning, or AGI authorization follows from readiness or from R1C alone.
