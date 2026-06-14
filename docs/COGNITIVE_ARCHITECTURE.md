# AN-RA Cognitive Architecture

**Release:** `cognition-v1`  
**Lifecycle owner:** `cognition.services.CognitionServices`

AN-RA's cognitive layer is a versioned extension over the unchanged base-model lineage. It does not add parameters to an unattached 25M, frontier, or 3B model. Learned causal routing is attached explicitly, starts at an exactly zero residual gate, and is saved in a separate checkpoint tied to base-checkpoint and tokenizer hashes.

## Capability Contracts

| ID | Capability | Owner | Promotion evidence |
| --- | --- | --- | --- |
| C-01 | Causal Reasoning Engine | `cognition/cre.py` | A-01, zero-gate parity |
| C-02 | Epistemic Tracker | `cognition/epistemic_tracker.py` | A-02, calibration history |
| C-03 | Longitudinal Human Model | `cognition/lhm.py` | consent, encryption, deletion, A-03 |
| C-04 | Scientific Self-Improvement | `cognition/ssie.py` | signed authorized experiments, A-04 |
| C-05 | Cross-Domain Synthesis | `cognition/cdse.py` | verifier and expert review, A-05 |
| C-06 | Experience Consolidation | `cognition/cec.py` | idempotent session reports, A-06 |
| C-07 | Multi-Agent Self-Debate | `cognition/self_debate.py` | evidence-gated synthesis |

## Non-Negotiable Boundaries

- Inferred owner attributes require opt-in. Persistent sensitive state additionally requires encryption.
- Wellbeing records are observations, never diagnoses.
- Unverified CEC lessons remain quarantined and cannot enter replay.
- SSIE may propose an experiment but only the owner may authorize execution.
- Cross-domain output is a candidate hypothesis until verification and expert review.
- Debate persists summaries and verifier evidence, not private scratch reasoning.
- A-03 through A-07 are research evidence and do not block cognitive-extension promotion.

## Colab T4

`notebooks/AN_RA_T4_TRAINING.ipynb` and `scripts/colab_bootstrap.py` are the canonical entry points. Supported classes:

- `t4_full_25m`: complete 25M training and cognitive integration.
- `t4_frontier_smoke`: construction profiling, adapters, compatibility, and evaluation.
- `t4_3b_preflight`: analytical checks and blocker reporting without full allocation.

Run:

```powershell
python -m training.train_unified --mode preflight --model-size 25m
python -m training.train_unified --mode preflight --model-size frontier
python -m training.train_unified --mode preflight --model-size 3b
```

A blocked preflight is a valid, useful result. It must be resolved through evidence or hardware, never bypassed in a notebook.

## Causal Corpus Evidence

`data/causal_corpus.py` deterministically builds the exact 7,500-record development curriculum and verifies its counts and hashes. Its current evidence maturity is `synthetic_verified`, not promotion-grade external evidence.

Promotion remains blocked until those templates are replaced or independently validated against pinned, licensed medical, physics, policy, A/B-test, SCM, simulator, and statistical-fallacy sources. Synthetic curriculum metrics cannot satisfy A-01 by themselves.
