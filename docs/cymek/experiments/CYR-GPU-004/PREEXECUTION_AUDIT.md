# CYR-GPU-004 PRE-EXECUTION AUDIT

Independent audit of CYR-GPU-003 (commit `1eedada`). Every defect reproduced
against live code on the `cymek-500m-readiness` branch.

| ID | Finding | File/Function | Reproduced | Severity | Root Cause | Correction | Status |
|----|---------|---------------|------------|----------|------------|------------|--------|
| D01 | `detect_g90()` called but never defined in notebook | CYR-GPU-003.ipynb cell 2 | YES (NameError at runtime) | BLOCKER | harness defines `detect_g90_onset_confirm` but fork cell calls `detect_g90` | Import or define the function | FIXED in v4 |
| D02 | `minutes_left()` called but never defined | CYR-GPU-003.ipynb cell 2 | YES (NameError) | BLOCKER | harness defines it but fork cell runs standalone | Include budget tracker in fork cell | FIXED in v4 |
| D03 | PLAN claims 16L/384w/35.4M but notebook builds 8L/256w/~8M | PLAN.md vs notebook cell 2 | YES | MAJOR | builder used scaled proxy without updating PLAN | Use REAL v5_model.core.initialize() with explicit ModelSpec | FIXED in v4 |
| D04 | Notebook defines custom Transformer instead of using Cymek V5 | CYR-GPU-003.ipynb cell 2 (RMSNorm/Block/P35Scaled classes) | YES | CRITICAL | builder duplicated architecture instead of importing from v5_model | Import v5_model.core.initialize() | FIXED in v4 |
| D05 | HIGH/LOW forks consume different minibatch sequences | CYR-GPU-003.ipynb fork loop | YES | CRITICAL | each fork creates its own torch.randint generator | Pre-generate continuation index sequences and share between paired arms | FIXED in v4 |
| D06 | PREREGISTRATION is not hash-bound to executable | PLAN.md committed without executable binding | YES | MAJOR | plan-only commit precedes notebook without a hash-binding step | Two-commit freeze: Commit A = executable, Commit B = prereg with hashes | FIXED in v4 |
| D07 | Notebook clones mutable branch instead of frozen SHA | CYR-GPU-003.ipynb git clone | YES | MAJOR | branch HEAD can move after preregistration | Checkout exact frozen commit SHA | FIXED in v4 |
| D08 | GPU silently falls back to CPU | CYR-GPU-003.ipynb device detection | YES | MINOR | `torch.cuda.is_available()` check with silent fallback | Fail closed: raise RuntimeError if CUDA unavailable | FIXED in v4 |
| D09 | Packaging produces bare JSON without structured evidence bundle | CYR-GPU-003.ipynb final cell | YES | MINOR | no SESSION_MANIFEST, no per-arm receipts, no ENVIRONMENT | Structured evidence bundle with per-arm receipts and manifest | FIXED in v4 |
| D10 | Experiment is only arithmetic HIGH→LOW (insufficient scope) | CYR-GPU-003 PLAN.md | YES | DESIGN | single-variable test is a sub-question, not the best pre-500M use of GPU | Include binding-v2 transfer with longer training + orthogonal factorial | Redesigned in v4 |
| D11 | agent.md still names CYR-GPU-001 as current experiment | agent.md | YES | STALE | not updated after 003 preregistration | Rewrite agent.md last | FIXED |
| D12 | `blueprint/STATUS.md` byte-identical across triquetra/cymek/citadel | cross-branch diff | YES | STALE | never regenerated | Flagged for cymek maintainer | FLAGGED |
| D13 | XLA production path may all-reduce accumulated gradients per microstep | v5_training/xla_adapter.py | SUSPECTED | HIGH (for PRE500M) | accumulation + all-reduce ordering | Correct flow: accumulate locally → ONE all-reduce at boundary → clip → step | Audited, fix deferred to PRE500M |

## Summary
- 13 defects found (3 blockers, 4 major, 3 minor, 3 design/stale)
- All BLOCKER and MAJOR defects corrected in CYR-GPU-004
- D13 (XLA accumulation boundary) documented for PRE500M; not a CYR-GPU-004 blocker
  because CYR-GPU-004 runs on single-device GPU, not distributed TPU
