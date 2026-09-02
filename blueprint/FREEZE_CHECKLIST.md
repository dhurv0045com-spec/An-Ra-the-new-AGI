# V5 Freeze Checklist

The V5 training path is **not frozen** until every item below has an explicit answer, evidence link/receipt, or deliberate rejection.

Ground Blueprint v0.4 state: **250M implementation contracts and shortcut-resistant E0/E1 research harnesses pass; sealed promotion certification remains incomplete.** Checked items below are research decisions or development invariants, not authorization to train.

Required critical path: **E0 benchmark certification → E1 tokenizer → E2 architecture → E3 data/objective → E4 minimal curriculum/optimization → E5 102M replication → E6 freeze review.**

## Cognition contract

- [x] exact cognition primitives defined
- [x] development benchmark implemented
- [ ] sealed promotion benchmark frozen
- [x] OOD axes defined
- [x] representation vs selection vs realization metrics separated in executable APIs
- [x] local six-heuristic state and five-heuristic rule red-team passed after false-green repair
- [x] model-scoring contract certified against deterministic oracle/broken/random controls
- [ ] production scorer certified on random-weight P35 × real tokenizers and target devices
- [ ] external sealed/natural review passed

## Tokenizer

- [ ] 16k/24k/32k comparison completed or a documented reason exists to skip
- [ ] sequence inflation measured
- [ ] nonce/copy/identifier behavior measured
- [ ] code/math effect measured
- [ ] tokenizer artifact/hash frozen

## Architecture

- [ ] depth-vs-width experiment completed
- [ ] attention topology decision justified
- [ ] context-length decision justified
- [ ] Q/KV head choice justified
- [ ] FFN allocation justified
- [x] no unnecessary experimental modules in baseline
- [ ] exact parameter count verified from executable model
- [x] pure configuration parameter receipt equals 250,216,960 including affine QK norm

## Data

- [ ] natural-data domains and source manifests frozen
- [ ] cognitive families frozen at generator-version level
- [x] training/development/sealed/fresh namespaces and explicit seeds separated
- [ ] mixture ratios justified experimentally
- [ ] curriculum schedule/competence gates frozen
- [ ] deduplication and contamination checks passed
- [ ] exact token counts per family known

## Objective / optimization

- [ ] LM-only versus targeted-objective question resolved enough for V5 v1
- [ ] optimizer frozen
- [ ] betas/weight decay frozen
- [ ] LR frozen
- [ ] schedule frozen
- [ ] batch/global-token size frozen
- [ ] gradient clipping frozen
- [ ] precision frozen
- [ ] token budget frozen

## Training integrity

- [x] optimizer/live-parameter identity invariant tested locally on exact P35; target/distributed path remains open
- [x] one-step real-update canary passed locally on exact P35; target path remains open
- [x] multi-step canary passed locally on exact P35 (three updates); long-run target path remains open
- [x] Adam step/moment change verified locally, including FP32 moments on the master-parameter BF16 path
- [x] parameter SHA change verified locally; target/distributed receipt remains open
- [x] context-appropriate resume tolerances calibrated at CUDA 1K / CPU 128; strict cross-kernel bitwise equality remains diagnostic only
- [x] local sampler/cursor exact-resume canary passed on CPU and CUDA (target distributed pack reader still required)
- [x] framework-neutral cumulative lifetime token ledger starts at zero, schedules by tokens, and handles an exact partial final update
- [x] framework-neutral transaction binds source commit/model/data/pack/tokenizer/run/optimizer/schedule/curriculum identities
- [ ] real distributed trainer emits and validates those bindings on target hardware
- [ ] target TPU/XLA preflight passes on the declared world size (CPU hosts must report blocked)

## Checkpoints

- [ ] immutable milestone cadence frozen
- [ ] full-resume milestone writer tested on target hardware
- [ ] remote durability verified
- [x] local atomic transaction rejects stale writers, partial inventory, corruption, and unsafe crash stages
- [x] exact middle-P35 model/AdamW/scheduler/RNG/cursor/ledger join restores and continues from a clean copy
- [ ] checkpoint promotion rule preregistered
- [x] final/latest chronology is mechanically forbidden as the sole promotion basis
- [x] evaluation/durability/promotion receipt schemas separate raw/assisted evidence, clean restore, gate identity, chronology, and independent signature
- [x] exact local random-weight P35 scorer has CPU/CUDA parity and rotation invariance
- [ ] production candidate-scoring policy passes null-bias and target-TPU gates
- [x] scorer fixture crosses surface family with hidden role and fails on contingency leakage
- [x] development scorer runner is resumable, hash-bound, and refuses partial/fresh aggregation
- [ ] powered development scorer tournament passes clustered TOST/Holm, panel, decoy, intervention, and CPU/CUDA gates

## Scaling

- [ ] tiny-model experiments support the chosen direction
- [ ] mid-scale replication supports it
- [x] expected compute/time/storage estimated for the 250M/5B center
- [ ] abort criteria defined
- [ ] evidence threshold for a post-V5 ~400M run met

## Scientific claims

- [ ] negative results retained
- [ ] baselines strong enough
- [ ] no future fixture used for tuning
- [x] raw and assisted capabilities required to be reported separately
- [x] external intervention explicitly excluded from native Core claims
- [x] AGI claims excluded from narrow benchmark success

## STEP 2 research decisions recorded

- [x] V4's certified-token limitation recorded; unsupported lifetime-token claims rejected
- [x] direct billion-scale launch rejected pending scale-transfer evidence
- [x] dense conventional V5-A baseline selected provisionally
- [x] query-swap contrast selected as the sole auxiliary-objective candidate
- [x] same-query margin rejected
- [x] synthetic-data provenance and verification contract defined
- [x] Connector-versus-Core boundary defined
- [x] behavioral, not chronological, checkpoint promotion established
- [x] six-experiment information-gain program defined
- [x] compute estimate and abort logic documented
- [x] EXP v10/v11 pair/composition claims reclassified as contaminated
- [x] old 166-VIE bank rejected as qualified causal evidence
- [x] inherited VNext implementation physically removed from ESOES
- [x] E0 deterministic development receipt, independent solver, generator property sweep, and chance/power audit created
- [x] between/after-event state queries, split-held-out rule structures, pooled fail-closed lexical/position/rule gates, context/difficulty/output axes, and metric-specific confidence procedures implemented
- [ ] natural-source custody and real sealed commitment complete
- [x] E1 artifact-bound static audit, Pareto harness, and matched-budget tournament plan implemented
- [x] local-development 16k/24k/32k candidates independently trained, audited, and 24k determinism replicated
- [ ] real 16k/24k/32k tokenizer candidates audited
- [x] local CUDA attention-path canary replicated; target TPU/XLA GQA path remains open
- [x] exact P35 full-stack CPU/CUDA execution canary replicated with randomized case order
- [x] paired exact-stack CPU/CUDA residual-initialization signal canary replicated through native 4k context; target TPU/XLA and real-update checks remain open
- [x] paired CPU/CUDA QK-norm scale-control canary replicated through native 4k context; learned-quality and target TPU/XLA checks remain open
- [x] exact-stack CPU/CUDA BF16-versus-FP32 forward/backward parity replicated through native P35 2k; 4k V5-A, target TPU/XLA, and real-update/long-run checks remain open
- [x] exact P35 RoPE conformance replicated against an independent float64 oracle at native 4k; base choice, extrapolation, target TPU/XLA, and learning checks remain open
- [x] 250M model/run/lineage contracts and implementation blueprint created
- [x] implementation-complete `V5_TRAINING_SPEC_v1.0.md` and source-bound executable receipt created
- [x] every candidate constant has explicit value, evidence class, or fail-closed external identity
- [x] conditional-realization denominator uses correct unassisted selection
- [ ] power-sized family/Wilson evaluator and sealed custody complete
- [ ] E1/E2/E3/E4/M102 and TPU/remote gates fill the launch manifest

## Freeze artifact

`V5_TRAINING_SPEC_v1.0.md` now freezes the implementation candidate. It deliberately contains null external identities and `main_training_authorized=false`; the signed launch manifest is produced only after every unchecked scientific and target-infrastructure gate resolves.

After that point, agents execute the spec. Any scientific redesign requires a version bump rather than silent modification.
