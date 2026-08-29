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
- [x] local shortcut/leakage red-team passed; external sealed/natural review remains open

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

- [ ] optimizer/live-parameter identity invariant tested
- [ ] one-step real-update canary passed
- [ ] multi-step canary passed
- [ ] Adam step/moment change verified
- [ ] parameter SHA change verified
- [ ] sampler/cursor exact-resume test passed
- [ ] cumulative lifetime token ledger starts at zero and is unambiguous
- [ ] source commit/data/tokenizer/config bound into receipt

## Checkpoints

- [ ] immutable milestone cadence frozen
- [ ] full-resume milestone writer tested on target hardware
- [ ] remote durability verified
- [ ] checkpoint promotion rule preregistered
- [ ] final checkpoint is not automatically promoted

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
- [x] E0 deterministic development receipt, independent solver, 20-seed property sweep, and initial chance/power audit created
- [x] semantic-time state queries, split-held-out rule structures, pooled heuristic gates, context/difficulty/output axes, and metric-specific confidence procedures implemented
- [ ] natural-source custody and real sealed commitment complete
- [x] E1 artifact-bound static audit, Pareto harness, and matched-budget tournament plan implemented
- [x] local-development 16k/24k/32k candidates independently trained, audited, and 24k determinism replicated
- [ ] real 16k/24k/32k tokenizer candidates audited
- [x] local CUDA attention-path canary replicated; target TPU/XLA GQA path remains open
- [x] exact P35 full-stack CPU/CUDA execution canary replicated with randomized case order
- [x] 250M model/run/lineage contracts and implementation blueprint created

## Freeze artifact

When all required items are resolved, produce one immutable `V5_TRAINING_SPEC_v1.0.md` containing exact values, hashes, manifests, commands, evaluation fixtures, promotion rules, and falsification criteria.

After that point, agents execute the spec. Any scientific redesign requires a version bump rather than silent modification.
