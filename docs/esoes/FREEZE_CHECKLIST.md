# V5 Freeze Checklist

The V5 training path is **not frozen** until every item below has an explicit answer, evidence link/receipt, or deliberate rejection.

STEP 2 state: **research synthesis complete; experiments and executable receipts incomplete.** The provisional values live only in `V5_MASTER_BLUEPRINT.md`. Checked items below are research decisions, not authorization to train.

Required critical path: **E0 benchmark certification → E1 tokenizer → E2 architecture → E3 data/objective → E4 minimal curriculum/optimization → E5 102M replication → E6 freeze review.**

## Cognition contract

- [x] exact cognition primitives defined
- [ ] development benchmark implemented
- [ ] sealed promotion benchmark frozen
- [x] OOD axes defined
- [x] representation vs selection vs realization metrics separated
- [ ] shortcut/leakage review passed

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

## Data

- [ ] natural-data domains and source manifests frozen
- [ ] cognitive families frozen at generator-version level
- [ ] train/dev/test seeds separated
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
- [ ] expected compute/time/storage estimated
- [ ] abort criteria defined
- [ ] evidence threshold for ~300M run met

## Scientific claims

- [ ] negative results retained
- [ ] baselines strong enough
- [ ] no future fixture used for tuning
- [x] raw and assisted capabilities required to be reported separately
- [x] external intervention explicitly excluded from native Core claims
- [x] AGI claims excluded from narrow benchmark success

## STEP 2 research decisions recorded

- [x] V4's certified-token limitation recorded; unsupported lifetime-token claims rejected
- [x] direct 300M–3B launch rejected pending scale-transfer evidence
- [x] dense conventional V5-A baseline selected provisionally
- [x] query-swap contrast selected as the sole auxiliary-objective candidate
- [x] same-query margin rejected
- [x] synthetic-data provenance and verification contract defined
- [x] Connector-versus-Core boundary defined
- [x] behavioral, not chronological, checkpoint promotion established
- [x] six-experiment information-gain program defined
- [x] compute estimate and abort logic documented

## Freeze artifact

When all required items are resolved, produce one immutable `V5_TRAINING_SPEC_v1.0.md` containing exact values, hashes, manifests, commands, evaluation fixtures, promotion rules, and falsification criteria.

After that point, agents execute the spec. Any scientific redesign requires a version bump rather than silent modification.
