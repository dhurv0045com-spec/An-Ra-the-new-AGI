# V5 Freeze Checklist

The V5 training path is **not frozen** until every item below has an explicit answer, evidence link/receipt, or deliberate rejection.

## Cognition contract

- [ ] exact cognition primitives defined
- [ ] development benchmark implemented
- [ ] sealed promotion benchmark frozen
- [ ] OOD axes defined
- [ ] representation vs realization metrics separated
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
- [ ] no unnecessary experimental modules in baseline
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
- [ ] raw and assisted capabilities reported separately
- [ ] no external intervention described as native Core capability
- [ ] no AGI claim implied by narrow benchmark success

## Freeze artifact

When all required items are resolved, produce one immutable `V5_TRAINING_SPEC_v1.0.md` containing exact values, hashes, manifests, commands, evaluation fixtures, promotion rules, and falsification criteria.

After that point, agents execute the spec. Any scientific redesign requires a version bump rather than silent modification.