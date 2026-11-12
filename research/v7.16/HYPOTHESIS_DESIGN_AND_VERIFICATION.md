# V7.16 Causal Falsification & Mechanism Suite — Design & Verification

Status: **design implemented and CPU-verified; GPU execution pending owner-run
sessions.** Scientific status remains `NOT_RUN`; every evaluation stays
`CAUSE_UNRESOLVED` until a controlled replicated intervention isolates a
mechanism. Claim ceiling: `CONTROLLED_DEVELOPMENT_MECHANISM_ONLY`; all
authorization flags false (`anra_v5/elendril/v716_protocol.py`,
`PROTOCOL_SHA = 26617d4512f68435a0955d978bd89db3068867f0dddbf6a7e22ef20a3e0abe0e`).

## 1. Custody chain (Phase A runs before any data generation)

| Pin | Value |
| --- | --- |
| Parent branch | `Elendril` head `29b1da646bd06c7f2deeb5e5f3c9f08fe24cabbb` |
| Essay audit pin | `9092acb1` |
| V7.1 core source | `6104ffd7134cb4cf4b3f340bd060d24a1e2dd200` |
| V7.15 parent FULL SHA-256 | `1826c85b5a858d2a600661f7e5916f0fa812c55a8d53c37013c695df28a15d19` |
| V7.15 S8 source / run | `e5779922a0072a070f64356f38ea47090e543d6a` / `3981e34a2ffe4b4b9528b4b4d588883b` |
| V7.15 S20 source | `4cc1e993c09d1475a3bc1e250fda58909d539407` |
| V7.15 S8 ANALYSIS zip | `08cca3edcec070b85fac752faf89f156b85fb7b7fc2fbd9b02c908461175c02b` (16,023,595 bytes) |

`anra_v5/elendril/custody_gate.py` verifies archive bytes against these pins
and emits a strict gate record. A missing or mismatched parent archive yields
`NO_GO` (reason `NO_GO_MISSING_PARENT_ARCHIVE` class failures) with **zero**
optimizer steps, zero generated datasets and no output-directory side effects
beyond the campaign record. Alternatively a previously extracted V7.1 core
output can satisfy custody through `verify_parent_output` (identity must equal
`{source: V7.1 core SHA, protocol: V7.1 PROTOCOL_SHA, phase: core}` or the
current checkout's core identity).

`research/v7.16/PREREGISTRATION.json` was committed before any run. The
campaign re-verifies it at every start: the parsed object must equal
`v716_protocol.preregistration()` exactly and the file bytes must equal
`canonical(preregistration()) + b"\n"` (round-trip verified). The committed
`EXECUTION_BINDING.json` template pins
`preregistration_sha256 = digest(preregistration())` (canonical bytes, sans
trailing newline) and `protocol_sha256 = PROTOCOL_SHA`; the campaign fills it
write-once (`custody_gate.write_once`) only after data generation, positive
control canaries and production-shaped profiles, but **before** the first
registered update, recording `parent_full_verified`, `s8_analysis_verified`,
the split/surface digests and per-seed `init_weights_sha256` from the profile
receipts.

## 2. Exposure lock

Blocked cross-entropy baseline exposure, identical for every mechanism arm:
**4 epochs / 2,368 optimizer updates / 75,776 supervised tokens /
8,914,404 unpadded input positions** (parameter inventories S3 = 3,084,800,
S8 = 7,970,944, S20 = 19,871,616). Codified semantics
(`exposure_semantics`): optimizer updates and supervised tokens are locked
exactly per arm; unpadded input positions are recorded per epoch and stay
identical for H1/H2A/H2B/H4/H5, while H3 counterfactual evidence may change
rendered prompt length by construction (a legitimate, preregistered
divergence). The intermediate-supervision reference exposure (3,068 updates /
109,280 tokens) is reported separately and never conflated.

## 3. The six falsifiable arms (one factor at a time vs. blocked CE)

| Arm | Registered mechanism | Falsified when |
| --- | --- | --- |
| H1-COADAPT-MASK | training-time controller logit masking from public state only (masked logits −1e9 before CE) | confirmation < 270/336 mean, or mechanism diagnostics show no masking effect |
| H2-ARK014-INVARIANCE | per-epoch deterministic surface curriculum (canonical → reversed rows → uniform timestamp offset → symbol permutation), digit-width-preserving tick shift | < 270/336 or row-reversal collapse persists |
| H2B-NORM | controller canonicalization sorting observation rows by (entity, tick asc) before rendering; un-augmented data | < 270/336 or no E04 delta vs. baseline |
| H3-CONTRAST-MEM | 25% shape-matched replacement with stale-overwrite / consistent-control / unresolved-contradiction pairs | < 270/336 or stale-memory < 28/48 |
| H3B-CONTRASTIVE | cosine-margin contrastive penalty (weight 0.1, margin 0.5) between decision-token hidden states of reversed vs. chronological renderings | < 270/336 or probe MI unchanged |
| H4-OUTCOME-DELIB | outcome-verified bounded rollouts (K ∈ 1/2/4; registered K=2) scored by U(τ); CE updates only on traces above the RLOO baseline | < 270/336 or kept-trace substitutions never exceed the baseline |
| H5-POINTER-READ | minimal pointer-copy gate plus two-read relational pooling; parameter ceiling 3,090,000 at S3; fresh-only (no resume) | < 270/336 or hop-4 accuracy unchanged |
| H6-ARK017-CONTINUAL | continual stage arms: b-only / replay / fresh-b (delegated unchanged to the qualified v7.15 continual path) + native ark017-sparse replay buffer (10% swap fraction, EMA 0.3) | retention/novel-B contrast diverges from the registered control pattern |

Registered costs: U(τ) = I(correct ∧ publicly knowable) − 0.1·verified_queries
− 0.0·deliberations − 0.05·illegal.

### H4 executed-rollout provider (no fabricated traces)

`tools/elendril_v716_worker.py` ships `ShapedRolloutProvider`. For every chosen
(case, decision) task it **executes** K=2 real episodes via the qualified
`run_episode`/`verify` path with a seeded decoder:

- `SamplingPolicy` keeps the exact `NeuralPolicy` prompt rendering, budget
  accounting and no-repair `parse_action`, replacing argmax with a seeded
  multinomial restricted to the action support (`ROLLOUT_SUPPORT`: EOS, `!`,
  `A`–`P`, `Q`, `R`, `~`) — identical rollouts would make the RLOO baseline
  degenerate. Rollout seeds derive from `digest([provider schema, recipe seed,
  update, task, k])`, so the same update replays deterministically.
- Traces are scored with the registered U(τ) parameters; `select` keeps traces
  above the RLOO leave-one-out baseline only.
- A kept trace is substituted into the registered batch only when its action
  byte length satisfies `answer_bytes + 1 == eligible count` and
  `UTF8.frame(prompt, actions, max_length=pad_to)` reproduces the registered
  sample shape exactly (same total tokens, same eligible count). Every other
  outcome is ledgered to `rollout-audit.jsonl` with a machine-readable reason
  (`episode ended before decision slot`, `action outside the registered V6
  grammar`, `answer byte length breaks the shape contract`, `frame rejected:
  …`, `incomplete executed rollout group; baseline row kept`).
- Unqualified tasks keep the baseline row: the registered batch, the update
  count and the supervised-token ledger can never move. Audit rows record
  `update` (0-based pre-increment), a nested `selection` block
  (prompts/rollouts/kept/kept_fraction/mean_utility/rollouts_per_prompt),
  `substituted` and `rejections`.
- A `policy_factory` seam exists solely for bounded CPU canaries; production
  always resolves to `_sampling_policy` (the seeded multinomial decoder).

CPU smoke (`make_dataset(worlds=8)`, micro spec, 2 real steps with a
deterministic teacher+noise stub): 12/12 tasks substituted, zero rejections,
exposure ledger unchanged, shape contract held for every row, independent
provider replay byte-identical.

## 4. Campaign phases (`tools/elendril_v716_campaign.py`)

- **mechanism** — 21 predeclared jobs (7 S3 arms × seeds 7/19/41), two GPU
  lanes, `--arm-subset`/`--seed-subset` narrow without dropping matrix cells.
  Preflight order: preregistration verification → custody gate → disk reserve
  → positive learning/recovery canaries → production-shaped profiles (8 timed
  updates + subprocess exact-recovery qualification; the profile receipts
  provide per-seed `init_weights_sha256`) → write-once binding → measured
  two-lane capacity forecast over the **whole** registered slice (no
  score-based pruning; a shortfall raises DEFERRED).
- **continual** — 9 delegated v7.15 control arms (b-only/replay/fresh-b run by
  the qualified `tools.elendril_v71_worker` under the v7.1 protocol identity)
  plus 3 native ark017-sparse jobs (v7.16 worker; stage dataset =
  registered `new-b` batch base, replay pool = memory/missing samples of
  `replay.json`; the stage transition binds `child_recipe`/`child_data` to the
  exact child plan). Retention comparisons run through the existing
  acquisition gate on the recorded parent Skill-A; a parent that never
  acquired A yields `NOT_QUALIFIED`, not a fabricated comparison.
  Continual arms are never profiled (fresh-only): their capacity forecast uses
  S3 mechanism profiles × 1.25 margin.
- **promotion** — a single winning mechanism arm rerun at S8, admitted only by
  `promotion_gate`: all three seeds present, exactly 336 confirmation
  observations, aggregate ≥ 279/336, every seed ≥ 79/112, plus supplied
  diagnostic evidence (`--promotion-evidence`, schema
  `elendril-v716-promotion-evidence-v1`) with stale-memory ≥ 28/48 and
  row-reversal consistency ≥ 0.5. Missing evidence or a failed threshold
  raises `PROMOTION_BLOCKED` before any S8 update. S20 is registered to run
  only after an S8 promotion.

Worker safety: owner flag required, isolated single-CUDA enforcement, wall-cap
validation, `check_source` pin + clean-tree requirement, outside-source output
mandate, SIGTERM preserves the last verified checkpoint, transactional
checkpoints with exact resume where the registration allows it (H5 and
ark017-sparse are fresh-only and refuse recovery by design).

## 5. Verification status

**CPU-verified (this branch, before any GPU session):**
- 50 contract tests in `tests/test_v716_contracts.py` (protocol round-trip,
  custody gate fixture verification, write-once semantics, exposure locks,
  per-arm mechanism contracts including RLOO selection and utility accounting,
  H4 shape gating, H5 fresh-only refusal, ark017 ledger bounds, diagnostics
  determinism, the single-pass H2 audit ledger, the H3 multi-stale-row
  control, and the enforced invalid-action promotion rule) plus the
  post-review harness seam tests: an ark017-sparse CPU
  stage-migration canary proving the parent weights actually load through
  `migrate_stage`, predeclared `selected_jobs` slices, `_profile_conditions`
  (one measured S3 shape for continual), the promotion gate's admit/block
  matrix, the report's delegated-stage identity mapping and the notebook
  builder's structure, parseability and parent-flag guard.
- H4 provider end-to-end smoke on real cases (above).
- Preregistration/binding byte round-trips (`research/v7.16/*.json`).

### 5.1 Review iteration (helper-contract verification pass)

A second pass re-derived the contracts of every reused helper against its
source. Defects found were fixed and are now contract-tested:

- **Stage migration was never forwarded** — `_continual_trainer` accepted
  `transition`/`parent_store` but dropped them at the `V716Trainer` call, so
  ark017-sparse would have trained from scratch. The engine now receives both
  (migration canary proves parent weights load and `stage_transition` is
  recorded in metadata).
- **Child-data digest drift** — the engine computes `data_sha` from the
  V7-enriched provenance (`v7_sampling_method` injected by `V7Trainer`), so the
  worker's transition builder must digest the same enriched mapping. The
  enrichment is now the shared helper `v7_training.sampling_provenance`, used
  by both the engine and `stage_transition`; the bare mapping could never have
  satisfied `migrate_stage`, even on GPU.
- **fresh-b retention source** — the delegated v7.1 worker writes the
  pre-training B state to `initial.json` when its request carries no parent;
  the campaign retention block now reads `initial.json` for fresh-b and
  `after-transfer-before-B.json` for every parent-carrying arm.
- **Continual profiling waste** — the continual phase profiled all seven S3
  mechanism shapes (14 lane runs); it now profiles exactly one (`_profile_conditions`)
  because the protocol forecasts continual capacity from a single S3 shape
  with a 1.25 margin.
- **Delegated identity collision** — the report's `_identity` read the
  delegated v7.1 job's `mechanism='ce'`, merging b-only/replay/fresh-b into one
  aggregate; the registered arm now comes from the stage encoded in the job id
  (`H6-ARK017-CONTINUAL`), with exact fallbacks for registered job-id shapes.
- **Promotion gate parsing** — seed extraction uses the `-s(\d+)$` suffix
  match, and the mechanism-summary fingerprint is a plain SHA-256 of the file
  bytes (`digest()` expects JSON-able values, not bytes).
- **Notebook custody flags** — the launch cell passes `--parent-archive` only
  when no extracted parent exists; after extraction the verified directory is
  the sole custody input, so the gate record never carries a misleading
  archive-mismatch row.

### 5.2 Adversarial first-principles audit (independent re-verification)

A third pass re-audited the whole pipeline from first principles, tracing
execution paths end to end rather than trusting prior conclusions. Verified
sound (no change needed): H1 mask causality (`packed_layout` returns a
`[b,1,q,k]` broadcast mask, so the H5 copy pathway's `mask[:,0]` is
per-query causal — no leak, no cripple); H2B train/eval isolation
(evaluation renders through the arm-agnostic `render_public`, so H2B must
generalize); H3B penalty applied exactly once per epoch at the first
update via `_epoch_pairs` with the extra forward/backward ledgered; H4
provider (real episodes, real `verify()` outcomes, shape-contract
substitution, every rejection ledgered, rollout acceptance/substitution
counts surfaced in the worker result); H6 acquisition gates with
NOT_QUALIFIED records preserved and `after-transfer-before-B` retention
source; exposure-ledger accounting; optimizer ownership and nonfinite
guards; worker profile/recovery (challenge-verified republish plus exact
payload comparison); `prepare_data` transactional reuse; double-verified
ZIP export; the campaign's dual-lane queue with serialized snapshot
exports; and the notebook's safe default-off launch path.

Defects found were fixed and are now contract-tested:

- **H2 audit inflated the committed fallback ledger** — `audit()`
  evaluated every epoch three times (examples, case count, determinism
  digest), each recording into the same fallback counters, so provenance
  reported up to four times the fallback count actually consumed by the
  training row build. The audit now runs one pure evaluation per epoch
  (`_evaluate_epoch`), reports that pass's counters, and leaves the
  consumption ledger untouched: audit provenance is position-independent
  and equals exactly one consumption per epoch.
- **H3 consistent-control did not neutralize the planted memory value** —
  with three or more subject rows only the first stale row in row order
  was re-attributed, which could leave the oldest row (the one the paired
  stale-overwrite arm attaches to memory) attributed to the subject. The
  control now re-attributes every non-latest subject row to the distractor,
  guaranteeing a single-latest consistent subject history; supervision
  shape parity is preserved for every eligible case because `belief`
  resolves per-source latest values and the keeper row is untouched. The
  shipped generator produces at most two subject rows per eligible case,
  so S3 data is unchanged; the fix matters for donor data and future
  dataset variants.
- **Preregistered invalid-action promotion rule was never enforced** —
  `promotion_requires` includes "raw <= 861 across 6,720 records;
  grammar == 0", but the gate checked only accuracy, stale-memory and
  row-reversal thresholds. The gate now sums the arm's final-epoch raw
  invalid actions and grammar invalid actions from the summary rows and
  blocks when `raw > floor(861*records/6720)` (the V7.15 blocked-CE
  invalid rate, reproduced exactly at the registered 6,720-record scope)
  or when any grammar row is nonzero. Real reports always carry integer
  `invalid` per cohort, so the check cannot spuriously block a complete
  run.
- **CI never ran on the working branch** — `.github/workflows/v16-preflight.yml`
  triggered only on `review/v16-preflight-2026-11-12`; `opposing-star`
  is now a push trigger so every branch push runs the CPU preflight.
- **Test-count reconciliation** — the historical "423 passed" claim
  corresponds to 422 passed + 1 environmental skip (Windows symlink
  privilege, `test_elendril_v6_export.py:60`); the count is now 424
  passed + 1 skipped with the two new mechanism regression tests, and
  50 V7.16 contract tests.

**Pending GPU (owner-run only):** all 21 mechanism jobs, 12 continual jobs,
promotion jobs, production profiles, positive control canaries, independent
replay rescore, and every research number. Nothing in this document reports a
trained-model result.

## 6. Delivery and reporting

`tools/elendril_v716_export.py` produces `ELENDRIL_V716_ANALYSIS.zip` /
`ELENDRIL_V716_FULL.zip` over the qualified V7 transport with readback
verification and a per-run delivery receipt; `tools/elendril_v716_report.py`
emits per-cohort rows plus per-arm aggregates keyed by registered arm
(`arm_aggregates[arm]["jobs"]`) with status AVAILABLE/PARTIAL/NOT_RUN and
`cause_status: CAUSE_UNRESOLVED`. Archives verify bytes, not science. The
repository keeps only curated Markdown/JSON under `research/v7.16/`; raw
Kaggle ZIPs and receipts stay external per the branch instructions.

## 7. Honest boundaries

The mechanism suite is an instrument for falsifying six causal hypotheses
about the Elendril development lineage at matched exposure. It cannot and does
not claim: general reasoning improvement, any AGI-adjacent capability, or that
a passing promotion gate establishes a cause — only that the preregistered
falsification conditions did not fire at S8. Failures are reported as results.
