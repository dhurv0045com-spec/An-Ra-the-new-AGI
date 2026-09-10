# ARK-019 V3 ADDENDUM — R2-INFORMED HIERARCHICAL CAPABILITY GUARDIAN

## Status

**PREREGISTERED AFTER ARK-017 V2 OUTCOME AND BEFORE ARK-019 V3 IMPLEMENTATION / EXECUTION.**

This addendum supersedes the mechanism-selection and arm sections of `PLAN.md` / `PLAN_V2_ADDENDUM.md` where explicitly stated. The scientific purpose, CONTROL/SEALED firewall, real-text substrate requirement, exact-resume requirement, and claim boundaries remain.

## Evidence that is allowed to influence this design

ARK-017 V2 final audited evidence is fixed before this addendum:

- bundle SHA-256: `c648d3fde569ca66fb34b54e56c09589a020a71bb5f7dcb88e476c224292d32c`;
- primary verdict: `BOTH_LEVERS_SUFFICIENT`;
- `NARROW_HIGH`: 4/6 SEALED robust failures;
- `NARROW_HIGH_CAP1X`: 0/6;
- `NARROW_HIGH_REPLAY_1OF16`: 0/6 despite much larger cumulative movement than failing HIGH;
- secondary screen: CAP4X 0/3, CAP16X 0/3, replay 1/32 0/3, replay 1/64 0/3.

ARK-018 V4 is also fixed evidence: real-text pretraining completed for both seeds; heavy specialized rehearsal changed later binding acquisition speed, so R3 must measure new-skill plasticity rather than optimizing retention alone.

The R2 secondary dose screen is **guidance, not a universal threshold claim**. V3 therefore tests 1/64, 1/32 and CAP16X prospectively as controller actions rather than promoting them directly.

## Question

> On a real-text-pretrained ~21M decoder, can a state-dependent controller preserve an acquired invariant while a disjoint new capability is learned and real-text training continues, with less protection cost than static replay and without materially slowing new-skill acquisition?

The policy under test is:

`HIGH plasticity normally -> margin warning -> sparse 1/64 support -> formal failure -> 1/32 support -> persistent failure -> temporary CAP16X emergency brake -> de-escalate after recovery`.

## Frozen substrate

Use the exact ARK-018 V4 prepared cache and `SCIENCE_ONLY` final checkpoints in Google Drive:

- root: `/content/drive/MyDrive/genisis-arkenstone/ARK018_SCIENCE_BIRTH_V1/`;
- model: ARK-018 `Ark018GPT`, ~20,954,880 parameters, vocab 8192, context 256;
- science file identity already bound by ARK-018: `b397427cd5964b7cc2a41264ca8789a0c020f96d4e403b314900798711a2ead5`;
- tokenizer and prepared-cache identities must be read from and validated against the ARK-018 prepared receipt at runtime;
- seed substrates: `31801`, `31902`, arm `SCIENCE_ONLY`, checkpoint step must equal the prepared horizon (8000 in the completed campaign).

No re-pretraining and no Birth-treated checkpoint are allowed.

## SKILL_A / SKILL_B construction

Select 24 frequent, lowercase, ASCII, single-token words using the frozen ARK-018 token-count rule. Partition deterministically:

- SKILL_A keys = selected IDs 0..5; values = 6..11;
- SKILL_B keys = selected IDs 12..17; values = 18..23.

The namespaces are token-disjoint.

SKILL_A uses the ARK-018 declarative template:

`Facts: K means V; ... Query: K means`

SKILL_B uses a different table-like template:

`Map: K -> V | ... Requested K =>`

Each skill uses 3 bindings/fact-set, fact-set split before query expansion, deterministic order augmentation during acquisition, and 400 TRAIN / 50 CONTROL / 50 SEALED fact-sets. SKILL_B uses a different deterministic fact-set shuffle seed. CONTROL drives qualification/controller state. SEALED is measurement only and never changes the controller.

## Parent acquisition gate

For each science-only substrate seed, acquire SKILL_A at HIGH `3e-4`, batch 64, deterministic order augmentation, max 1500 updates, CONTROL eval every 100 updates.

A parent qualifies only after **3 consecutive** CONTROL evaluations with:

- canonical >= 0.90;
- ORDER_ONLY >= 0.85;
- QUERY_ORDER >= 0.85.

Then measure SEALED exactly once and require the same robust thresholds. If either parent does not qualify, V3 is `BLOCKED_PARENT_ACQUISITION` and no matched Guardian claim is made.

The qualified parent model + optimizer state is frozen and hashed before any continuation arm.

## Matched sets

Two qualified SKILL_A parents x two prospective SKILL_B stream seeds = **4 mandatory matched sets**.

SKILL_B order seeds: `319001`, `319002`.

Every arm in a matched set starts from the exact same parent model/optimizer snapshot and receives the same real-text starts and same SKILL_B semantic IDs at every step before any SKILL_A replay replacement.

## Continuation objective and horizon

V3 uses a fixed **32 sequence-slot update contract** chosen to keep a complete four-set experiment practical on a Colab T4 while retaining real-text pressure:

- 28 real-text LM slots from the exact ARK-018 TRAIN token cache;
- 4 SKILL_B answer-prediction slots;
- replay, when active, replaces real-text slots only and never SKILL_B slots;
- each slot contributes one per-sequence mean-loss unit before averaging, so no long real-text sequence numerically overwhelms a binding slot;
- exact real-text tokens displaced by replay are recorded.

This slot-normalized V3 contract **supersedes V2's packed equal-target-token implementation detail**. It is fixed prospectively for all V3 arms and is not a production objective proposal.

Horizon: **1200 continuation updates per arm**. Eval every 100 updates. No arm stops early after SKILL_B qualification.

## Primary arms

Four arms are mandatory in every matched set.

### 1. `PLASTIC_HIGH`

- HIGH `3e-4` throughout;
- 28 real + 4 SKILL_B slots;
- no SKILL_A replay;
- no movement cap.

### 2. `STATIC_REPLAY_1OF64`

- HIGH throughout;
- exactly one SKILL_A non-canonical support slot every second update, deterministically scheduled, replacing one real-text slot;
- therefore one replay slot / 64 total sequence slots averaged over each two-update block;
- static protection from step 1.

### 3. `GUARDIAN_REPLAY`

State machine:

- `PLASTIC`: HIGH, no replay;
- `SPARSE64`: HIGH + 1/64 SKILL_A replay;
- `REPLAY32`: HIGH + 1/32 SKILL_A replay.

Transitions use SKILL_A CONTROL only:

- margin warning = either ORDER_ONLY or QUERY_ORDER < 0.90 on two consecutive evals while formal qualification may still hold;
- first margin warning: `PLASTIC -> SPARSE64`;
- any formal qualification failure: immediately enter `REPLAY32`;
- from `REPLAY32`, 3 consecutive healthy evals (canonical >=0.95, ORDER_ONLY and QUERY_ORDER >=0.90) de-escalate to `SPARSE64`;
- from `SPARSE64`, 3 consecutive healthy evals de-escalate to `PLASTIC`;
- a formal failure at any lower state escalates immediately to `REPLAY32`.

After SKILL_B CONTROL qualifies for 3 consecutive evals, the arm enters `CONSOLIDATE`: maintain at least `SPARSE64` for the remaining horizon, but formal failure can still escalate to `REPLAY32`.

### 4. `GUARDIAN_HYBRID`

Identical to `GUARDIAN_REPLAY`, except persistent failure in `REPLAY32` escalates to temporary `EMERGENCY_CAP16X`:

- if formal SKILL_A failure is present at two consecutive evaluations while already in `REPLAY32`, activate CAP16X + replay 1/32;
- after 3 consecutive healthy evals, remove the cap first and return to `SPARSE64`;
- subsequent healthy de-escalation can return to `PLASTIC` unless SKILL_B is already consolidated.

### CAP16X definition in V3

A matched-set-specific cap anchor is estimated **before any arm outcome is observed**. From the exact parent snapshot, run a 32-update shadow LOW branch (`3e-6`) on the matched continuation stream, record full applied parameter-delta norms, restore the parent, and freeze:

`CAP16X = 16 * median(shadow_LOW applied_delta_norm[1:32])`.

The shadow branch is calibration only and is never included as a scientific arm. The exact 32-step trace, median and cap are written to the matched-set entry receipt before continuation arms execute.

CAP is applied by post-step projection of the actual parameter delta back onto the pre-step point when the raw applied delta exceeds the frozen cap. AdamW optimizer moments are not rewritten; this is therefore an **applied-update cap**, not a trust-region optimizer claim.

## Replay semantics

Every replay event uses exactly one SKILL_A TRAIN semantic example rendered with a guaranteed non-identity fact permutation. The schedule and permutation are deterministic from `(parent_seed, skill_b_seed, step, replay_level)`.

- 1/64: exactly one replay slot on every even continuation step, zero on odd steps;
- 1/32: exactly one replay slot every continuation step.

Replay replaces a real-text slot. SKILL_B exposure is exactly identical across all arms.

## Primary endpoints

For each arm/matched set:

1. SKILL_A SEALED robust failure risk and robustness area;
2. SKILL_B CONTROL qualification-confirmation step;
3. SKILL_B final SEALED robust metrics;
4. science CONTROL + SEALED NLL change from parent entry;
5. replay duty cycle and exact replay slots;
6. displaced real-text sequences/tokens;
7. cap duty cycle and number of capped steps;
8. full/step parameter movement telemetry;
9. controller state transitions and trigger values.

## Primary success rule

`HIERARCHICAL_GUARDIAN_PROXY_CANDIDATE` requires all of the following across the 4 mandatory matched sets:

1. Interference exists: `PLASTIC_HIGH` has >=2 SKILL_A SEALED robust failures **or** mean SKILL_A SEALED robustness-area deficit >=0.15 versus `STATIC_REPLAY_1OF64`.
2. At least one Guardian has old-skill failure risk <= static replay +0.10 and, when PLASTIC_HIGH failures exist, >=0.30 below PLASTIC_HIGH.
3. Candidate median SKILL_B qualification step <=1.50x PLASTIC_HIGH median.
4. Candidate final SKILL_B SEALED robustness is within 0.05 of PLASTIC_HIGH.
5. Candidate SEALED science NLL is no more than 5% worse than PLASTIC_HIGH.
6. Candidate uses protection on <60% of pre-SKILL_B-qualification updates.
7. Candidate replay displaces <5% of real-text sequence slots before SKILL_B qualification.
8. Exact-resume smoke passes.

Tie-breaking between Guardians is frozen: old-skill risk -> SKILL_B speed -> science NLL -> replay cost -> cap duty cycle.

Additional flags:

- `SPARSE_REPLAY_GUARDIAN_SUFFICIENT` if GUARDIAN_REPLAY qualifies and HYBRID adds no needed rescue;
- `EMERGENCY_CAP_ADDS_VALUE` if HYBRID qualifies while replay-only does not because persistent failures are rescued by CAP16X;
- `STATIC_PROTECTION_ONLY` if static replay protects but neither dynamic Guardian meets plasticity/cost criteria;
- `INCONCLUSIVE_LOW_INTERFERENCE` if PLASTIC_HIGH never creates a meaningful contrast;
- `CONTROLLER_NOT_SUPPORTED` if no intervention preserves A while B learns.

## Exact resume / fault tolerance

Every arm checkpoint must persist model, optimizer, step, controller state, all health counters, B-qualification streak, replay counters, cap counters, cumulative movement, exact stream identities, and CPU/CUDA RNG states. A pre-execution smoke test must compare an uninterrupted branch with save/reload for the next **10 updates** and require identical parameter hashes and controller telemetry.

Checkpoints are written at every 100-step evaluation boundary and final step. Completed exact-identity arms may be reused after a Colab disconnect; incompatible artifacts abort rather than overwrite.

## Runtime policy

Target a Colab T4 and a **175-minute campaign wall with 5-minute packaging reserve**. Before scientific execution, benchmark one representative mixed update + evaluation on the real model/cache and conservatively project the four-set/four-arm campaign with a >=1.30 safety factor.

If all four mandatory matched sets cannot fit conservatively, V3 **fails closed before scientific training**. Do not reduce seeds, drop arms, shorten horizon, or change thresholds after calibration/outcomes.

## Outputs

Final bundle: `ARKENSTONE_ARK019_V3_GUARDIAN_RESULTS.zip`.

Required artifacts include:

- `ARK-019_V3_ENTRY_RECEIPT.json`;
- `ARK-019_V3_SMOKE.json`;
- parent receipts and hashes;
- matched-set cap calibration receipts;
- per-arm checkpoints/receipts;
- `CAPABILITY_STATE_TRACE.json`;
- `GUARDIAN_POLICY_SPEC.json`;
- `ARK-019_V3_RESULT.json`;
- failure receipt when applicable;
- manifest with artifact SHA-256 values.

## Claim boundary

A positive result promotes only a **real-text proxy Guardian candidate** for larger/Cymek replication. It does not establish a universal replay fraction, universal cap multiplier, production scheduler, broad continual learning, PRE500M/500M authorization, AGI, consciousness or identity.