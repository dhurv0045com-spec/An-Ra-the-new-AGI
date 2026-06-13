# AN-RA Master Goals

> One backlog for architecture, evidence, training, agency, robotics, deployment and research.

**Reviewed:** 2026-06-13
**Status keys:** `DONE`, `ACTIVE`, `NEXT`, `BLOCKED`, `IDEA`

`DONE` means the implementation contract and tests exist. It does not imply a model or deployment has been promoted.

## Critical Path

| ID | Status | Goal | Completion evidence |
| --- | --- | --- | --- |
| P0-01 | DONE | Canonical architecture ownership | owners documented and exercised by tests |
| P0-02 | DONE | Exact frontier and 3B model contracts | exact-count tests |
| P0-03 | DONE | Deterministic tokenizer migration | legacy-row and deterministic-row tests |
| P0-04 | DONE | Four-stage resumable campaign | stage and campaign-resume tests |
| P0-05 | DONE | SSG structured blockers | deterministic 3B status output |
| P0-06 | BLOCKED | Promoted frontier release | signed release plus three-seed evidence |
| P0-07 | BLOCKED | Authorized 3B growth candidate | promoted parent, hardware profile, manifests, parity |
| P0-08 | BLOCKED | Full 21B-token campaign | licensed inventory, compute and stage evidence |

## Evidence Backlog

| ID | Status | Goal | Verifier |
| --- | --- | --- | --- |
| EV-01 | NEXT | Freeze at least 50 public/integration IBS tasks | suite manifest |
| EV-02 | NEXT | Freeze at least 50 private owner tasks | gitignored approved manifest |
| EV-03 | NEXT | Freeze 200-question private memory benchmark | benchmark hash and approval |
| EV-04 | NEXT | Measure target-hardware optimizer profile | optimizer report |
| EV-05 | NEXT | Measure 3B memory and throughput | hardware profile artifact |
| EV-06 | NEXT | Produce truth coverage above 95% | RLVR report |
| EV-07 | NEXT | Produce service uptime/recovery evidence | telemetry artifact |
| EV-08 | NEXT | Define sovereignty-accuracy benchmark | M-10 evidence schema |

## Training And Data

| ID | Status | Goal | Verifier |
| --- | --- | --- | --- |
| TR-01 | DONE | DEL ordered intake | ingestion tests and rejection report |
| TR-02 | DONE | Immutable hashed `uint16` shards | manifest and immutability tests |
| TR-03 | DONE | SADL and bounded OGRS | mix-control tests |
| TR-04 | DONE | WSD and one-time owner annealing | resume/phase tests |
| TR-05 | DONE | Protected-parameter PCGrad | accumulation/conflict tests |
| TR-06 | DONE | Verified CDR replay lifecycle | closure and flush tests |
| TR-07 | NEXT | Download and publish licensed FineWeb-Edu shards | immutable source manifest |
| TR-08 | NEXT | Run 25M repeated smoke sessions | seven clean reports |
| TR-09 | BLOCKED | Stage A candidate | at least 5B trained tokens and gate evidence |
| TR-10 | BLOCKED | Stages B-D candidates | preceding stage promotion and required data |

## Evaluation And Promotion

| ID | Status | Goal | Verifier |
| --- | --- | --- | --- |
| EP-01 | DONE | Seven-dimension IBS weighting | IBS tests |
| EP-02 | DONE | Three-seed aggregate and confidence evidence | IBS aggregate artifact |
| EP-03 | DONE | Signed atomic promotion | release-manifest tests |
| EP-04 | DONE | Automatic smoke rollback | promotion rollback test |
| EP-05 | DONE | Evidence-backed M-01 through M-12 snapshots | metric schema and canonical readers |
| EP-06 | NEXT | Produce random baseline artifact | baseline release |
| EP-07 | NEXT | Run public and private suites for frontier | three-seed candidate report |

## Agency And Memory

| ID | Status | Goal | Verifier |
| --- | --- | --- | --- |
| AM-01 | DONE | Typed HGP mission trees | schema and depth/leaf tests |
| AM-02 | DONE | Deterministic execution/recovery | agent-loop tests |
| AM-03 | DONE | Verified trajectory store | hash and verification tests |
| AM-04 | DONE | Hybrid source-preserving retrieval | fusion tests |
| AM-05 | NEXT | Collect 1,000 machine-verified trajectories | M-04 evidence |
| AM-06 | NEXT | Meet combined Recall@1 and Recall@3 targets | private memory benchmark |

## Robotics

| ID | Status | Goal | Verifier |
| --- | --- | --- | --- |
| RB-01 | DONE | Simulation/shadow workflow boundary | workflow tests |
| RB-02 | DONE | State/action codec and uncertain world model | codec/model tests |
| RB-03 | DONE | Offline-only world-model training | boundary test |
| RB-04 | BLOCKED | Activate planning assistance | 100,000 transitions, accuracy and improvement gates |
| RB-05 | IDEA | Physical actuation promotion program | owner approval, emergency stop, supervised hardware validation |

## Service And Operations

| ID | Status | Goal | Verifier |
| --- | --- | --- | --- |
| SO-01 | DONE | Persistent jobs and sessions | SQLite service tests |
| SO-02 | DONE | Typed `/goal` alias | route/schema tests |
| SO-03 | DONE | Request IDs, timeouts and audit records | API tests |
| SO-04 | DONE | Enforce fail-closed production authentication policy | production-mode test |
| SO-05 | NEXT | Add role separation beyond owner/anonymous | authorization matrix |
| SO-06 | NEXT | Web console consumes only real evidence | frontend integration tests |

## Continual Improvement

| ID | Status | Goal | Verifier |
| --- | --- | --- | --- |
| CI-01 | DONE | Skip adaptation below 100 examples | threshold test |
| CI-02 | DONE | Isolated LoRA/DoRA candidates | candidate lifecycle tests |
| CI-03 | DONE | Standard evaluation/promotion/quarantine | continual tests |
| CI-04 | DONE | Disable proposal auto-application below 20% success | policy test |
| CI-05 | NEXT | Run first real adapter candidate | signed adapter release or quarantine |

## Documentation Rule

Every goal must point to executable evidence. Dated audits stay historical. Current manuals describe live ownership. Research notes propose experiments; they do not declare promotion.
