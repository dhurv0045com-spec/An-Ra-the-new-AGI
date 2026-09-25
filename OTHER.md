# OTHER: the thinking behind Cymek and Signac

The README gives the current action. This note explains why it is the current action, what the system reuses, what we changed after seeing the results, and how the pieces fit into one evidence trail.

## The result that changed the work order

FORMATION-MUX-001 S5 ran all 24 planned arms successfully on two Tesla T4 GPUs over about 8.14 hours. The campaign completed; the problem was scientific sensitivity, not an interrupted run.

| What S5 measured | Observed result | What it tells us |
|---|---:|---|
| One-token termination | 1.00 exact by about update 300; sustained through update 2,000 | The model learned this short, directly measured behavior. |
| Two-hop composition | About 0.29 at the endpoint and still rising | Some longer-range behavior formed, but was not near mastery. |
| Missing-information abstention | About 0.14; valid EOS was 1.00 | The abstention task remained weak even though answer termination worked. |
| Multi-token identity/copy | Formation AUC was only 0.000–0.008 across the 24 arms; exact-match remained at the floor | The main endpoint had too little dynamic range to adjudicate the treatment contrasts. |
| Binding and state order | Roughly 0.00–0.05 | These multi-token families also remained near the floor. |

Every CS-MECH official arm also recorded `clip_fraction = 1.0`. That is an optimization warning to carry forward, not proof that the run failed. The preregistered contrast outputs remain `NULL`; their corrected interpretation is **`INCONCLUSIVE_AT_ZERO_BASELINE`**. We cannot infer that the mechanisms were equivalent when the capability measure barely moved in either group.

The [S5 postmortem](docs/cymek/experiments/FORMATION-MUX-001/observed/S5_KAGGLE_2026-09-15/POSTMORTEM.md) and [returned-bundle audit](docs/cymek/experiments/FORMATION-MUX-001/RETURNED_BUNDLE_AUDIT.md) preserve the full metrics and why the formal nulls are inconclusive.

The expensive mistake would be to rerun the mechanism sweep before fixing that uncertainty. So the next Cymek task is a short audit of saved checkpoints, with no new training.

## Why METRIC-RES-001 is next

METRIC-RES-001 reuses the exact 16 preserved S5 checkpoints: four CS-MECH treatment arms across four seed bundles. It keeps the model weights and frozen S5 questions unchanged while asking whether sequence-level exact match hid partial token-level skill.

The audit scores the public identity examples and uses termination as a positive control. It measures teacher-forced token accuracy, longest-common-prefix, accuracy by answer position and length, gold-token rank and margin, a shared-output comparison, and replayed clip telemetry. If the termination control fails to reproduce at least 0.90 exact-plus-valid-EOS, the instrument is invalid and no scientific conclusion is allowed. Sealed examples are never opened.

This is a measurement experiment, not a retraining experiment. It constructs no optimizer, performs no backward pass, changes no weights, and cannot retroactively change the S5 verdicts. Its decision rules were frozen before any audit outcome:

The complete thresholds and integrity checks are in the [METRIC-RES-001 preregistration](docs/cymek/experiments/METRIC-RES-001/PREREGISTRATION.json) and its [operator instructions](docs/cymek/experiments/METRIC-RES-001/README.md).

| Audit outcome | Meaning | Next move |
|---|---|---|
| `INSTRUMENT_INVALID` | The positive control did not reproduce | Repair the audit; draw no conclusion. |
| `FORMING_BUT_UNMEASURED` | Token accuracy and prefix reveal skill hidden by exact match | Re-adjudicate the frozen S5 contrasts in token space; do not rerun training yet. |
| `OPTIMIZATION_CHOKED` | Identity remains weak and clip telemetry is saturated | Design the bounded learning-rate/schedule probe. |
| `OUTPUT_COMPETITION` | A shared-output comparison shows meaningful rescue | Test that mechanism on a resolved measurement. |
| `MIXTURE_OR_DATA_LIMITED` | The audit finds partial but insufficient formation | Use the preregistered baseline exposure/mixture gate. |
| `GENUINELY_ABSENT` | The instrument works but identity remains absent | Stop the current vocabulary/mechanism line and use the queued scale-transfer question. |
| `INCONCLUSIVE` | No frozen category fits | Preserve the receipts and plan a new test before training. |

The required `resume.pt` files live in the original S5 Kaggle Output tree; they are not in the S5 result ZIP. Attach that Output to the Kaggle notebook before starting. The Kaggle T4 x2 path is preferred and takes about 10–20 minutes; the Colab T4 notebook is an alternative at about 20–40 minutes. Run one platform, not both.

## What we kept

**A stable model contract.** M102 uses the existing An-Ra V5 `ModelSpec` and training interfaces. Its exact size is 101,790,080 parameters. This gives comparisons one identifiable subject and prevents a model rewrite from obscuring the question.

**The actual update semantics.** The objective remains causal next-token prediction over eligible targets, with answer termination supervised. Accumulated microsteps make one optimizer update; distributed ranks share a checked global token denominator and update boundary. The run identity records the exact model, source, data, evaluation, seeds, and runtime.

**Prior findings, with their labels intact.** The development builder currently carries 81 structured experiment records plus the curated Signac evidence index. Records retain evidence class, source identity, supported and unsupported claims, and caveats. These records can inform development tasks; they are not all verified facts, a production corpus, or proof that a model learned their contents.

**The task and evaluation foundation.** The cognition generator has nine frozen families, family-aware packing, and 20 realizable interference dose/position cells at its development surface. Training and evaluation templates use separate namespaces. A 440-case E0 development certificate checks the evaluation suite itself; it explicitly does not say that a model was evaluated.

## What we changed on purpose

**We changed the order of expensive work.** S5 taught us to establish that the control can leave the floor before spending hours on a large causal comparison. METRIC-RES-001 checks measurement first; any later retraining depends on its frozen outcome.

**We measure partial behavior instead of relying on one all-or-nothing score.** Sequence exactness remains useful, but by itself it can report zero when most answer tokens are correct. Token accuracy, prefix length, position, answer length, EOS, and confidence margins help tell “nothing formed” from “something formed but the endpoint cannot see it.” The rules for interpreting these measures are fixed in advance.

**We left the model itself interpretable.** We did not add a special memory module, router, recurrent block, or auxiliary loss just to make the system sound more intelligent. Any intervention must be motivated by an observed gap and tested as a separate, prospective question.

**We optimized computation without claiming a new capability.** The shared rotary-position angles are built once per forward pass. The tied output projection and loss are computed in 512-position chunks with activation checkpointing. Host tests compare outputs and gradients with reference computations. This reduces redundant work and intended activation storage; Kaggle TPU throughput and memory still need actual measurement.

**We kept RSI diagnostic.** The weakest-axis analysis and registered-trial analyzer can help choose a future experiment. They do not modify the curriculum, update model weights, or make autonomous self-improvement claims.

## How the two research lanes fit together

Cymek and Signac share reproducibility standards but answer different questions. Their evidence does not substitute for one another.

| Lane | Current question | Current work | What a pass would establish |
|---|---|---|---|
| Cymek | Was multi-token identity/copy skill hidden by S5’s exact-match metric, or was it absent? | METRIC-RES-001 checkpoint audit on Kaggle T4 x2, no training | Which preregistered Cymek question should come next; no retroactive change to S5. |
| Signac | Can the M102-scale system execute and recover on Kaggle’s free TPU with the declared workload? | Bounded synthetic eight-device canaries; optional 21-update profile with a fail-closed 85% memory ceiling | Engineering qualification for that exact target session; not permission to start research training. |

Signac’s primary geometry is width 640 × 20 layers, 10 query and 5 KV heads, a 1,600-wide SwiGLU, and a 4,096-token context. Two near-100M TPU candidates remain unpromoted until measured. The production campaign is configured to save a resumable checkpoint every 200 completed optimizer updates, but this cadence has not yet been verified on Kaggle. The notebook creates a checksummed results ZIP; download it or save a Kaggle Notebook Version with outputs enabled so the temporary session does not erase the evidence.

The [Signac architecture](docs/signac_100m/ARCHITECTURE.md), [readiness gates](docs/signac_100m/READINESS.md), and [Kaggle runbook](docs/signac_100m/KAGGLE_TPU_RUNBOOK.md) carry the exact runtime and launch requirements.

## The decision standard

The work is progressing when a test makes one uncertainty smaller and leaves a receipt another person can inspect. A passing CPU smoke means local code ran. A passing Kaggle canary means the recorded target path ran. A research result requires a frozen question, qualified data, an independent evaluator, capable baselines, the planned seeds, and the declared analysis.

Even a replicated result supports only the named behavior under the tested conditions. A bigger model, a useful diagnostic, a successful checkpoint, or high TPU utilization does not by itself demonstrate general intelligence.
