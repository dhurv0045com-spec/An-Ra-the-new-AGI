# V7.17 — S20 optimization helps; grounded composition remains unresolved

Owner results analyzed 2026-11-13. Run `f443d3f4141b4163bbc130d46ebd7395`,
source `724af4cb508173e0eb3f4397d5dcb4a42a66712f`, protocol
`b3030032f5758a1eaf81b6677db9655b475fd18955feda7b28c627cb6eee0088`.
**Eight of eight registered optimization runs executed; this is not a universal capability success.**

## What was verified

The supplied `ELENDRIL_V71_ANALYSIS (3).zip` contains 159 manifest-listed files
plus the manifest, 17,589,179 compressed bytes. SHA256:
`10d024a5daa64b24a63e549b6c8a78c31c8eff141772ec84efee4b5a9f02e1fc`.
All inventory hashes passed. The inherited V71 filename and V6/V7.1 transport
headings are legacy labels; source, protocol and phase identify V7.17.

Independently reconstructed the suite and train/validation/confirmation
identities. Replayed **44 evaluation reports** (40 registered model reports,
four production profiles). Audited **32 diagnostic milestones** against their
checkpoint/update/data identities, target marginals, confusion counts and
recorded decoded exact matches. All eight trajectories contain updates 1–2368
without gaps and finite logged loss. Twelve Brier values required the existing
Linux/Windows tolerance of at most two binary64 ULPs; raw receipts were not altered.

All four production probes report update/gradient-parity and exact fresh-process
recovery PASS. Eight initial diagnostic gradient controls report PASS. This
analysis checks their receipts, not the missing gradient tensors or fresh-VM
numerical equivalence. The separate FINAL_STATUS and FULL ZIP were not supplied:
**final dual-export certification and independent checkpoint restoration remain
unverified locally.** A requested-complete marker is not a final delivery receipt.

## The controlled result

All runs use fresh 19,871,616-parameter S20 models, identical training worlds and
9472 examples, four epochs, 2368 updates, batch16/micro2 and the registered
blocked sampler. Peak LR is the intervention; schedule and exposure stay fixed.
Initial tensor fingerprints match within each paired comparison, and LR pairs
have identical sampling-plan hashes. Crossed runs instead hold initialization
fixed and change only order seed. No model size was reduced.

**Primary scores below use raw learned actions, not grammar-constrained decoding.**

| Initial seed | Order seed | Peak LR | Base /112 | Confirmation /112 | Full-train exact action | Held-out value accuracy | Three-hop /16 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 7 | 3e-4 | 46 | 54 | 55.10% | 19.72% | 0 |
| 19 | 19 | 3e-4 | 80 | 79 | 69.37% | 52.11% | 0 |
| 41 | 41 | 3e-4 | 75 | 78 | 65.82% | 43.66% | 0 |
| 7 | 7 | 1e-4 | 91 | 88 | 74.93% | 59.15% | 0 |
| 19 | 19 | 1e-4 | 92 | 99 | 88.72% | 69.01% | 0 |
| 41 | 41 | 1e-4 | 90 | 97 | 88.17% | 66.20% | 0 |
| 7 | 19 | 3e-4 | 57 | 56 | 58.57% | 25.35% | 0 |
| 19 | 7 | 3e-4 | 45 | 47 | 53.91% | 9.86% | 0 |

The three baselines exactly reproduce V7.16's compact raw metrics across all
16 evaluation cohorts, including calibration and work counts. This is aggregate
behavioral reproduction, not independent tensor/checkpoint reproduction.

Lowering LR improves confirmation by **34, 20 and 19 cases**, respectively.
Mean confirmation accuracy increases **62.80% → 84.52%**; base increases
**59.82% → 81.25%**. Paired confirmation outcomes contain 81 newly correct and
eight newly incorrect seed/case observations. These are three initializations
on shared development worlds, not 336 independent experimental replications.

Mean full-training exact-action accuracy increases **63.43% → 83.94%**. The
baseline failed substantially even on its own training examples; its weakness
cannot be described solely as held-out generalization failure. Mean held-out
value-token accuracy increases **38.50% → 64.79%**. EOS accuracy is 100% on the
final full training/validation measurements in all eight runs, with zero
teacher/greedy exact-action gap on the identical sampled final cohorts. This
does not prove the entire decoder is bug-free, but the measured residual error
is principally wrong actions/values rather than failure to emit EOS.

The controlled LR intervention demonstrates **optimization/recipe sensitivity
contributing to the observed scores**. It does not identify an internal neural
mechanism, prove a specific optimizer bug, or establish that four epochs suffice.

## Seeds are controls, not assigned difficulty levels

At initialization 7, changing order7→19 changes confirmation 54→56. At
initialization19, the same change changes 47→79. Initialization19 is worse
than initialization7 under order7, but better under order19. This rank reversal
shows a substantial initialization/order interaction in this exploratory 2×2.
It does not support labelling seed7 intrinsically defective or seed19 universally
good. The matrix has only two crossed initializations and no independent repeats.

## What improved, and what did not

Across the three matched seed blocks:

| Raw evaluation | Baseline correct | Lower-LR correct | Observations per condition |
|---|---:|---:|---:|
| Confirmation | 211 | 284 | 336 |
| Balanced changed payload | 60 | 91 | 96 |
| Crossed query evidence | 67 | 103 | 288 |
| Memory manipulations | 38 | 75 | 192 |
| Evidence row order | 321 | 425 | 672 |
| Novel aliases | 55 | 40 | 336 |
| Novel instruction template | 23 | 33 | 336 |
| Novel one-hop chains | 2 | 0 | 48 |
| Two-hop chains | 0 | 0 | 48 |
| Three-hop chains | 0 | 0 | 48 |
| Two-hop plus engineered first-hop clamp | 0 | 0 | 96 |

Lower LR is not a general repair: alias performance regresses and instruction
template performance remains low. Grammar results must be kept separate. For
example, seed19 crossed-query raw performance improves 31→38/96, while its
grammar comparison worsens 49→39/96. Quoting the grammar numbers as primary raw
scores would reverse this conclusion. Grammar decoding does not rescue the
lower-LR three-hop failures (0/48).

## Where the learner first diverges

For the 48 lower-LR three-hop episodes, **46 first actions are RETRIEVE despite
Memory=0**, and two are ABSTAIN. All fail before a correct answer. One explicit
seed19 trace contains:

```
Follow 3 links from A. Query=0 Memory=0
A=E@8701/map; E=J@8702/map; J=O@8703/map
```

All required facts are already in the rendered prompt; the independently
verified answer is O. The learner's first action is RETRIEVE with recorded
confidence 0.9994087, yielding invalid_action. This localizes the observable
failure to the first policy decision. It does not reveal whether the network
failed to extract relations, interpret the task, bind values, or enforce action
availability internally.

This is **not only a multi-hop-depth problem**: all lower-LR novel one-hop chain
episodes also fail. The engineered first-hop clamp changes the public question
and is not a direct intervention on the network's internal representation;
its failure does not by itself prove an internal binding defect.

On the distinct 112-world validation diagnostic, all lower-LR models select
QUERY and RETRIEVE correctly on the 16 initial missing and memory cases and
answer all 16 memory-return states correctly. After a correct query, missing
states are correct 14/16, 16/16 and 16/16; conflict-return states 16/16 for all
three. Yet known-value initial states score only 2/16, 5/16 and 3/16, and route
initial states 3/16, 5/16 and 5/16. Thus “the learner never incorporates returned
evidence” is too broad: incorporation works in these recorded states, while
direct known-value selection and route generalization remain weak. Teacher-state
diagnostics are not interchangeable with autonomous interactive episode scores.

**CAUSE UNRESOLVED** for the remaining grounding/composition failure. Competing
hypotheses include surface-conditioned strategy selection, action-availability
generalization, relational/value binding, inadequate task supervision and
insufficient optimization/exposure. No one of these is established by a trace.

## Costs and next decision

Recorded controller duration before finalization: **6688.57 seconds /111.48
minutes**. Summed worker reservations: **13088.61 seconds /218.14 GPU-minutes**,
including profiles/recovery/audits. These are process timings, not GPU kernel
utilization or paid-cloud cost. The evidence-row training intervention phase
was not executed; do not treat it as a negative result.

**Next experiment:** keep S20 and use 1e-4 as the strong comparison recipe,
retaining a labelled original-LR comparator. First isolate public task/action
grounding using matched one-hop, two-hop and three-hop worlds, with a gold-free
availability-only action mask as a separate engineered intervention. Prohibit
RETRIEVE only when Memory=0 and QUERY only when Query=0; leave all answer letters
and ABSTAIN available. If this rescues correct answers, invalid action selection
is a contributing cause under that intervention. If it merely substitutes wrong
answers/abstention, the remaining evidence/answer mapping is still unresolved.

Pair this with semantically matched one-hop question rewrites and symbol/row
counterfactuals; the one-hop failure should be explained before attributing
everything to deep composition. Add matched route-supervision/availability
contrast training only after that diagnosis, preserving compute/exposure and
all three seeds. Success must survive novel surfaces, not only the normal format.

**Next execution:** preserve/download the existing FULL ZIP if frozen-checkpoint
tests are desired: it contains weights and optimizer generations absent from
this analysis ZIP, allowing diagnosis without retraining. Check its final
delivery receipt and hashes before use. Do not overwrite or promote the old
models, discard negative cohorts, select only the best seed, or scale to 2B on
the assumption that this remaining failure is explained. Larger models remain
part of the project direction; this study does not establish a capacity ceiling.

## Reproduce this analysis

```
python -m tools.analyze_elendril_v717_results --zip "<owner-analysis.zip>" --output "<new-metrics.json>"
```

The analyzer performs no model inference or training and does not extract or
execute archive contents. An optional `--train-file` reuses a verified generated
train.json rather than regenerating it. [Curated metrics](METRICS.json) retain
all raw/grammar cohort scores, diagnostic trajectories, matched comparisons,
source/data identities and the archive hash. Original owner archives stay outside Git.
