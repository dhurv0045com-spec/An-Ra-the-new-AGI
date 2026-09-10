# CYR-GPU-012 / R1 — DESIGN REASONING

## Why this experiment exists

CYR-GPU-011 established a large representation-associated capability-formation gap, but its compact-vs-production comparison simultaneously changed vocabulary size, token segmentation, numeric atomization, special-token layout and tied embedding/output parameter burden. Calling that result “the 24k tokenizer is bad” would exceed the evidence.

R1 changes **one causal bundle**: the number of tied embedding/output classes. Arithmetic strings are encoded to the exact same active IDs in every arm, so the network sees the same sequence length and same discrete input sequence. Batch, examples, ordering, objective and optimizer are also fixed.

## Why shared initialization is copied rather than merely using the same seed

Changing embedding shape can consume a different number of RNG draws during initialization and thereby shift later block weights even when the seed is identical. R1 prevents that confound: all non-embedding parameters and the active 19 embedding rows are copied from a V19 reference model; only extra unused embedding/output rows remain arm-specific. A SHA-256 receipt checks that the shared state is identical before training.

## Why 512k rows rather than forcing every arm to 1.152M

The target is causal attribution under a strict three-hour wall, not a duplicate of V11. V11 already showed a compact signal around 517k rows and production floor even at 1.152M. Two matched causal arms at 512k provide much more information per minute than spending the entire wall rerunning a known production-BPE null.

The endpoint is fixed before outcomes and is identical across arms. If the mechanism remains ambiguous, a later experiment can extend only the informative arm rather than redesigning this one post hoc.

## Why fixed batch 64

Batch size affects optimization dynamics. Hardware resolution therefore may change **number of seed pairs**, never batch or the per-arm semantic endpoint. If batch64 cannot complete one matched pair under the wall, R1 refuses to run scientifically.

## Why two seeds are preferred but one is admissible

The prior compact transition is seed-variable. Two matched pairs are the strongest result that plausibly fits the T4 wall; one pair is still a useful developmental mechanism screen but cannot receive a replicated label. The resolver makes this choice from calibration only, before outcomes.

## Why V4096 is optional

A middle vocabulary size can reveal dose response, but it must never displace a second primary 19-vs-24576 replication. It launches only from genuine remaining calibrated wall budget after primary pairs are protected.

## Why no new architecture mechanism is introduced

The existing Cymek V5 4L/128w bridge can already partially generalize under compact representation. Adding recurrence, MoE, special numeric heads or auxiliary objectives would destroy attribution. R1 must first learn whether a simpler representation burden explains the observed failure.
