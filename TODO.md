# An-Ra — What We Do Next

Updated: 2026-07-20

This is the single short forward plan. `PROGRESS.md` records what happened,
`ENGINEERING_LOG.md` preserves evidence, and `V4_ARCHITECTURE_GATE.md` defines
the frozen model contract. This file says only where we go next.

## Where we are now

- [x] Build the V4 foundation: one 181,132,071-parameter model, one 32,768-token
  tokenizer, AdamW, deterministic seed 1301, phase-gated native systems, and
  checkpoint schema 9 with exact optimizer/RNG/data-cursor resume.
- [x] Correct fundamental defects including RoPE layout, residual initialization,
  composed control bounds, configuration wiring, AMP false progress, and
  session-restarting data sampling.
- [x] Make MTP executable as a real +2/+3-token training objective and prove one
  bounded full-model GPU step. MTP is implemented but not yet proven better.
- [x] Add executable MoE and curriculum pilot paths without placing them in the
  first baseline.
- [x] Add verified-process supervision, reversible hash-bound LoRA/DoRA capability
  extensions, and an inspectable adaptive reasoning-budget policy.

## The direction

We will build one strong trainable core, then add intelligence systems through
explicit interfaces. A subsystem is not accepted because its file exists or a
smoke test runs. It enters the model only when it improves a frozen baseline in
a controlled comparison and can be removed or rolled back safely.

## Execution order

### 1. Finish the trainable foundation

- [x] Recover and audit the corpus tail; approve the final data volume.
- [x] Publish licensed, deduplicated, decontaminated, immutable V4 train and
  validation shards with exact hashes and source mix.
- [x] Prove one full-context optimizer update on the local RTX 4050.
- [ ] Repeat that bounded update under the corrected Phase-A source policy and
  prove exact kill/restart recovery from its signed checkpoint.
- [ ] Train the dense V4 seed-1301 baseline and freeze its checkpoint lineage.
- [ ] Measure throughput, memory, losses by source, perplexity, generation
  coherence, reasoning, retrieval, and verifier-grounded behavior.

### 2. Build and select intelligence subsystems

- [ ] Compare MTP with dense V4 at identical data, tokens, optimizer, seed, and
  compute; promote it only if held-out capability improves.
- [ ] Evaluate MoE, MoD/native routing, RIM, ESV, DSTP, and HAL independently;
  remove inert mechanisms and keep only measurable positive contributors.
- [ ] Connect adaptive reasoning budgets to real retrieval, decomposition,
  candidate generation, correction, and verifier calls with strict limits.
- [ ] Strengthen memory/retrieval, self-correction, agents/tools, and capability
  adapters behind common provenance, authorization, evaluation, and rollback
  contracts.
- [ ] Pilot SSM/hybrid attention, latent reasoning, vision, world models, trained
  retrieval, and self-development one at a time. Invent or adopt better
  technology when it serves An-Ra’s purpose better than the current subsystem.

### 3. Post-train and prove the resulting model

- [ ] Run SFT, RLVR, STaR, DPO, and self-distillation as isolated ablated stages.
- [ ] Complete private capability, contamination, adversarial, calibration,
  identity-continuity, and regression evaluation.
- [ ] Prove cluster telemetry, preemption recovery, chaos behavior, a 24-hour
  soak, canary deployment, and rollback before release.

### 4. Rebuild documentation after implementation stabilizes

- [ ] Recreate `docs/ARCHITECTURE.md` from the final real code paths, subsystem
  states, inputs, outputs, checkpoints, and evidence—not from proposals.
- [ ] Recreate `docs/DEVELOPER.md` with the final setup, commands, APIs,
  configuration, testing, training, and extension contracts.
- [ ] Recreate `docs/WALKTHROUGH.md` as the updated end-to-end story from data to
  tokenizer, training, checkpoint, inference, memory, agents, proof, and release.
- [ ] Reconcile `README.md`, `PROGRESS.md`, the V4 gate, and the architecture
  explorer so every surface describes the same system.

## Inputs only the owner can provide

- [ ] Approve corpus/storage scope and provide GPU or Colab training time.
- [ ] Hold the signing authority for real launches and promotions.
- [ ] Decide product priorities when two valid capability directions compete.

## Finished means

The repository is ready only when clean immutable data produces a resumable,
useful V4 checkpoint; every promoted subsystem wins measured comparisons; and
the full model passes behavior, safety, observability, deployment, and rollback
gates. Compilation and one-step canaries are necessary evidence, not completion.
