# An-Ra V4 Architecture Gate

Status: architecture-frozen for the current 181M lineage; capability not promoted
Last audited: 2026-08-11
Checkpoint contract: schema 9 / `anra_v4_rope_interleaved_v1`

## Verdict

The canonical V4 geometry is internally coherent and appropriate for a single
T4-class training session. It is now the only selectable model profile. The
gate does not claim that an untrained architecture is intelligent; it proves
that the model to be trained has consistent tensor geometry, positional math,
initialization, phase boundaries, checkpoint identity, and a plausible memory
configuration.

## Frozen geometry

| Property | V4 contract | Engineering reason |
| --- | ---: | --- |
| Parameters | 181,132,071 | Large enough to test the complete system while remaining practical on a 16 GiB T4 |
| Vocabulary | 32,768 | Useful coverage without allowing the embedding table to dominate the model |
| Width | 896 | Tensor-core-friendly width with 64-dimensional attention heads |
| Layers | 18 | Balanced depth/width for the parameter and compute budget |
| Query / KV heads | 14 / 2 | Grouped-query attention; seven query heads share each KV head |
| SwiGLU width | 2,432 | Rounded 8/3 expansion, divisible by 64 |
| Context | 2,048 | The useful maximum supported by the initial T4 campaign |
| Attention | QK-Norm; 1,024-token local window; every fourth layer full | Controls quadratic cost while preserving periodic global communication |
| Embeddings | Tied input/output | Saves 29.4M duplicate output parameters |
| Dropout | 0.0 | Data volume and explicit evaluation determine regularization; no train/inference mismatch |
| Initialization | depth-scaled residual v1 | Prevents residual variance from compounding across 18 layers |
| Microbatch / accumulation | 1 / 32 | Preserves 65,536 tokens per optimizer update without the former T4 OOM risk |
| Foundation optimizer | AdamW, beta 0.9/0.95, decoupled matrix-only weight decay | Standard, exactly resumable baseline; alternative optimizers must win a matched pilot |
| Schedule | 2% linear warmup, then cosine decay to 1e-5 | One checkpointed schedule with no hidden adaptive overlay |
| Canonical run seed | 1301 | A stable replay address, not an intelligence or quality parameter |

## Corrections made by this gate

1. **Rotary coordinates were repaired.** `_rotate_half` rotates adjacent pairs,
   but the cache previously concatenated frequencies. V4 now repeats each
   frequency for its coordinate pair, restoring a valid norm-preserving rotation.
2. **The rotary layout is checkpoint-bound.** Schema 9 checkpoints must declare
   `anra_v4_rope_interleaved_v1`; missing or older semantics are rejected rather
   than silently resumed.
3. **Attention control is bounded after composition.** The product of ESV,
   DSTP, HAL, and per-layer controls is clamped to `[0.5, 2.0]`, preventing
   collapsed or nearly uniform attention from multiplicative extremes.
4. **Native controls start neutral.** Residual depth, layer temperature, DSTP,
   ESV, and RIM begin as identity/no-op behavior. Learned layer structure must
   emerge from evidence instead of an arbitrary temperature curve.
5. **Configuration is executable truth.** `d_ff` and `rope_base` now reach the
   actual model instead of being ignored. Invalid head grouping, feed-forward
   alignment, router layer indices, and RoPE dimensions fail before training.
6. **T4 memory risk was reduced.** Microbatch 4 was replaced by microbatch 1
   and accumulation 32, retaining the same effective token batch.
7. **Resume is now a mathematical continuation.** Schema 9 captures and
   restores Python, NumPy, Torch, CUDA, DataLoader, optimizer, scheduler,
   mixed-precision, raw-sampler cursor, and training-recipe state. Missing or
   incompatible state blocks training resume instead of silently reseeding or
   resetting optimizer moments.
8. **Corpus sampling no longer restarts each session.** The raw-corpus sampler
   is counter-based by `(algorithm, seed, position)`, so a resumed run yields
   the exact remaining suffix and its cursor must agree with consumption
   evidence.
9. **The optimization baseline was simplified.** AdamW plus cosine warmup/decay
   is the only foundation algorithm. The unproven dynamic-regret overlay was
   removed from the base trainer; Adafactor, Muon, GaLore, and other candidates
   remain isolated experiments.
10. **AMP overflow cannot create false progress.** A skipped mixed-precision
    optimizer step no longer advances the scheduler, checkpoint step, token
    count, sampler cursor, or subsystem balance state.

## Canonical versus gated systems

The dense decoder, GQA, QK-Norm, hybrid attention, SwiGLU, RMSNorm, tied
embeddings, and depth-scaled initialization are canonical from the first token.
MoD, RIM, ESV, DSTP, and HAL are structurally present but phase-gated. Phase A
is a true dense baseline with those parameters frozen. Later phases require an
explicit subsystem recipe. MTP, MoE, HAL, cognitive extensions, and moonshots
remain off the critical path and cannot enter a checkpoint implicitly.

## Intelligence foundation attached to V4

The foundation now distinguishes three mechanisms that were previously easy
to confuse:

1. **Learning signal.** Verified DFC rows use the
   `verified_dfc_process_spans_v1` objective. Only tokens inside complete
   `<hyp>`, `<verify>`, `<err>`, and `<upd>` spans receive the bounded 1.25x
   weight. Ordinary data, validation data, malformed spans, and truncated
   spans are unchanged. The objective is checkpoint-recipe-bound so a resume
   cannot silently change it.
2. **Attachable capability.** `anra/extensions.py` is the one reversible
   LoRA/DoRA contract. It freezes the 181M base, requires explicit target
   modules, stores adapter tensors only, and binds each capability to the base
   checkpoint hash, tokenizer hash, model profile, source commit, and exact
   tensor shapes. A failed load detaches cleanly and cannot remain marked
   active. Nothing is merged into the immutable base checkpoint.
3. **Adaptive use of compute.** `inference/reasoning_budget.py` emits a bounded,
   inspectable plan for direct response, verification, retrieval/decomposition,
   or search/verification. `/reasoning/plan` does not execute actions or alter
   weights. Missing retrievers/verifiers are reported as blockers rather than
   simulated.

These mechanisms make verified learning, later capability extension, and
inference-time effort explicit. They do not establish that a trained model is
intelligent. Their value must be measured after the dense V4 baseline exists.

## Verification evidence

- Actual model construction: exactly 181,132,071 parameters.
- Full-model CPU probe: finite logits with shape `[1, 4, 32768]`.
- RTX 4050 dense micro-canary: one real BF16 AdamW update at sequence length
  64 completed in 5.48 seconds with 3,499.82 MiB peak allocated memory, finite
  CE/total loss (`10.546875` / `10.558056`), and finite pre-clip gradient norm
  (`64.08894`). This exercises the complete 181,132,071-parameter model, not a
  reduced proxy.
- Seed replay: two fresh seed-1301 constructions produced the same model
  fingerprint (`5d6854db...`), initial loss (`10.566532`), and sampled logits.
  Seed 1302 produced a different fingerprint (`8cbcff...`) and logits. Its
  lower random initial loss (`10.479838`) is evidence that initial loss must
  not be used to choose a "better" seed.
- RTX 4050 MTP research micro-canary: the full 182,739,495-parameter candidate
  completed one BF16 AdamW update at sequence length 64 in 4.08 seconds with
  3,527.00 MiB peak allocated memory, finite base CE (`10.611328`), weighted
  MTP loss (`2.084721`), and finite gradients. This proves local feasibility,
  not a quality improvement; MTP remains a matched pilot.
- RTX 4050 capability-extension canary: the full 181,132,071-parameter V4 base
  was frozen and DoRA was attached to 54 Q/V/down projections. Exactly 919,296
  parameters were trainable. One BF16 AdamW step at sequence length 16 passed
  in 5.80 seconds with 1,300 MiB peak allocation, finite total loss
  (`10.655715`) and gradients. The first run exposed and then verified the fix
  for GPU device/dtype inheritance. No dataset or checkpoint was written.
- Layer layout: local windows on 14 layers and full attention on layers 4, 8,
  12, and 16 (one-based).
- Focused architecture, training-contract, data, optimizer, and resume suites
  pass locally; exact commands and scope are recorded in the engineering log.
- Explicit tests cover RoPE pair phases, norm preservation, temperature bounds,
  local-window exclusion, GQA/QK-Norm, neutral native controls, forward/backward
  gradients, phase freezing, checkpoint rejection, complete RNG replay, exact
  schema-9 resume, counter-sampler suffix replay, optimizer selection, and
  preflight consistency.

## What remains unknown until training

Architecture inspection and micro-canaries cannot prove language quality,
useful reasoning, full-context memory, or sustained training throughput. The
first seed-1301 run must still measure peak T4 memory at the real 2,048-token
context, tokens per second, loss by data class, gradient norms, held-out
perplexity, and generation coherence. Any future geometry change requires a
new architecture version and a new scratch checkpoint; V4 must never be
mutated silently after training begins.

The local CUDA path is now visible: PyTorch 2.11.0+cu128 detected the NVIDIA
GeForce RTX 4050 Laptop GPU (6,141 MiB). Only bounded sequence-64 updates were
run; no checkpoint or training campaign was created. The 6 GiB laptop canary
does not substitute for the canonical 2,048-token T4 run.

The bounded check is reproducible from the repository root through
`python -m scripts.run_v4_gpu_canary --variant dense --repeat 2`. MTP uses
`--variant mtp`; the reversible extension path uses `--adapter dora`. The
command performs exactly one update per repeat, writes no
checkpoint, and requires `--allow-large-context` above 256 tokens so a local
canary cannot silently become a long or high-memory job.

## Research-risk decision

The foundation deliberately separates **stable machinery** from **risky
learning hypotheses**. Dense V4 is the control that makes an invention
measurable. MTP is the first architecture-risk candidate because it teaches
the residual stream to predict more than the immediate next token while adding
no required inference path, and the real 181M-class candidate fits locally.
It will not replace dense V4 from a one-step canary. Promotion requires the same
tokenizer, data order, token budget, optimizer, evaluation, and seed address in
a dense-versus-MTP comparison; replication is required only if the first
comparison shows a meaningful gain. MoE, native routing, HAL, and moonshots do
not enter that comparison simultaneously.

## Trained-checkpoint audit — 2026-08-11

The downloaded `anra-v4-current-full-resume.pt` is a structurally valid SFT
child, not proof that the model is ready. It exact-loaded all 393 stored model
tensors into the canonical 181,132,071-parameter V4 geometry. Its checkpoint
declares schema 9, V4 tokenizer semantics, assistant-only SFT, step 5,000, no
skipped mixed-precision updates, and no approved MTP, MoE, MoD, RIM, ESV, DSTP,
HAL, or moonshot subsystem. The observed weakness therefore cannot honestly be
blamed on a missing experimental subsystem.

The SFT child reports validation improvement from `1.8391` to `1.3916` over
6,845 training conversations, but a real RTX 4050 probe failed basic behavior:
it did not compute `23 + 34`, produced false statements about the Sun, emitted
an incorrect addition function, and gave incoherent identity text. Sampling
changes did not repair these failures. This demonstrates the exact failure mode
the repository must guard against: lower validation loss is not equivalent to
useful language behavior.

Two architectural corrections now follow from this evidence:

1. Training, SFT smoke evaluation, and interactive chat share the versioned
   `anra-v4-sft-prompt/v1` renderer. Prompt bytes can no longer drift between
   the learning and serving paths.
2. SFT promotion now requires task-specific behavior checks. Non-empty and
   merely different outputs can no longer approve full SFT. The gate checks the
   fixed suite for a correct arithmetic result, real addition code, valid JSON,
   uncertainty, grammar correction, explicit steps, and appropriate dialogue.

The checkpoint remains useful as a research artifact, but it is not a release
candidate. The most likely limiting factor is that the 181M foundation saw only
roughly one pretraining token per parameter before a comparatively long SFT
stage. The correct next investment is continued dense V4 pretraining with new,
audited tokens to the 500M-token milestone, then 1B and approximately 3.6B if
the milestone curves remain healthy. Run the fixed behavior gate at each
milestone and build a new SFT child only after the parent demonstrates basic
language, factual, math, and code competence.

Do not change the 181M tensor geometry in place: doing so would invalidate its
optimizer and checkpoint lineage. Architecture inventions remain isolated
pilots from a frozen parent. A future 40–50 GPU deployment may use DDP/FSDP only
when those GPUs share a low-latency cluster; unrelated Colab or Kaggle sessions
remain exact checkpoint-baton workers and must not average gradients.

## Post-checkpoint architecture corrections — 2026-08-11

The full implementation audit found and corrected several issues that could
have invalidated the next expensive run:

- The 181M total is now reported as **180,093,312 dense parameters** plus
  **1,038,759 installed native pilot/control parameters**. The latter are part
  of the existing checkpoint ABI but are not evidence that MoD, RIM, ESV, or
  DSTP is active or useful.
- `--mtp` and `--moe` can no longer mutate `anra-v4-180m` in place while keeping
  its registered name. A candidate must have a distinct experimental profile,
  exact parameter count, frozen-parent hash, initialization artifact, and
  rollback evidence.
- Progressive 500M unfreezing now uses real `requires_grad=False` boundaries
  and clears inactive gradients to `None`. Zero gradients were insufficient:
  AdamW weight decay and optimizer moments could still move allegedly frozen
  inherited tensors.
- Growth parity now ignores padding and gates the real next-token distribution
  using cosine similarity, KL divergence, and top-1 agreement over a broader
  prompt set. Cosine alone could approve a distribution-changing child.
- A growth artifact now preserves the parent stage and complete lineage,
  including SFT hashes when the explicit policy is `post-trained-parent`.
  Growing an SFT child can no longer silently relabel it as generic pretraining.
- A fresh 500M trainer cannot start from a teacher and manifest alone. It must
  consume the parity-gated child initialization artifact; AdamW then starts
  fresh under the declared stabilization schedule.

The review also confirmed that same-host 40–50 GPU training is **not yet an
operational capability**. At audit time the distributed module only estimated
a profile; the canonical trainer still does not consume a distributed sampler,
reduce global metrics, fence rank-zero checkpoint publication, or save every
rank's optimizer/RNG state. Until a two-GPU kill-and-resume proof passes,
separate Colab/Kaggle workers remain sequential checkpoint-baton workers and a
large cluster must not be launched.

The first distributed implementation layer now exists without weakening that
gate. `training.distributed` defines an explicit torchrun/NCCL topology,
collective helpers, rank-zero result fencing, and the versioned
`anra-ddp-contract/v1`. The canonical counter/permutation samplers expose
absolute-position lookup, and `RankStridedSampler` partitions one global suffix
without overlap or padding. `scripts.run_ddp_rehearsal` exercises those pieces
with a tiny dropout model, real `no_sync` gradient accumulation, one global
cursor, isolated per-rank training RNG, and exactly one atomic rank-zero
checkpoint at each completed optimizer boundary.

On a real two-GPU host, first create the uninterrupted four-step reference:

```powershell
torchrun --standalone --nproc-per-node=2 -m scripts.run_ddp_rehearsal `
  --checkpoint output/v2/ddp-uninterrupted.pt --steps 4 --accumulation 2
```

Then create a separate lineage that completes step one, injects a synchronized
failure after the first microbatch of step two, and resumes from the protected
step-one checkpoint. The failed command is expected to exit nonzero; no partial
gradients or sampler movement may enter its checkpoint:

```powershell
torchrun --standalone --nproc-per-node=2 -m scripts.run_ddp_rehearsal `
  --checkpoint output/v2/ddp-resumed.pt --steps 4 --accumulation 2 `
  --fault-after-microsteps 3 --fault-rank 1

torchrun --standalone --nproc-per-node=2 -m scripts.run_ddp_rehearsal `
  --checkpoint output/v2/ddp-resumed.pt --resume --steps 4 --accumulation 2
```

Finally, the comparator recomputes each checkpoint fingerprint and requires an
exact match across model, optimizer, cursor, canonical consumed-index history,
topology/lineage contract, and every rank's Python, NumPy, CPU, and CUDA RNG:

```powershell
python -m scripts.compare_ddp_rehearsals `
  --uninterrupted output/v2/ddp-uninterrupted.pt `
  --resumed output/v2/ddp-resumed.pt
```

This is a correctness rehearsal, not permission to run the 181M model on a
large cluster. Canonical trainer integration follows only after uninterrupted
versus resumed rehearsal states compare exactly.

The Drive `latest_training_failure.log` dated 00:57 is historical. It recorded
a compact pack whose 161,133 unique windows were rounded to 161,136 for an
eight-microbatch optimizer boundary. Commit `acb1244` makes the final partial
optimizer update from only the remaining unique windows. The Drive canonical
pointer was subsequently replaced at step 10,400, so the old log must not be
interpreted as the state of the newer checkpoint.
