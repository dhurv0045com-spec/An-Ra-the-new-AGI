# An-Ra Execution Ledger

Updated: 2026-08-11

This is the one short forward ledger. Completed claims mean the code and focused
contracts exist; they do not mean a useful model has already been trained.

## Foundation now implemented

- [x] One operational line: V4 tokenizer (32,768 vocabulary), dense
  181,132,071-parameter model, AdamW, context 2,048, and routine seed 1301.
- [x] Signed `anra-training-contract/v4` binds source commit, tokenizer, data,
  model, seed, token window, parent checkpoint, destinations, and resource limits.
- [x] Schema-9 full-resume checkpoints preserve model, optimizer, scheduler,
  scaler, RNG, sampler cursor, step, tokens, and complete lineage.
- [x] Concurrent durability uses immutable 128 MiB chunks, resumable verified
  uploads, canonical pointers, protected replicas, and two-generation retention.
- [x] Compact FP16 artifacts are model-only and cannot be accepted for training
  resume.
- [x] The cluster has one fenced canonical writer plus standby, evaluator,
  data-builder, architecture-pilot, and archive roles. Public cross-VM gradient
  averaging is retired; same-host DDP/FSDP remains a future gated mode.
- [x] Deterministic 50M-170M token packs resume at signed global token boundaries
  while resetting only the new pack's local permutation cursor.
- [x] The 499,880,031-parameter growth child is registered with parent-hash
  binding, attention-mode preservation, identity insertion, padding-masked
  cosine/KL/top-1 parity proof, complete parent-stage lineage, fresh AdamW state,
  teacher alignment, and real AdamW-safe progressive unfreezing.
- [x] MTP, MoD, RIM, ESV, DSTP, HAL, MoE, and moonshots are classified and cannot
  silently enter the dense baseline. Promotion requires comparative evidence.
- [x] SFT/RLVR/STaR/DPO data and audit contracts, self-correction orchestration,
  reversible adapters, shared evidence, and rollback gates exist as pilots.
- [x] Registered V4 profiles reject in-place MTP/MoE mutation; experimental
  architectures require distinct names, exact counts, and parent-derived
  initialization evidence.
- [x] Parameter reports separate 180,093,312 dense parameters from 1,038,759
  installed but unpromoted native pilot/control parameters.

## Current evidence and next live execution

- [x] Train and preserve a V4 foundation lineage to roughly 180M pretraining
  tokens and produce a separate signed 5,000-step assistant-only SFT child.
- [x] Load the SFT child exactly on the local RTX 4050 and verify its checkpoint,
  tokenizer, tensor, and GPU contracts.
- [x] Replace the former non-empty-output SFT approval with a task-specific
  behavior gate shared by training and the local evaluation app.
- [ ] Treat the current SFT child as research-only: it improved validation loss
  but failed basic arithmetic, factual, code, and identity behavior on 2026-08-11.

- [ ] Freeze and push the exact An-Ra and GPU Cluster commits used by workers.
- [ ] Configure owner-held manifest signing, the Google Drive active vault,
  laptop archive, and optional OneDrive cold replica.
- [x] Run protected T4 sessions and verify resumable full checkpoints in the
  shared training home.
- [x] Resume the exact checkpoint from another authorized notebook account.
- [ ] Continue the dense 181M foundation lineage from the protected step-10,400
  checkpoint through 500M, 1B, and 3.6B cumulative
  tokens, using lightweight health checks and milestone behavioral evaluation.

## After the first useful 181M checkpoint

- [ ] Compare dense continuation with MTP under identical data, seed, optimizer,
  and token budget. Test MoD, RIM, ESV, DSTP, and transformer-HAL independently.
- [x] Wire the V4 SFT preparation path: receipt-bound audited JSONL builder,
  source-group-disjoint held-out validation, assistant-only numerical GPU
  trainer, portable signed lineage, separate child checkpoint, parent-versus-
  child validation evidence, and bounded T4 pilot notebook.
- [x] Build a signed, license-receipted SFT corpus and complete pilot and full
  child training. The result remains unpromoted because behavior failed.
- [ ] After a capable foundation milestone, build a new SFT child with the same
  strict behavior gate. RLVR/STaR and DPO remain unavailable for promotion
  until their evidence gates and training runs are real.
- [ ] Add retrieval, memory, correction, tools, and adapters in that order, with
  permissions, budgets, verification, provenance, and rollback.
- [ ] Compare exact-float and TurboQuant KV caches on the same trained V4
  checkpoint. Promote compression only if long-context capability,
  output-distribution drift, peak VRAM, and tokens/second jointly pass; the
  exact cache remains the rollback.
- [ ] Grow the frozen 181M parent to 500M, prove real-checkpoint parity, provide a
  32 GiB hot vault for its larger resumable states, and only then train the child.
- [ ] Pilot sparse compute, hybrid attention/SSM, multimodality, world models,
  robotics, and sandboxed self-development one question at a time.

## Not completed yet

- No V4 checkpoint has yet passed the fixed useful-language behavior gate.
- No optional architecture subsystem has earned promotion into the baseline.
- SFT was executed, but it is not accepted; no RLVR/STaR/DPO or 500M growth run
  has occurred.
- Same-host DDP/FSDP is deliberately blocked until the An-Ra runtime implements
  and proves it; separate Colab sessions will never synchronize gradients.

## Final completion gate

The system is complete only when protected checkpoints resume exactly, the
181M model produces coherent behavior rather than merely low loss, every enabled
subsystem wins a controlled comparison, the growth child preserves its parent,
and release evidence includes evaluation, rollback, and audit history.
