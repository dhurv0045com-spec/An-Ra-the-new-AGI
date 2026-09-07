# ARKENSTONE MASTER GPU CAMPAIGN — BINDING PLAN

This plan governs the next single-session Colab/T4 campaign. It is committed before any new campaign result is produced.

## Scope
Run only on branch `Arkenstone`. Existing historical results are inputs, not targets to reproduce selectively.

## Campaign A — ARK-007R fresh-checkpoint replication
Question: does the post-G90 low-LR protection effect survive on fresh independently acquired T2 checkpoints?

- Acquisition seeds: 909, 1010, 1111.
- Canonical T2 manifest SHA256: `0dd9305697045b0fbf4e7f268b46a4d7276e4794af5d78b60e999df914ae4236`.
- Model: existing Micro 4L/128w/4H compact-vocab harness.
- Acquisition optimizer: AdamW, lr=1e-3, betas=(0.9,0.95), eps=1e-8, wd=0.1, batch=64.
- Sustained G90: 3 consecutive evals >=0.90 at 200-step cadence.
- Distinguish `g90_onset_step` from `g90_confirmation_step`; treatment starts only after confirmation.
- Continuation seeds: 2701,2702,2703,2704.
- Each continuation order is materialized once and hash-bound; HIGH and LOW forks consume the identical indices.
- Arms: HIGH lr=1e-3; LOW lr=1e-5.
- Post-confirmation budget: 6000 optimizer steps.
- Collapse90: first 3 consecutive treated evaluations below 0.90.
- Primary evidence: paired collapse counts, risk difference, RET90, OOD area, final OOD, parameter displacement.
- No claim of universal optimizer law; task-family/micro-scale only.

## Campaign B — ARK-009 learnable non-arithmetic transfer
Question: can the Micro model acquire a clean variable-binding/retrieval task, and if so does low-LR retention protection transfer?

Task: synthetic symbolic key-value retrieval, not arithmetic. Each prompt contains 3 distinct bindings selected from six keys and six values, followed by a query for one key. Train/test prompts are exact-instance disjoint, value-balanced, and query-balanced. A query-swap diagnostic asks a different key from the same facts to expose query-blind shortcuts.

- Acquisition seeds: 1201, 1202.
- Training examples: 1200; held-out examples: 300; fixed task seed 4242.
- Acquisition lr=1e-3, batch=64, eval every 200, max 24000 steps.
- Qualification: sustained test exact >=0.90 AND query-swap exact >=0.85.
- If no seed qualifies, record `TRANSFER_BLOCKED_BY_ACQUISITION` and do not run retention forks.
- If a seed qualifies, snapshot at confirmation and run continuation seeds 3701..3706 with paired HIGH=1e-3 / LOW=1e-5 for 6000 treated steps.
- Same retention metrics and paired causal interpretation as Campaign A.

## Campaign C — ARK-010 recovery after collapse
Question: after a HIGH-LR continuation has already met the preregistered collapse90 criterion, can lowering LR to 1e-5 recover capability, or does it only prevent further damage?

- Source candidates: collapse events produced prospectively in Campaign A.
- At collapse confirmation, save the exact model+optimizer state.
- Continue the same future minibatch tail in two matched forks: HIGH-CONTINUE=1e-3 and RECOVERY=1e-5.
- Recovery budget: 4000 steps, eval every 200.
- Recovery90: first 3 consecutive evals >=0.90 after collapse confirmation.
- Report recovery incidence, time-to-recovery, OOD area, final OOD.
- If fewer than 2 collapse events exist, classify `INCONCLUSIVE_LOW_EVENT_RATE` rather than forcing a result.

## Execution integrity
- GPU/CUDA only; T4 recommended. TPU/XLA is not required.
- The notebook must define a safe `device_sync()` and a backward-compatible `xla_sync()` wrapper so the previous NameError cannot recur.
- All result files include this plan commit SHA, device, torch version, task/manifest hashes, seeds, order hashes, onset/confirmation/treatment steps, and receipt SHA256.
- Actual supervised-token counts come from the loss mask, never from a hard-coded tokens-per-example constant.
- The notebook is budget-aware and saves partial results rather than fabricating completion.
- Any optional exploratory analysis is explicitly labeled exploratory and cannot change the above confirmatory verdict rules.

## Claim discipline
`DEMONSTRATED` requires executed receipts. `REPLICATED` requires fresh independent acquisition checkpoints. Transfer is not demonstrated unless Campaign B both acquires the non-arithmetic capability and then shows the retention effect under matched continuations.
