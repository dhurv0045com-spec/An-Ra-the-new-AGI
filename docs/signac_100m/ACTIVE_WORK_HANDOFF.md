# Signac 100M Active Work Handoff

Snapshot checked on 2026-09-24 after the long-running Codex task was terminated.

## Recovery source

- Checkout: `codex/signac-100m`
- HEAD: `b4b285ee` (`feat(signac): prepare 100M TPU qualification package`)
- Latest executable source identity before the corpus-loader continuation: `18f0d41e29733a4870408370e9cb7ef09e2af5c69aea91c317491289e84fe39b` over 128 inventoried files, including the shared checkpoint dispatcher and Kaggle launch notebook.
- The checkout and its uncommitted work survived task termination.
- At recovery inspection: 71 modified tracked paths and 74 untracked files. The tracked diff then contained 7,357 insertions and 935 deletions. These totals include broad Cymek/E0 work and do not correspond one-to-one with the app's change counter.
- After the recovery continuation: 71 modified tracked paths and 76 untracked files, with 7,475 insertions and 954 deletions in the tracked diff.
- After the latest continuation: 71 modified tracked paths and 76 untracked files, with 7,593 insertions and 958 deletions in the tracked diff. No files are staged. These totals still include broad Cymek/E0 work and should not be treated as Signac-only changes.
- No commit or push was made by the continuing task. Preserve the complete working tree; do not reset, clean, or infer that every untracked path belongs to Signac. Several existing research and worktree paths are present alongside Signac files.

## Verification completed in the recovered task

- `git diff --check` passed.
- Python compile checks passed for `signac_100m`, `v5_training`, and `tools/signac_100m_preflight.py`.
- 125 tests passed across `tests/test_signac_kaggle_tpu_canary.py`, `tests/test_signac_100m.py`, `tests/test_signac_training.py`, `tests/test_signac_rsi.py`, and `tests/test_signac_phase1_eval.py`.
- 45 tests passed across XLA adapter RNG, production sampler/microsteps, and distributed checkpoint/resume suites.
- One host-only campaign test passed through the real `xla-development` orchestration, adapter/coordinator interfaces, v2 publication, and fresh-session resume using a one-rank fake runtime. It proves wiring only; there is no TPU, BF16, or multi-rank claim.
- Two focused production-entry tests confirm `execution="xla"` still fails closed and the development lane rejects unsafe update bounds.
- The core now computes shared RoPE angles once per forward (one angle-pair build instead of 40 per M102 forward). Eight model tests pass, including exact FP32/BF16 output and gradient parity against the previous formula and checkpointed-versus-plain full-model gradient parity. The combined model/backend/XLA campaign/adapter run passed 38 tests.
- Production single-step and accumulated-microstep training now use `forward_hidden` with the tied output table and checkpointed 512-position vocabulary-loss chunks. The Kaggle synthetic canary uses the same path. Full-logit parity tests cover masked targets, loss numerator, hidden and tied-weight gradients, CPU BF16 autocast recomputation, and empty-rank zero gradients.
- The consolidated recovery verification passed 110 tests across causal objectives, production updates, the model, the Kaggle canary, the host-only XLA development campaign, checkpoint dispatch, source identity, and Signac preflight contracts.
- Official PyTorch/XLA source rejects `torch.utils.checkpoint` with `use_reentrant=False`; transformer-block and output-loss checkpoint calls now route XLA tensors through `torch_xla.utils.checkpoint` with reentrant mode, while CPU/CUDA use the tested non-reentrant path. A mocked dispatch test checks the selected API and flags. A live Kaggle run is still required to qualify the target wheel and memory behavior.
- Python compile checks and `git diff --check` passed after the completed recovery changes.
- Python compile checks and `git diff --check` passed after these changes.
- The Kaggle preflight notebook parses as valid nbformat 4 JSON with seven cells.
- `python tools/signac_100m_preflight.py --target tpu` returns `BLOCKED` for missing runtime, data, and evaluation receipts, with `training_authorized: false` (verified again after the chunked-loss integration).
- A separate production certificate test could not start because this host Python lacks the optional `tokenizers` package; it failed at import, before campaign code ran.
- The contract was corrected so a rank with zero eligible tokens is permitted only on an all-ineligible terminal short tail; normal measured steady-state work still requires positive supervised tokens per rank.
- Host RSS telemetry is diagnostic and sampled after the final canary hashes. It does not replace XLA peak-memory counters or the 85% qualification gate.
- Update-certificate failures after `optimizer.step()` now poison the live backend: another update and checkpoint serialization are rejected until a valid shared model/optimizer checkpoint is restored. The old receipt is cleared at the start of the attempted optimizer step. Regression tests cover both direct and accumulated updates, plus recovery after restore.
- A Luna review found unconditional carry-forward of post-update hashes unsafe because it would miss out-of-band parameter or Adam-moment edits. The full pre-update hash is retained; the existing 2.44 GB per-rank transfer estimate remains an unresolved target throughput cost.
- Current focused verification: all 18 production-backend tests passed. In the six-suite checkpoint/resume run, 43 tests passed; four failed initially because two campaign fixtures still exposed only the old full-logits API and two Windows temp-directory renames returned access denied. The fixtures now expose the tied-hidden API, and all four affected tests passed on a fresh temporary root.
- The TPU preflight was rerun after the backend change and remains `BLOCKED` on data, evaluation, and runtime receipts, with `training_authorized: false`.

## Remaining qualification work

The branch is an engineering package, not a completed or authorized research run. `docs/signac_100m/READINESS.md` remains the source of truth for gates. In particular, the actual Kaggle TPU session must establish runtime identity, all-rank collectives, BF16 and RNG behavior, measured throughput and memory, numerical parity, and fresh-process checkpoint continuation. The real corpus, sealed/fresh evaluation, Citadel evaluator, and independent evidence audit also remain external prerequisites. Synthetic tests do not pass these gates or establish an AGI capability claim.

The working tree also contains broad Cymek/E0 research changes from the long task. Review them in place and keep the Signac worktree intact while deciding what to commit or push.

## Corpus-loader recovery continuation

- The unsafe working-tree scan in v5_data/corpus_loading.py was replaced with the pinned v5_data/first_party_corpus_manifest.json. It explicitly lists 23 curated research Markdown documents (227,043 source bytes), including the cross-branch experiment ledger and Signac evidence ledger. The loader requires the manifest's pinned SHA-256, validates each path and raw file hash, enforces per-document and aggregate byte bounds, rejects over-limit tokenized documents instead of truncating, and never discovers unlisted files.
- The manifest declares development-canary-only scope, and each resulting data source carries the first-party-development-only category. Dataset qualification rejects that category for production. These internal documents contain evaluation findings; they are not an audited production corpus. The production data gate remains blocked.
- signac_100m.source_identity already inventories JSON under v5_data; its source-tree digest therefore changes with this manifest. A regression test now proves that binding.
- Focused verification: 33 tests passed across the corpus-loader boundary, source identity, dataset qualification, and data pipeline; compileall and git diff --check passed. A word-tokenizer smoke loaded all 23 checked-in documents (30,726 words; largest document 7,752 words). The environment lacks the optional tokenizers package, so the 20,000-token cap has not been checked with the frozen BPE artifact here; overflow will fail closed when the real tokenizer is available.
- The TPU preflight still returns BLOCKED for data, evaluation, and target runtime, with training_authorized: false.
- Current executable source identity after scope enforcement: 12b019209140c39afe1a9d7318f7db450b3314195357163bfa71ec3c9bd2a6ee over 129 inventoried files.
- At the corpus scope-enforcement checkpoint, Git showed 77 modified tracked paths and a 7,810-insertion/1,001-deletion tracked diff. The later XLA continuation work below remains in the same uncommitted worktree and includes broad Cymek/E0 work.

## XLA checkpoint and Kaggle campaign continuation

- `v5_training.production_backend.production_shared_payloads` now detaches model and optimizer state trees and stages their tensors on CPU before `torch.save`. The copy preserves repeated tensor references, container types, and PyTorch state-dict version metadata. This addresses an unverified serialization assumption in the XLA development path; actual Kaggle transfer and serialization still require target evidence.
- The Kaggle notebook has an opt-in, disabled-by-default `RUN_XLA_DEVELOPMENT_CAMPAIGN` profile. It invokes `v5_training.kaggle_xla_development`, which runs the actual M102 `run_campaign` backend on synthetic development-only text for one update, launches a fresh eight-rank group, resumes from the shared checkpoint and per-rank RNG/cursor state, and runs the second update. The entrypoint binds source identity, validates rank/runtime receipts and checkpoint advancement, stores artifacts under `/kaggle/working`, and marks all output as unauthorized for research training.
- The XLA development campaign no longer creates a second 100M model plus Adam state on the TPU just to verify its final checkpoint. It reports `resume_equal: null` with an explicit deferred-verification marker; successful update two in the fresh worker group is the continuation check. The group aggregator verifies each rank receipt's content hash and session label. The two groups must also agree on source, model, data, and runtime identity before the resume is accepted.
- This is same-session integration coverage only. It does not qualify TPU behavior, durable Kaggle Output round-trip, production corpus/evaluation, or `execution="xla"`; the toggle remains off by default.
- Focused verification passed: 34 tests across bucket-supply and receipt validators, production checkpoint backend/resume, XLA development campaign wiring, and source identity. Python compile checks, notebook parsing (seven cells), and `git diff --check` passed. The broader production-entry test file could not run in this host because the optional `tokenizers` package is missing; it raised before campaign code started.
- Latest executable source identity: `53d10ce7374fca80b4e52a932bb57662c4c632b73540bfd13ff240f761c877c1` over 130 inventoried files, including the new Kaggle entrypoint and updated notebook.
- `python tools/signac_100m_preflight.py --target tpu` remains `BLOCKED` on data, evaluation, and target runtime; `training_authorized` is `false`. No Kaggle TPU session was available in this task, and no commit or push was made.

## Evaluation-gate hardening

- The static preflight no longer treats an arbitrary nonempty schema label as evaluation evidence. It requires `anra-signac-citadel-readiness/v1`, six bound SHA-256 fields, and passing scope/status/firewall/executable-truth/EOS checks before it will report `PRESENT_FOR_REVIEW`. Even a structurally complete receipt is never training authorization; development E0 certificates remain blocked as Citadel readiness.
- Focused verification passed: 19 tests across the Signac preflight and source identity, including arbitrary schema rejection, development-certificate rejection, and the review-only behavior for a complete contract-shaped fixture. Python compilation and `git diff --check` passed. TPU preflight remains blocked on qualified data, Citadel evaluation evidence, and actual Kaggle runtime receipts.
- Latest executable source identity: `4186ed26ea85e44ec4fecb89e6d23f60407fdb95721cd7a0d38e8a9d5ef8530f` over 130 inventoried files.
