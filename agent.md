# Agent brief: everything to know BEFORE the AGI training run

Date: 2026-09-07. Branch: `cymek-500m-readiness` @ `0d16035`.
Engine: 500M campaign state machine. Status: repaired and verified.
Campaign: NOT GO — see blockers. This file is the honest handoff.

## 1. What was built and proven
- One real chain (documents -> manifest -> pack -> sampler order -> layout ->
  exact-token microsteps -> accumulation -> ONE clip/step/state-advance ->
  checkpoint transactions -> resume verification -> receipt). No giant
  batches, no metadata-without-mutation, no silent drift. Any violation
  raises instead of training wrong.
- Proof, all executed (see `experiment.md` for measurements):
  - 33/33 contract tests, throttled CPU, single pass.
  - 27/33 on RTX 4050 (6 failures are VRAM-only, same 6 pass on CPU).
  - Interrupted run == uninterrupted run, byte-for-byte
    (model/optimizer/RNG/cursor/ledger hashes identical).
  - Rotation keeps milestones + head only; timebox stops RESUMABLE and
    resumes to COMPLETE; LR schedule never rewarms; TPU fails closed;
    production without a contamination commitment fails closed.
- Two real bugs were caught by running, not reviewing, and are fixed with
  regression tests (loss-vs-window denominator mismatch; resume fencing
  against the wrong parent).

## 2. Blockers — why the run is NOT GO
1. **DATA IS NOT READY (biggest blocker).** `materialize_first_party`
   honestly reports `DATA_NOT_READY`: first-party supply is orders of
   magnitude below 500M unique tokens. Running now means undertraining on
   repeats — the engine will execute it faithfully and the model will just
   be bad. Do not start the campaign to "see what happens."
2. **No certified execution hardware.** This laptop's 6 GB 4050 cannot fit
   the frozen 32,768-token microstep (~5 GB transient, measured). The Ryzen
   CPU can execute it but at toy speed. The real run needs data-center
   GPUs or TPU with PRE500M certification (collectives, memory fit, bf16
   execution) — all currently `PENDING_PRE500M_TPU`, none of it faked.
3. **Evaluation is not staged.** Sealed/fresh sets, production scoring
   policy, and the query-swap challenger are NOT_AUTHORIZED / not built.
   Training without staged eval means flying blind on contamination and
   capability — unacceptable for an AGI run.
4. **Soak unproven.** Single updates and 2-update campaigns are verified;
   a thousand-session, stop/resume-heavy long campaign has not run (P3).
   Rotation math says it holds; "says" is not evidence.

## 3. Verdict
- **Engine: READY** (for its certified envelope: exact accounting,
  fail-closed resume, honest receipts).
- **Campaign: NO-GO** until data + hardware + staged eval exist.
- Anyone telling you otherwise is selling something. The receipts in this
  repo are the receipts; there are no others.

## 4. Launch checklist (in order, no skipping)
1. [ ] Corpus reaches 500M unique first-party tokens; `materialize_first_party`
       flips to READY with real contamination benchmarks frozen.
2. [ ] PRE500M TPU/GPU certification passes on the actual training hardware
       (collectives, 8-replica topology, memory fit at 32k microsteps, bf16).
3. [ ] P3 soak + P2 bf16-vs-fp32 divergence quantified and accepted.
4. [ ] Sealed/fresh evaluation staged with a production scoring policy.
5. [ ] Dry-run the full 500M campaign config end-to-end (sessions, milestones,
       timeboxes) on a data_manifest subset before burning the real budget.
6. [ ] Only then: fresh `run_500m_session` loop to 500M, watching receipts,
       not vibes.

## 5. Machine rules (this laptop)
- Ryzen 7 170 / ~15 GB RAM / RTX 4050 6 GB. It is a VERIFICATION machine,
  not a training machine. Throttle CPU runs (8 threads, BelowNormal);
  never exceed without asking; small tests on the 4050 are fast and safe.
- Never modify production math to fit test hardware. The 4050 OOM is a
  finding, not a bug.

## 6. Where things live
- Engine: `v5_training/production_entry.py` (`run_campaign`, `run_500m_session`).
- Proof: `tests/test_production_entry.py` (33 tests; `ANRA_TEST_DEVICE=cuda`
  for GPU).
- Measurements + future work: `experiment.md` (P1–P7).
- Full detail: the commit history on this branch and the pushed code.
