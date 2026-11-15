# V8 data-control: learning occurred, generalization is weak, and the evaluator has a confirmed defect

**Audited by Codex, 2026-11-15.** Six owner-run B0/S20 jobs; seeds7,19,41; D0 versus D1; four epochs each. Source `cf043942511f0b14c343445e0c3fe7570bd7c63b`. This folder is named by archive identity, not an invented run ID.

**Decision:** preserve these results as measurements under the original evaluator. Do not promote D1, reject S20's capacity, or attribute the overall failure to one mechanism. Repair the environment transition and reevaluate frozen checkpoints first. **CAUSE OF THE OVERALL CAPABILITY FAILURE: UNRESOLVED.** A specific tool-state implementation defect is established separately below.

## What was verified

- Owner archive SHA256: `f5d28b8759d0252e9c5e487e97cf1a817765def5a0c1523f7e2caeec097e7ef5`; 28,077,063 compressed bytes; 366,288,727 uncompressed bytes;51 members. Every declared member checksum and manifest inventory was checked without extraction or unpickling.
- FINAL_STATUS reports COMPLETE, six expected/completed jobs and no execution errors. This is execution/export status, not a scientific pass. The FULL archive was not supplied or located in Downloads; its integrity, saved tensors and recoverability are not independently certified here.
- Replayed **126,720 recorded episodes**, including initial, epoch1, epoch2 and epoch4 traces. Recomputed semantic answers, action scores, observed fact transitions, tool payloads, public-text hashes and first-divergence records. Checked exact milestone/mode/case coverage and cross-job case identity. Final raw comparison covers3,840 linked records per model, drawn from768 base worlds; variants and seeds are not independent replications.
- All six update logs have complete finite sequences: D0 has2,368 updates/job; D1 has1,472. The24 milestone diagnostic aggregates agree arithmetically with their recorded counts. Neural logits/losses were not regenerated.
- Regenerated D1's3,584 worlds/5,888 teacher states from the immutable source and checked development structural hashes against training. No prohibited overlap was found. Historical test/review failures must not be mistaken for the state of this later source.
- Five focused offline-analysis tests pass. No local research training or neural inference occurred.

## Measured results under the original evaluator

| Seed | D0 raw success | D1 raw success | D1 minus D0 | D0 training-state accuracy | D1 training-state accuracy |
|---|---:|---:|---:|---:|---:|
| 7 |16.35%|7.60%|−8.75pp|74.93%|71.13%|
| 19 |16.46%|10.86%|−5.60pp|88.72%|83.17%|
| 41 |14.95%|7.29%|−7.66pp|88.17%|75.76%|
| Mean |**15.92%**|**8.59%**|**−7.34pp**|83.94%|76.69%|

Training-state accuracy is the recorded teacher-forced first-action metric, not autonomous multi-step performance. The gap is real as measured, but training and development also differ in worlds, rendering and rollout behavior; it is not a pure estimate of one generalization mechanism.

| Final raw cohort | D0 mean | D1 mean |
|---|---:|---:|
| One-hop R09 |10.31%|6.88%|
| Two-hop R10 |10.52%|9.22%|
| Temporal/source E13 |11.04%|6.61%|
| Missing evidence E14 |17.14%|0.63%|
| Conflicting evidence E15 |13.33%|1.56%|
| Null/unavailable A19, including linked variants |33.18%|26.61%|

Both-correct relevant-counterfactual pairs: **D0 22/2,304 (0.95%); D1 5/2,304 (0.22%)**. Neither qualifies robust binding/composition. Grammar-only scores nearly equal raw scores; merely restricting output grammar did not repair the observed failure. Availability masks are themselves affected by the stale availability state and cannot be treated as a clean diagnostic of whether unavailable actions explain failure.

There is a useful positive result hidden by aggregate failure: on initial E14 decisions, D1 matches the training teacher in **1,684/1,920 cases (87.71%)**, versus D0 **389/1,920 (20.26%)**. On E15 the corresponding counts are **1,603/1,920 (83.49%)** versus **177/1,920 (9.22%)**. This is agreement with the teacher's tool-choice policy, including its memory-before-query heuristic; it is not proof of optimal information seeking. D1 learned more of that initial policy while end-to-end completion failed.

The positive acquisition canary achieved53/53 teacher states and30/32 interactive training worlds at LR1e-4. That shows acquisition on this small training cohort is possible; it does not establish broad generalization or rule out evaluator defects.

## Confirmed engineering defect: consumed tools remain advertised

In `anra_v5/elendril/v8/evaluation.py`, `_EpisodeRolloutState.apply_step` records `queried=True` / `retrieved=True` and appends observations, but does not change `query_avail` / `mem_avail` to False. Subsequent prompts and policy masks still advertise the consumed tool. In contrast, `teacher.py:build_teacher_steps_for_world` consumes the availability flag, and `verifier.py:verify_episode_trace` rejects a repeated call.

This establishes a training/evaluation state-contract mismatch directly from source and raw traces. The existing recorded-action rescorer checks stored scores but does not independently enforce the public availability transition, so its PASS did not detect this inconsistency.

| Final raw records across three seeds | D0 | D1 |
|---|---:|---:|
| Records with a stale-availability step |2,051/11,520 (17.80%)|6,754/11,520 (58.63%)|
| Repeated consumed-tool calls recorded |1,094|17,043|
| Steps whose teacher target changes under the proper consumed-tool state |418|6,449|

Across all modes and milestones,30,905/126,720 records contain a stale flag;75,275 steps expose stale availability. These are record/step counts, not unique independent worlds. More tool use exposes D1 more often to the defective transition; therefore the raw D1–D0 difference cannot establish that D1 is intrinsically worse.

Example: D0 seed19, epoch1, world `5738f233deee574e908b482282eec6e3b918b0d91c6c964609f08db4f4180da9`. The model first retrieves memory. The next prompt still says `Query=1 Memory=1`, and the recorded teacher recommends RETRIEVE again. With the consumed memory state, the teacher would proceed to QUERY. This is a concrete erroneous intermediate state, not speculation about representation quality.

**Limit:** fixing the flag has not yet been tested on these frozen neural policies. Recorded traces cannot tell us which new action the model would choose after a corrected prompt. The audit does not assign a corrected success rate or claim this bug explains all failures.

## Failures that precede the defect

One-hop R09 is already answerable at the first decision. Of1,920 D1 initial R09 decisions, only122 exactly match the teacher. D1 proposes ANSWER553 times but431 of those values are wrong; it also requests tools909 times. No earlier consumed-tool flag exists at that point. D0 similarly has only177 exact initial answers, with522 wrong values among699 ANSWER proposals.

Thus substantial first-step evidence-use/action-selection problems remain independently of the tool-consumption bug. The causes may include rendering generalization, entity/value binding, optimization or inadequate supervision. The current evidence does not distinguish them causally.

Training trajectories should be compared on the same cases: on the fixed960-record intermediate subset, D1 seed19 moves9.90%→11.46%→10.63%; seed7 moves7.29%→6.15%→7.40%; seed41 moves6.77%→5.73%→7.40%. Learning losses improve, but held-out performance does not improve monotonically. These are not retention experiments.

## Compute and evidence limits

- D0 processed37,888 supervised states per run; D1 processed23,552. D1 therefore had37.84% fewer updates/states despite both being called four epochs. Corpus, sampler, rendering and exposure changed together. This was a registered recipe comparison, not an isolated architecture or data-only effect.
- Sum of recorded per-job wall intervals: D0 4,859.08s; D1 3,400.85s; total8,259.94s (137.67 worker-minutes). These timers include some evaluation/checkpoint work and exclude final evaluation/export. They are neither end-to-end campaign duration nor billed GPU kernel time.
- Recorded update timestamps span2026-11-14 22:18:21–23:52:59 UTC, or2026-11-15 03:48:21–05:22:59 IST:94.64 minutes. This is an observed interval inside the session, not its complete cost.
- Profiles report two Tesla T4 devices, approximately0.388/0.390 seconds/update and1,461/1,572MiB peak allocated VRAM. Save-at2/resume-to4 parity is reported for weights, optimizer, scheduler, cursor and RNG. Without FULL payloads this is a receipt, not a locally repeated checkpoint restore.
- The archive's cost table pools D0/D1 under B0 and labels accuracy-per-minute as an equal-work view. Those ratios do not constitute matched-compute experimental endpoints. Curated metrics here separate conditions.
- Intermediate checkpoint hashes and traces exist, but the launched source prunes intermediate checkpoint tensors. Those hashes alone cannot recreate intermediate neural inference; keep this limitation when designing follow-ups.
- J, F, pointer P, F-cap, recurrence, retention and transfer did not run. No sealed confirmation was executed. Historical V7.17 scores use different evaluation conditions and are not directly comparable to these percentages.

## Next execution and experiment

**First, repair and qualify the instrument.** Consume QUERY/RETRIEVE availability exactly once; keep runtime, teacher, verifier and trace reconstruction consistent. Include remaining availability and action budget in any shared observation-cache key. Add model-free transition tests covering query, retrieval, empty/irrelevant payloads, repeated calls and sequential/batched equivalence. The audit must reject a trace that advertises a consumed tool. Preserve the current results and source unchanged as historical evidence.

**Next, perform frozen-checkpoint reevaluation, not new training.** Recover and verify the six final checkpoints from `ELENDRIL_V8_FULL.zip`. Run the original and repaired evaluator on identical3,840-case cohorts, seeds and final weights. Register success/failure, repeated-call counts, initial action agreement, post-tool response behavior and paired per-world differences. This isolates the environment intervention. Export complete traces under a new evaluation identity; retain both outcomes even if the repair does not improve accuracy. The current analysis ZIP cannot substitute for FULL weights.

**Then separate remaining causes.** At fixed weights, test the same one-hop world under familiar training rendering and held-out tuple rendering, with relevant/irrelevant fact swaps. If the rendering intervention changes performance, that narrows the bottleneck without retraining. If substantial acquisition/generalization failure remains, preregister a D0/D1 ×1,472/2,368-update comparison with matched seeds, explicit scheduler horizons and separately reported token exposure. Equal updates alone do not equal processed tokens. Only after these controls are interpretable should the typed/factorized/pointer architecture screen proceed.

No claim of AGI, capacity ceiling, or a solved root cause follows from this run. Its useful discoveries are the initial tool-policy learning, severe remaining evidence-use weakness, and a concrete evaluator defect that can now be isolated experimentally.

## Reproduce this audit

From the repository root:

```powershell
python -m tools.analyze_elendril_v8_results --archive 'C:/Users/ankit/Downloads/ELENDRIL_V8_ANALYSIS.zip' --output '<external-output>/METRICS.json'
python -m pytest tests/test_analyze_elendril_v8_results.py -q
```

The analyzer is bound to this exact archive SHA and semantic-source identity. It streams raw records, regenerates procedural D1 data, performs no model inference, and emits only curated metrics. Owner ZIPs and checkpoint tensors remain outside Git.
