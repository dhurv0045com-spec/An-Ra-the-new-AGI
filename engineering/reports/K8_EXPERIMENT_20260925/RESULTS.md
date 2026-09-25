# K8 owner campaign: result and protocol audit

**Disposition: the campaign did not complete.** E0 qualified both T4 workers; E1, E3, and E4 reached their planned update targets. The E1 and E2 evaluations departed materially from the frozen coverage plan, E2 learned-policy episodes exhausted their call budget without submitting answers, E3 retention was zero, and both E5 workers were stopped by a 571-token confirmation prompt against a 512-token context. E6's artifact checks passed, but that does not make the campaign successful. This run provides no evidence of AGI or recursive self-improvement.

## Run identity and integrity

The four supplied archives are different exports of one campaign, not four replications:

| Export | Bytes | SHA-256 |
| --- | ---: | --- |
| k8-d383144c62de-results-a0800c27d780-failed-full-campaign-819dba65.zip | 253,067 | 2c40c703c8774b672b33d7d413cca813411016db03fc3216d15d6a29067af7a2 |
| k8-d383144c62de-results-a0800c27d780-results-1b4ab37e.zip | 365,187 | b8865afc802b4bef3b20499bc1366fd43fe510151f6a981a86679075ac7086de |
| k8-d383144c62de-results-a0800c27d780-safety-results-d23fb708.zip | 365,757 | 2a069b6fc27c8f95685a847fc58009e471b15a7cdef7789dc1bf26586953b434 |
| k8-d383144c62de-results-a0800c27d780-xprobe-2c0b3f97.zip | 365,182 | e2be12600e9e4e200475f1bef5f1da223bc1dff2835f39ed43a07c54821b59d2 |

All four ZIP central directories and member CRCs verified successfully. The run ID is k8-d383144c62de; the shared instance is a0800c27d780. Their common source closure is f6d3cef20b34fe5481b4c2cbd4fd5001143115c504693d6eac7571e1707343d7, data hash is f71a3a45f5aca99279eaa27b0cdc227b8f3d2f7630821f01473e5471cba0f3fc, configuration identity is 5d9cf2155c9f84e034313cf5a605bf806a66bea8babcb23b446622cc1b5c4ade, and tokenizer identity is 62423dd27ba60f027cd0ca8e01e036cb08ed008005da3f0d49edd0a783290012. The source-bound build verifier records clean source commit 9b806b68271b0da4aa13cc10f4b51d76818a73c6 and ready_for_owner_experiment: true; that commit is an ancestor of the current Gandiva branch. The model had 6,493,952 parameters, width 256, eight layers, and a 512-token context.

The frozen allocation was 480 wall minutes, with training stopping at minute 450 and 30 minutes reserved for export. E0 independently qualified cuda:0 and cuda:1 on Tesla T4s and recorded 12/12 committed updates across two receipts. This confirms that both accelerators participated; it does not show that they sustained any particular utilization percentage.

## Outcomes

| Phase | Observed result | What the result supports |
| --- | --- | --- |
| E0 | Two distinct T4 devices qualified; 12/12 updates committed. | The live two-worker admission path worked. |
| E1 | Four jobs (A/B × seeds 1701/1702), 4,000 commits each. Each job recorded 0/32 successes and 32 EOS stops. Each 32-row sample contains only inventory tasks: eight distinct mechanism IDs repeated four times apiece. | The model stopped generation, but did not answer these sampled tasks correctly. The reported 32 rows are not 32 independent cases and do not cover the registered task families. |
| E2 | The frozen selection says 128 confirmation clusters per family. Each seed's output contains 128 total groups—43 rule-inquiry, 43 inventory, 42 program—rather than 384. | E2 is a protocol-coverage deviation and must not be described as the planned 128-per-family evaluation. |
| E3 | Four jobs completed 4,000 commits each. Each reports two verified tool outputs with receipts. Retention is 0/4 in each job, 0/16 overall. | Tool-call execution was recorded; retention of the tested earlier skills was not. |
| E4 | Four jobs completed 4,000 commits each. Gate gradients and segment isolation passed. The enabled arm learned nonzero gates. | The gated architecture path trained and passed its mechanism proof; no task-level benefit was measured. |
| E5 | Both workers failed during independent P1 successor-choice capture: the prompt was 571 tokens for a 512-token context, so the decoder correctly refused to truncate or copy another policy's choice. | The final method-selection/confirmation comparison did not qualify. RSI was not tested to completion. |
| E6 | The five xprobes passed, with zero optimizer updates. Their phase list includes E0–E4 and E6, but omits E5. | Export, allocation, checkpoint-lineage, accounting, tokenizer, and random-init geometry checks passed where reported. This is not a pass for the campaign or E5. |

E1's selection defect is visible both in the outputs and in the executor: it walks development-measurement JSONL files in sorted path order and stops as soon as it has 32 rows, without family balancing or mechanism deduplication. The next evaluation needs a deterministic, paired sample with explicit per-family coverage and unique mechanism counts recorded in its receipt.

E2's mode scores below are per seed, with 128 cases in the actual output. The symbolic comparator is a separate reference, not a model result.

| E2 mode | Seed 1701 | Seed 1702 | Truncated episodes |
| --- | ---: | ---: | ---: |
| a-direct | 33/128 | 38/128 | 0/128 each |
| a-fixed | 33/128 | 38/128 | 0/128 each |
| a-random | 33/128 | 38/128 | 0/128 each |
| b-planner | 33/128 | 38/128 | 0/128 each |
| b-policy | 0/128 | 0/128 | 128/128 each |
| b-workspace | 0/128 | 0/128 | 128/128 each |
| b-memory | 0/128 | 0/128 | 128/128 each |
| symbolic reference | 103/128 | 105/128 | 25/128; 23/128 |

E2 made no optimizer updates. The three learned modes (b-policy, b-workspace, and b-memory) all used the full 16-call budget in every episode and never submitted a successful answer. The 128 goal-swap probes per seed also all reached 16 calls without an answer; all 128 contradiction probes per seed were skipped because history was empty, and no complementary-coverage group was fully covered. These are concrete controller/runtime failures to address before interpreting the low scores as a clean estimate of model capability. The planner calibration joined zero rows, so its tie with direct answering is not evidence that planning improved performance.

E4's S1 gates after training were [0.01414, 0.05316] for seed 1701 and [0.00736, 0.05289] for seed 1702; S0 remained [0, 0]. Forward cost rose from 0.01238 to 0.01430 seconds for seed 1701 (about 15.5%) and from 0.01276 to 0.01364 seconds for seed 1702 (about 6.9%). The recorded E4 output contains no held-out task score or transfer comparison, so these costs and gate values establish operation, not improvement.

E5 failed because the canonical proposer payload includes the identities and families for a six-task confirmation block. The production proposer refuses to truncate an overlength prompt, preserving independent model-origin choices. That refusal is correct; the input contract needs a compact, training-matched representation and a regression asserting that the real confirmation payload plus its generation reserve fits the model context. Task and checkpoint identities should remain bound in receipts even if opaque task IDs are removed from the model-visible choice payload.

There is also an E5 accounting distinction to resolve. The top-level phase aggregation reports zero attempted and committed updates for E5, while its checkpoint milestones include P0/P1/P_fixed snapshots at update index 12 and M0/M1/M2 trial snapshots from 811 through 1,022. The failed reservations report 6,675.58 aggregate device-seconds. Those intermediate snapshots are not a qualified completed comparison, but the report should reconcile them with the phase counters before calling E5 “zero work.”

## What can be restored from these files

The ZIPs are results-only exports. Their restore evidence reports zero payload files in the archives, restorable_from_archive: false, and 23 checkpoint payload paths intentionally omitted. The xprobe ran inside the Kaggle working directory and found 12 payloads there; that live-session observation does not put those weight bytes into the downloaded ZIPs. The supplied files preserve logs, evaluations, source/data identities, ledger records, and checkpoint metadata. They cannot restore or resume the learned model by themselves.

## Engineering disposition before another campaign

Do not rerun the unchanged notebook and call the result a replication. The next source-bound launch should first close these gates:

1. Repair E1 selection: sample across all three families, deduplicate mechanism identities, pair the exact same held-out IDs across A/B and seeds, and record both row count and unique-case count.
2. Enforce the E2 frozen target: 128 confirmation clusters per family per seed (384 groups), with a preflight assertion that refuses to continue if family counts or mode episode counts are short.
3. Fix the E2 controller budget contract so learned modes can stop inquiry when evidence is sufficient and reserve a legal call for the final answer. Keep “truncated/no answer” separate from a completed but incorrect answer.
4. Bound the E5 method-choice prompt under the actual tokenizer and context, including the generation reserve. Train and decode from the same compact public payload; preserve task identities, proposer checkpoint identity, and choice origin in the audit record. Keep refusal behavior for genuinely invalid choices.
5. Reconcile E5 trial snapshots, reservation events, committed updates, and device seconds in the phase report. A failed phase must retain useful intermediate evidence without presenting it as a qualified confirmation.
6. Give the export an explicit weights-inclusive option if later resumption is required. The results-only ZIP must continue to say clearly that it is not restorable.

Keep this run immutable as a negative, partially qualified result. Any fixes change the source closure; rebuild and validate that exact source, use a fresh run ID, and rerun E0 before dependent phases. A passing execution pipeline would still not establish AGI or RSI; capability, transfer, retention, and improved learning efficiency must be measured on valid held-out tasks against controls.
