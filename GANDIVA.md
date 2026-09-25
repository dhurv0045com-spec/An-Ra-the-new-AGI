# Gandiva: purpose, inheritance, and work in front of us

Gandiva is the branch where the An-Ra ambition meets a testable piece of engineering. Its purpose is to find out whether a small model trained from random initialization can acquire useful cognitive behaviors, carry them to unfamiliar cases, keep them after new learning, and choose better ways to learn. It gives those questions a working model, environments, training paths, controls, and a finite experiment. It does not claim to have produced AGI.

The name marks a line of work, not a separate scientific result. Git branches share files until someone changes them. Gandiva inherited most of its starting documents unchanged, which is why its original README and agent instructions described BRAMASTRA as though nothing had moved. This guide and the branch's updated entry points make the distinction explicit.

## What Gandiva is doing differently

BRAMASTRA supplied the broad research goal and a first K8 experiment design. Gandiva carries that design through the actual model consumers and the run machinery, then asks the owner-run campaign to test the mechanisms against held-out tasks and controls. The changes concentrate in these areas:

- **Learning that reaches the behavior being measured.** The token, action, value, and next-feedback objectives are connected to the real trainer and inference paths. The evaluator records what each head actually did; an unused head or a scripted substitute does not count as learned ability.
- **Cognition grounded in public evidence.** Tasks carry typed goals, received events, legal actions, budgets, and explicit unknowns. The model does not receive hidden answers. The runtime separates observed history from imagined outcomes and keeps the environment as the authority on what happened.
- **Acquisition, transfer, and retention.** K8 compares a from-scratch baseline with a multi-objective treatment, then measures unfamiliar tasks, tools, and protected earlier skills. New task families and counterfactual controls are meant to expose shortcuts, not decorate a benchmark score.
- **Architecture adaptation and recursive method selection.** E4 tests a gated shared-block change. E5 measures candidate learning methods, trains a proposer on those outcomes, then evaluates a successor against frozen and fixed-method controls. The code can execute that chain; only a successful, replicated owner-run result could support an RSI claim.
- **A reproducible two-GPU campaign.** Each T4 runs an independent seed and paired work; memory is not pooled and this is not DDP. E0 checks the live devices, updates, resume path, timing, and allocation before dependent phases begin. Checkpoints, run identity, event receipts, and final archives are part of the experiment rather than afterthoughts.

The active K8 model is a roughly 6.49-million-parameter decoder with random initialization. That size is chosen to make a bounded mechanism experiment possible on the available hardware; it is not offered as an AGI-scale architecture. A separate 100M TPU path has preflight code and design work, but no qualified Kaggle TPU training run. It is not the current K8 launch.

## What Gandiva inherited

The branch starts from `BRAMASTRA` commit `63d60b3523937703183f1e578fe066df08a53acd`.

| Inherited material | How Gandiva uses it |
| --- | --- |
| An-Ra's from-scratch AGI objective and the broad BRAMASTRA research program | They set the destination. They are not evidence that the destination has been reached. |
| The `bramastra_lab` package, decoder, campaign foundation, and original K8 protocol | Gandiva extends their real interfaces and keeps the existing package and notebook filenames for compatibility. `bramastra_k8.ipynb` downloads the `Gandiva` branch by default; the old names do not select the old branch. |
| Older blueprints, branch notes, and experiment proposals | They remain dated source material. A frozen proposal is not automatically the current operating instruction. |
| Prior experiment archives and Kaggle result files | They are evaluated against their own source/data identities. Results from one closure are never silently attributed to another. |

Gandiva is not a merge of the Cymek or Citadel code branches. Their published notes can inform later research, but this branch's claims must come from its own source, protocol, and results. Compare code and history with `git diff BRAMASTRA...Gandiva` and `git log BRAMASTRA..Gandiva`; a shared filename does not mean shared branch work.

## Latest owner result

The 25 September owner campaign is documented in
[the result and protocol audit](engineering/reports/K8_EXPERIMENT_20260925/RESULTS.md).
Both T4s qualified, and the source-bound build passed, but the campaign failed
in E5. E1 measured only eight unique inventory mechanisms repeated four times;
E2 ran 128 groups per seed instead of the frozen 128 per family, and all
learned policy, memory, and workspace episodes exhausted their call budget.
E3 retention was 0/16. E4 passed its gate-gradient and segment-isolation proof
but recorded no task-level improvement. No cognition, RSI, or AGI result has
been established.

The four downloaded archives are views of one run, not replications. The
results-only packages contain no model-weight payloads and cannot resume
training. Do not launch another unchanged campaign. Close the E1 sampling,
E2 coverage/controller, E5 prompt, and E5 accounting gates in the report,
then rebuild and rerun E0 against a fresh source closure.

## Pre-run build state

The following build-readiness note is historical; it preceded the owner run
above and does not authorize an unchanged rerun.

The implementation is ready for a fresh owner-run K8 campaign. The last full local build verification passed all 24 F01–F24 requirements, all seven test groups, and all seven production-interface exercises. It made zero optimizer updates. The report is [`build_verification.json`](engineering/reports/FINAL_K8/gandiva-post-e4-c6c64c89-20260925-pytest-temp/build_verification.json); it records source commit `c6c64c89eb568c44cc1deed683a089580ff29494`, source closure `7f4a24d1b70639f5231cfec283ad0eb4a68feb977512ecdc58459998a70c0457`, and `ready_for_owner_experiment: true`. Later commits recorded verification and clarified documentation; they did not change the campaign implementation. The notebook performs fresh verification for the exact source it acquires.

An earlier owner campaign ran E0–E3 and stopped when both E4 workers hit a CPU-index/CUDA-model mismatch. That defect is fixed. Its measured outcomes were poor: E1 held-out success was 0/32 in each arm; E2 learned policy, memory, and workspace each scored 0/256 with episodes truncating; E3 showed tool-execution receipts but retention was 0/4 per job. E4 and E5 produced no results. These outcomes are recorded in [the result report](engineering/reports/K8_EXPERIMENT_20260923/RESULTS.md) against that run's own source and data identities. They are evidence about that run, not a result for the repaired build.

The verification report establishes local build readiness only. It does not pass Kaggle's live E0 checks, show that the model learns, establish RSI, or establish AGI. No local optimizer updates were made during verification.

## Pre-run launch sequence

This launch sequence was written before the latest owner campaign. Follow the
repair gates in “Latest owner result” before using the notebook again.

1. Start the normal self-sustaining [`notebooks/bramastra_k8.ipynb`](notebooks/bramastra_k8.ipynb) from the pushed `Gandiva` branch on Kaggle with two T4 GPUs. It acquires the source and prepares or validates the required data bundle.
2. Let the notebook make a fresh source-bound build report. E0 then checks both physical devices, actual model updates and resume, measured workload cost, and the live allocation. If a gate fails, the campaign stops and preserves the evidence.
3. If E0 qualifies, run E1–E6 within the registered 480-minute allocation: training stops at minute 450, and the remaining 30 minutes are reserved for final scoring and export. Do not reuse the closed run ID from the failed campaign.
4. Read the exported evidence against the frozen protocol, including negative results and costs. Only then choose a follow-up experiment based on the observed bottleneck. No success is presumed and no AGI claim follows from a passing pipeline.

The operator steps and recovery rules are in [`RUN_EXPERIMENT.md`](engineering/final_delivery/RUN_EXPERIMENT.md). The full comparison design is in [`experiment.md`](experiment.md); F01–F24 are preserved in the [build acceptance contract](engineering/FINAL_EXPERIMENT_EXECUTION.md).

## Reading map

| File | What it answers |
| --- | --- |
| [`README.md`](README.md) | What this repository branch is and where to start |
| [`AGENTS.md`](AGENTS.md) | How an implementation agent should work on Gandiva now |
| [`engineering/STATUS.md`](engineering/STATUS.md) | Latest verified state, past run, and remaining owner gates |
| [`engineering/FINAL_K8_PROGRESS.md`](engineering/FINAL_K8_PROGRESS.md) | Detailed chronological engineering cursor |
| [`experiment.md`](experiment.md) | Frozen K8 hypotheses, schedule, treatments, and analysis rules |
| [`engineering/final_delivery/RUN_EXPERIMENT.md`](engineering/final_delivery/RUN_EXPERIMENT.md) | How to launch, recover, and export the owner run |
| [`engineering/TPU_100M_COGNITION_PROGRESS.md`](engineering/TPU_100M_COGNITION_PROGRESS.md) | Separate 100M TPU preflight; not part of current launch |

Historical documents remain in place for provenance. When an old plan conflicts with this branch guide, current status, or the run guide, follow the current branch documents and retain the old text as dated history.
