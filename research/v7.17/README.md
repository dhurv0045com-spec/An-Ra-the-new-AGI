# V7.17: locate the S20 failure

Status: optimization phase executed by the owner; [audited results](../../Experiment%20Result/V7.17/README.md) show a consistent lower-LR improvement and continued novel-chain failures. Representation training remains pending. The protocol below preserves the original registered design.
**CAUSE UNRESOLVED.** This continues the existing learner and preserves every
V6/V7/V7.1/V7.15/V7.16 result and checkpoint reproduction path.

## What the audit actually established

Seeds 7, 19 and 41 are selected reproducible random-number-generator settings,
not predefined tasks. The previous code used each seed for both fresh model
initialization and training order. Training worlds instead use fixed seed 606,
validation 707, confirmation 909. Existing results cannot separate initialization
from ordering sensitivity. `blocked` changes only the first epoch's ordering.

Two code reviews found no demonstrated defect in the inspected causal mask,
next-token shift, target-only loss mask, microbatch denominator, tied-parameter
ownership, optimizer stepping or training/inference prefix. This is bounded
source review, not proof that the complete system has no bug.

Actual training-data inspection found 9,472 unique visible prompts, zero
contradictory prompt/target duplicates. The deterministic row-order intervention
preserves 9,472 examples, all targets and every sequence length. Original data
identity: `7786244187963f75e6cda5b10e4d4e0d81927b7000154126fff538dea15539e5`;
intervention: `afcf7f7b44502492124d87d7b82ea288c748d421b12a0925cae190f17e485b7b`.
This audit does not establish that the task or labels are scientifically adequate.

## The implemented experiment

The machine-readable preregistration is
[`v717_protocol.py`](../../anra_v5/elendril/v717_protocol.py). All research runs
use fresh **19,871,616-parameter S20** models, four epochs, the same dataset,
batch 16/microbatch 2, FP32, cosine schedule, clipping 1. No S3 research runs.

| Separate owner session | Registered jobs | Sole changed controls |
|---|---:|---|
| `optimization` (default) | 8 | Baseline 3e-4 LR × seeds 7/19/41; 1e-4 LR × same seeds; crossed init/order 7/19 and 19/7 |
| `representation` | 6 | Same three baselines; reverse evidence rows in a deterministic half of training worlds × same seeds |

The existing diagonal baselines plus crossed runs form an exploratory 2×2
initialization/order comparison for 7 and 19. Seed 41 supplies a third paired
LR/representation replicate; it is not crossed. Repeated baselines between
sessions are deterministic repeats, never additional independent seeds.

Before updates, and at epochs 1, 2 and 4, save full training and validation
teacher-forced loss/accuracy, separate first-action/value/EOS confusion counts,
and greedy exact-action decoding for the first 16 sorted worlds per kind,
including every supervised state in those worlds. These decoded measurements
use teacher observation states; independent raw interactive rollouts remain
separate. The teacher/decode gap uses identical bounded cohorts.

An initial FP32 gradient check compares accumulated microbatch gradients with
per-example references, without updating weights, gradients or optimizer state.
Both GPUs also perform production-shaped updates and exact fresh-process
checkpoint continuation. These verify engineering mechanics, **not positive
skill acquisition**. An S20 exact-subset memorization control is not included;
failed train fitting may still require that smallest next experiment.

Raw interactive evaluations retain initial/epoch1/epoch2 base and hop-3
conditions, all stress conditions at epoch4, and a separate final grammar
decoder. Confirmation is final only. No endpoints are dropped to fit time.

## Reading the result

`analysis/S20_DIAGNOSIS.md` and `.json` provide trajectories and paired changes.
Nested `analysis/jobs/<job>/measurements/` contains denominators, predictions,
token confusions and initial per-parameter gradient evidence. The original
epoch reports contain complete interactive episode traces; update JSONL files
contain losses, LR, gradient norms, sampled indices and charged work.

| Observed pattern | What it localizes; what it does not prove |
|---|---|
| Poor full-training acquisition | Keep optimization, exposure, representation and infrastructure hypotheses open |
| Good training, poor held-out behavior | Generalization gap; not proof of inadequate model capacity |
| Teacher-forced success, decoded failure on the same examples | Inference/serialization gap; inspect first wrong token and EOS |
| Teacher-state decoded success, interactive failure | Observation/policy rollout gap; inspect first divergent action |
| Lower LR consistently helps all three paired seeds | Evidence of optimization sensitivity under this recipe; not a universal repair |
| Row-order intervention improves reordered worlds | Evidence of formatting sensitivity; scrutinize other conditions for regressions |

No change or inconsistent changes falsify the simple intervention hypothesis.
Neither phase tests larger datasets, much longer training, natural-language
pretraining, external tasks or AGI. Shared procedural authorship and a public
development confirmation set limit claims. Twenty million parameters are an
experimental floor here, not the project's intended final scale.

## Owner launch and recovery

Import [`ELENDRIL_V717_KAGGLE_T4X2.ipynb`](../../notebooks/ELENDRIL_V717_KAGGLE_T4X2.ipynb)
into Kaggle. Enable Internet and GPU T4×2; Run All launches automatically.
No input attachment or package replacement is required. The notebook fetches
an immutable source commit and generates audited data in writable working.

Both GPU lanes must pass runtime, update/recovery, gradient-parity and measured
diagnostic timing checks. The whole phase must fit within 330 minutes with
2× timing margin and export reserve. Storage admission includes retained raw
generations and a staged full ZIP. NO_GO preserves receipts rather than
silently reducing the experiment. Passing CPU tests is not GPU qualification.

Checkpoint state is saved every 64 completed updates and at epochs 1/2/4;
milestones survive pruning. Same-output restart resumes the exact lineage.
For a fresh VM, set RESTORE_ARCHIVE to the same-source, same-phase **FULL** ZIP
and use a fresh destination. Archive identity and all bytes are checked before
extraction, then runtime/capacity qualification repeats. Fresh-VM numerical
equivalence is not yet demonstrated; insufficient disk can still yield NO_GO.

Download `ELENDRIL_V71_ANALYSIS.zip`, `ELENDRIL_V71_FULL.zip` and
`FINAL_STATUS.json` from exports, and save Kaggle output before ending the VM.
The inherited ZIP names are intentional. Analysis ZIPs update after each job;
the complete full ZIP is built and verified at finalization. Abrupt VM loss
before a saved output can still lose session-local checkpoints.

COMPLETE requires every job, all four diagnostic milestones bound to the same
evaluation checkpoint/update, all required epoch/grammar reports, recorded-action
replay, checkpoint components, and both verified current-run ZIPs. Root-cause
status remains unresolved regardless of engineering delivery status.

Official environment references checked 2026-11-13:
[Kaggle runtime image](https://github.com/Kaggle/docker-python),
[PyTorch 2.11 reproducibility](https://docs.pytorch.org/docs/2.11/notes/randomness.html).
Runtime versions are measured rather than assumed. PyTorch does not promise
identical behavior across releases/platforms; recovery qualification is explicit.
