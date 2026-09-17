# ONE-STEP-001 — engineering learning check

Registered before execution. Assistant-selected engineering scope, not a user-specified
protocol and not a test of AGI or the general value of curriculum learning.
The earlier mechanism-probe and 600-update pilot drafts remain UNEXECUTED;
this record supersedes their launch intentions without rewriting those drafts.

Exactly one FP32 Adam update (lr1e-5, default betas .9/.999, eps1e-8,
weight_decay0, clip norm1), fresh optimizer, on 64 new unordered-pair-disjoint
addition examples. Initialization from CYR-012 FINAL model SHA256
94e82920eaf03089630b85b9061bd0fa680b629c0bcd526e58bf301712127a99.
No optimizer resume, no new promoted checkpoint, no baseline pipeline changes.

Fixture: generator v5_experiments/one_step_pilot.py enumerates a<b,
10<=a,b<80, sum<=99, excluding all original data partitions, orders canonical
pairs by SHA256('one-step-8121/a/b'), chooses first64 for training and next100
for holdout. These are development examples, not a sealed benchmark; previously
used diagnostic families may overlap. Equal pairs and reversal duplicates absent.
Holdout is excluded from the training gradient. No rows selected from scores.

Before and after: exactly one forward pass over all100 holdout rows with teacher
forcing. Shift logits one token against targets; compare all answer tokens AND EOS,
ignore prompt/padding. Record raw answer-position token argmaxes and exact flags.
This is named TEACHER_FORCED_WHOLE_ANSWER_WITH_EOS, not a free-generation evaluation.
Gradient: mean answer+EOS CE on training64 only. Record finite loss/norm and actual
parameter change. Full causal mask from packed_layout. No torch.compile/autocast.

Any positive holdout exact delta is descriptive only; no statistical, cognitive,
AGI or production claim. Zero/negative delta ends THIS one-step experiment only:
it does not falsify learning at larger doses. No tuning/retry based on scores.

Hardware: two CPU threads, RTX4050 with CUDA3GiB cap; preflight RAM>=3GiB,
GPU free>=4GiB, temperature<85C; guard RAM>=2GiB,temp<85C before each pass;
wall cap120s in runner and terminal180s. Preserve original model hash.
Hash script/helper/plan/data/parent and loaded frozen sources before computation;
write fixture and provenance before measuring. Outputs outside repository at
C:/Users/ankit/cyr012-evidence/one-step01; refuse existing output path.
Failing fixture/scoring unit tests precede implementation. No full-run smoke is
needed for a single engineering update. Save receipt and raw predictions, NOT weights.
No commit/push. This does not close or replace CYR-012 Branch B.
