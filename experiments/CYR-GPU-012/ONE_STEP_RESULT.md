# ONE-STEP-001 — completed engineering check

Verdict: **NO_ONE_STEP_ACCURACY_GAIN**. Not AGI or demonstrated cognition.

The assistant selected this bounded scope; the user requested progress toward
cognition/AGI, not a specific one-step protocol. The mechanism-probe and 600-update
curriculum plans were drafted but **not executed**. Their proposed experiments
remain unmeasured. This check does not fulfill the larger capability goal.

## Observed execution

- One fresh Adam update, learning rate 1e-5, training batch 64.
- Holdout: 100 canonical pairs excluded from all original data partitions and
  this update's training batch; prior diagnostic-family overlap not excluded.
- Teacher-forced whole-answer exact including EOS: **27/100 -> 27/100**.
- Before and after answer-position argmax token arrays were identical.
- Training loss before update: 15.895657539367676; preclip gradient norm:
  31.98756217956543; changed parameter elements: 986,880.
- Runtime to result: 9.04 seconds; peak allocated CUDA memory: 0.07531 GiB.
- Maximum sampled GPU temperature: 52 C; minimum sampled available RAM: 3.95 GiB.
- Free GPU memory after model release: 4.8828125 GiB. Process exited 0.
- Original CYR-012 FINAL checkpoint hash remained unchanged.

Raw evidence: `C:/Users/ankit/cyr012-evidence/one-step01/` contains FIXTURE,
PREREGISTRATION, BEFORE, AFTER, RESULT and RESOURCES JSONs. RESULT records the
raw-file hashes. Independent rehashing and exact-count recomputation passed.
Focused local tests: 8 passed; git diff --check passed. No CI execution claimed.

## Interpretation and limits

This demonstrates a real gradient/update path, not an accuracy improvement.
A one-step null at this learning rate cannot reject curriculum learning, adequate
training doses, or cognitive capability in general. Teacher-forced evaluation is
not the original free-generation endpoint and does not replace CYR-012 Branch B.
No child checkpoint was saved or promoted. No new diagnosis of the V11 discrepancy.

The script shifts logits one token against next-token labels and uses the shifted
answer/EOS eligibility mask (one_step_run.py:91-96). The recorded finite nonzero
gradient norm and changed parameters provide no evidence of a detached loss.
No additional gradient replay was run; zero accuracy delta alone warrants no
pipeline fix. Hypotheses about probability changes remain unmeasured because
before/after full logits were not retained.

## Next decision, not launch authorization

A meaningful next capability experiment requires a fixed-dose curriculum arm
and an equal-dose original-data control starting from identical parent weights
and fresh optimizer states. Freeze novel pair-disjoint evaluation cases before
training, retain free-generation outputs and a retention measure, and distinguish
pair interpolation from unseen-band extrapolation. Predeclare resources and
success/failure thresholds. The existing single-arm 600-update draft is not that
causal comparison. No such run was launched in this step.

No commit/push or production changes. Cognition/AGI remains unachieved.
