# AN-RA POST-500M TRAINING FORENSIC MODEL AUDIT

**Date:** 2026-08-24 · **Auditor:** independent forensic pass · **Evidence commit:** `7b957a1`
**Machine:** CPU-only, float32, greedy decoding · **Raw artifacts:** `output/forensic_audit/`

---

## 1. Executive Verdict

**LIVE CORE-VNEXT HEAD:** `259fe3b` (audit evidence committed on top as `7b957a1`)

**BEST CHECKPOINT:** There is only ONE model. The artifact named
`anra-v4-tpu-latest to 500m token.pt` is **bitwise identical in weights and
optimizer moments to the step-20k parent**.

**VERDICT CLASS: F — INSUFFICIENT EVIDENCE** (no trained child exists to evaluate)

**CONFIDENCE: HIGH**

One paragraph: The file claiming "~500M tokens" of new training contains the
step-20k parent's exact weights (203/203 shared tensors bitwise equal) and its
exact optimizer moments (exp_avg bitwise equal, Adam step frozen at 20,000).
Only *bookkeeping metadata* differs: global_step relabeled to 22,517,
pack_step 2,517/2,517, a final-loss metric of 2.0479, and a fully decayed LR
envelope (param-group lr = 2e-05). A 39-prompt frozen behavioral battery under
identical greedy decoding produced **39/39 identical outputs** between parent
and "new". Whatever happened during this campaign, **no weight update from it
reached this file.** This is not a model report; it is an evidence report that
the training result never landed.

---

## 2. Training Lineage

| Role | Artifact | File SHA256 | Weights vs parent |
|---|---|---|---|
| A. Parent | `anra-v4-current-full-resume.pt` (schema v9, global_step 20000, source_commit `0107980…`) | `8b39323a…` | — |
| B. Historical degraded | `anra-v4-tpu-latest.pt` (schema v1, step 30400, loss 2.105) | `ccaffdfb…` | distinct (older failed campaign) |
| C/D/E. "New 500M" | `anra-v4-tpu-latest to 500m token.pt` (schema v3, step 22517) | `8ac1bc7a…` | **bitwise identical to parent** |

Lineage graph:

```
step-20k parent ──?──> [claimed ~500M-token campaign] ──?──> "new500m" artifact
        └──────────────── bitwise identity across this arrow ─────────────┘
```

No intermediate candidates exist on disk; no candidate directory was produced.
The claimed lineage has no observable interior.

## 3. Token Accounting

From the artifact's own `trainer_state` (anra-training-state/v2):

- tokens_per_optimizer_step: **131,072** (bs 1 × accum 8 × world 8 × seq 2048)
- pack_step: **2,517 / 2,517** (pack marked complete)
- dataset_windows: 161,133 (= the same Phase-A pack the parent consumed:
  `cont_170m_to_500m_seed1301`, manifest SHA `87cf95be…`)

- **NEW_TRAINING_TOKENS: 0 provable.** If pack_step 2,517 had really executed,
  arithmetic says 2,517 × 131,072 ≈ **330M tokens** were consumed — but the
  weights prove zero updates landed, so those tokens (if any compute occurred)
  produced no persisted parameter change.
- **CUMULATIVE LINEAGE TOKENS: ~330M unique** (one full pass of the Phase-A
  pack at step 20,000). The "500M token" label remains unproven; it matches the
  pack's design name (`170m_to_500m`), i.e., the *target* of the original
  campaign, not measured consumption.
- **REPEATED TOKENS:** unknowable from this artifact; cursor says epoch 0,
  batch_in_epoch 20,136 — internally inconsistent with pack_step 2,517 unless
  rescaled, which itself suggests the state was written by a different
  bookkeeping path than any real single continuous run.

## 4. Training Health

No training log was provided or found for this campaign. From checkpoint
metadata alone: final metric loss 2.0479, final LR envelope decayed to min
(`lr_schedule: wsd_pack_v1, total_steps 2517, warmup 0, min_lr_ratio 0.1`;
param_groups lr = 2e-05 = exactly base 2e-4 × 0.1). Schedule arithmetic is
self-consistent. **Health rating: n/a — no run to grade.** The absence of logs
alongside a completed-cursor checkpoint is itself a finding.

## 5. Validation Loss

NO CLEAN VALIDATION LOSS COMPARISON AVAILABLE — moot, since the compared
artifacts are the same model.

## 6–8. Behavioral Battery & Distribution Health

Frozen 39-prompt battery (continuation / conversation / factual / arithmetic /
copy / context-binding / selective-binding / query-swap / composition /
instruction / uncertainty / degeneration / distractor), identical greedy
decoding, stateful executor:

- **Identical greedy outputs: 39/39**
- factual 0/6 both · arithmetic 0/6 both · copy 0/4 both · code emission 0/8 both
- mean repetition 0.412 both · distinct-1 0.238 both · mean length 24.0 both

Both models remain below the language floor established by prior audits:
HTML/XML tag salad after `<answer>`, no echo ability, no nonce binding, no
query sensitivity.

## 9. Query/Context Analysis · 10. Long Context · 12. Weight-Delta

Weight-delta forensics is categorical: **zero tensors changed**, so layer-wise
drift, embedding/head drift, attention-vs-MLP drift are all exactly zero by
identity, not by smallness. Logit KL between the two artifacts is identically 0
(same function). Representation and attention analyses would compare a model
with itself; omitted as meaningless.

## 14. Optimizer / Resume Integrity

This is the sharpest finding: the artifact carries optimizer state whose
`exp_avg` is **bitwise equal to the parent's** and whose Adam `step` is frozen
at 20,000. If 2,517 real updates had occurred and been saved, step would read
22,517 and moments would have moved. Combined with identical weights, the
conclusion is forced: **this file was assembled around the parent's untouched
state** (or saved from a worker whose updates never reached the host state that
was serialized).

## 15. Old 30.4K Degradation Check

Not applicable — no new model exists to degrade. Both artifacts reproduce the
parent's known failure modes exactly (tag loops, France/France loop, `<p>` runs).

## 16–18. Capabilities

NEW OR STRONGER: none measurable. REGRESSIONS: none (nothing changed).
FUNDAMENTAL FAILURES unchanged: context imitation, nonce binding, query
conditioning, instruction adherence, tool-result use — the substrate still
cannot perform the primitives the Connector experiments require.

## 19–20. Best Checkpoint / Parent Recommendation

BEST: **PARENT (step-20k)** — trivially, since it IS the candidate.
Should it be replaced? **NO** — there is nothing to replace it with. Do not
relabel this artifact as a trained child; that would poison future lineage
comparisons with a false positive.

## 21. Recommended Next Experiment

QUESTION: Did the campaign actually run, and if so where did its updates go?
WHY: The receipt claims a completed pack; the weights claim zero updates.
Exactly one of those is false, and knowing which determines everything.
SMALLEST EXPERIMENT: On the Kaggle account that ran it — re-run the notebook
with `max_steps=25`, then immediately compare `latest.pt` weights to parent
(bitwise check + one logit probe). If they move, the save path works standalone
and the original run's artifact was overwritten/mis-exported. If they don't
move, the trainer→host serialization path is broken and must be fixed before
any long campaign.
EXPECTED: weights differ after 25 steps.
FALSIFICATION: bitwise equality after ≥25 logged optimizer steps proves the
checkpoint pipeline drops updates; do not resume any campaign until fixed.

## 22. Scientific Bottom Line

After the "~500M-token campaign," An-Ra is **still exactly the step-20,000
model** — byte for byte. The campaign's product is a bookkeeping wrapper, not a
trained network. No capability was gained because no computation changed. The
honest next step is not another audit but a 25-step reproducibility run on the
machine that produced this artifact.

---

### Evidence receipts

- Battery script + frozen prompts: `scripts/forensic_battery.py`
  (prompts_sha256 recorded inside output JSON)
- Weight forensics: `scripts/forensic_weights.py`
- Raw results: `output/forensic_audit/battery_results.json`,
  `output/forensic_audit/weight_delta.json`
- Checkpoint file SHAs: parent `8b39323a…` · new `8ac1bc7a…`
- Decoding: greedy, rep_penalty 1.0, no_repeat_ngram 0, max_new_tokens 24, CPU fp32
