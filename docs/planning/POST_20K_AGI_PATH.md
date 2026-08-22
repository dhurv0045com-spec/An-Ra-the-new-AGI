# An-Ra Post-20k Training Path — from foundation to doing

**Branch:** `core-vnext` · **Status:** executable design · **Principle:** every phase
ends at a measured gate, never at a step count. A phase that fails its gate is
not promoted; the gate failure itself is the information.

## Where we are (measured)

| Fact | Evidence |
|---|---|
| Step 0–20,000 trained healthy on the real Phase-A pack | checkpoint `tokens_seen=327,827,071`, manifest-bound `mix_report`, val loss 1.884 |
| Pack pass 1 completed exactly at step 20,000 | 161,133 windows consumed = full `cont_170m_to_500m` pack |
| Post-20k degraded (repeat passes, constant LR) | TPU phase ended at train loss 2.10 > 1.88; generation shows memorization attractors |
| Trainer lacked the documented LR schedule | `PROGRESS.md` specifies warmup+decay; old TPU trainer had neither |

**Recovery anchor: resume from the step-20,000 checkpoint, not step-30,400.**

## The ladder

```
Phase R (recover)      resume step-20k + WSD schedule + verified packs
   │  gate G-R
Phase B (broaden)      500M → ~3B unique tokens, one pass each
   │  gate G-B         ← "Make it Speak"
Phase S (shape)        SFT on audited instruction data, behavior-gated
   │  gate G-S         ← "Make it Behave"
Phase C (cognition)    connector experiments: credit assignment, tools, memory
   │  gate G-C         ← "Make it Learn"
Phase D (doing)        agentic execution with verification loops
```

Each phase has: **entry condition**, **data contract**, **training config**,
**exit gate**, and **rollback**. No phase starts on hope.

---

## Phase R — Recover (days)

**Entry:** step-20,000 checkpoint verified (`anra-v4-current-full-resume.pt`,
step 20000).

**Data:** existing verified pack(s). If only the exhausted 330M pack exists,
Phase R is short by design: WSD decay to completion over ≤1 repeat pass to
settle weights, then stop.

**Config:** `train_xla.py --resume-from <exact step20k path>` (canonical; train_tpu.py is deprecated)
— the scheduler decays LR 2e-4 → 2e-5 across the budget, ending cleanly at the
pack boundary instead of grinding through repeats at full LR.

**Exit gate G-R:** validation loss ≤ 1.85 AND generation produces
grammatical local structure without rare-word attractors dominating
(manual spot check of 10 greedy generations).

**Rollback:** keep step-20k artifact untouched; Phase R output is a new file.

---

## Phase B — Broaden (weeks; the real fix)

**Entry:** G-R passed.

**Data contract:** NEW token packs from the published 11.4B corpus
(`cont_500m_to_3b`, in 330M-token slices like Phase-A used). Each slice:
manifest.json + SHA-256 shards + **zero window overlap with previous packs**
(the replay-budget rule from iterate500: never exceed 4 lifetime passes per
window; target exactly 1).

**Config:** continue WSD per slice: stable through the slice, decay in its
final 10%. LR floor 2e-5 across slice boundaries so knowledge persists.

**Compute reality (v5e-8, bf16):** ≈131k tokens/step ⇒ ~7,630 steps per
billion tokens. 2.5B fresh tokens ≈ 19,000 steps ≈ **15–20 Kaggle sessions**
at 450 min each. This is the price of a speaker; there is no discount.

**Exit gate G-B ("Make it Speak"):** run
`connector/experiments/cognitive_credit/capability_probe.py`:
P1 knowledge-use ≥4/5, P2 plan-follow ≥4/5, P3 verbatim echo ≥4/5,
P4 tool-result use ≥4/5. These four are the *substrate floor* for everything
after. Below them, no downstream phase is interpretable.

---

## Phase S — Shape (SFT, days)

**Entry:** G-B passed. **Never SFT a model below the floor** — iterate500
proved this: their SFT child improved validation loss while failing 100% of
behavior checks, because the substrate couldn't carry instructions.

**Data:** existing audited bundle
(`sft-v4-train.jsonl`, 4,622 examples, license-receipted, source-group-disjoint
validation — already built and frozen).

**Config:** assistant-only answer-masked loss (the iterate500 repair),
LR 1e-5 cosine over 2–3 epochs, checkpoint as separate child artifact with
parent-hash binding.

**Exit gate G-S ("Make it Behave"):** fixed 50-case behavior suite
(arithmetic, factual recall, identity, code, refusal) ≥80% — the same gate
iterate500 defined after their SFT failed it. Plus: cognitive-credit
experiment runs end-to-end on the child without hitting the substrate floor.

---

## Phase C — Cognition (the X-factor, weeks)

**Entry:** G-S passed.

Now the interesting science becomes possible: the cognitive-credit experiment
(`connector/experiments/cognitive_credit/`) runs against a substrate that can
actually execute interventions. For the first time we measure:

- does intervention-based diagnosis beat self-report/heuristic baselines?
- does diagnosis-selected repair actually fix failures?

**Gate G-C:** intervention diagnosis beats outcome-only heuristic by ≥15
points AND repair success ≥50% on diagnosed cases. Either result — success
or honest null — is publishable evidence about An-Ra's architecture.

---

## Phase D — Doing (agentic)

Retrieval → tools → planning → verification loops, in that order, each behind
its own permission/budget/rollback contract (iterate500's ordering, kept).
Out of scope until G-C: self-modification, autonomous experimentation.

---

## Session playbook (each Kaggle TPU session)

1. Attach: step-checkpoint dataset + current token-pack dataset.
2. Run `core_vnext_tpu_training.ipynb` top-to-bottom.
3. Cell 3 verifies pack hashes fail-closed — a mismatch stops everything.
4. Train up to 450 min; checkpoint atomically replaces one output file.
5. Before ending: run the probe cell; record metrics in the session log.
6. Upload checkpoint to Drive vault; keep two-generation retention.

## Non-negotiables (learned from iterate500's errors)

1. Fail closed on data provenance. The notebook must refuse an unverified
   dataset rather than fall back to whatever file matches a name.
2. One canonical checkpoint path per lineage, atomic replace, SHA recorded.
3. Scheduler state resumes exactly; LR multiplier logged every 10 steps.
4. Loss numbers alone promote nothing; only gates do.
5. Never SFT below the capability floor; never pretrain past an exhausted
   pack at full LR.
