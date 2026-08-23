# PURPOSE — what this repository actually is

Read this before touching anything. It is written from measured evidence,
not aspiration. Every claim below has a receipt in `output/`.

---

## The one-sentence version

An-Ra is an attempt to build a small intelligence that **notices its own
failures, diagnoses them with controlled experiments, makes the smallest
corrective change, and proves on sealed tests that it gained the capability
without losing the ones it already had.**

Everything else in this repo exists to serve that sentence or to keep it
honest.

## The ultimate goal, stated precisely

Not "a strong 180M language model." Not an agent framework. The goal is:

> **Causal capability accumulation**: a closed loop in which the system can
> repeatedly acquire cognitive primitives (context binding → selective
> binding → composition → tool reasoning → …), where each acquisition
> (a) was justified by verified evidence about why a failure happened,
> (b) was produced by the smallest intervention that evidence supported,
> (c) transferred to tasks and formats the training never saw, and
> (d) did not silently destroy previously acquired capabilities.

The end state worth building toward: a system that can say, with receipts —

> "I failed in pattern X. Competing explanations were H1–H4. Intervention Y
> flipped the failure under controlled conditions. I proposed and trained the
> minimal corrective change. The child improves on sealed unseen tasks and
> retains everything I could do. The causal evidence predicts when this
> repair will work again."

That is cognitive credit assignment becoming self-improvement. Nothing
shorter is the goal.

## What An-Ra is explicitly NOT

- Not a chatbot, product, or demo. Chat quality is a diagnostic, never a goal.
- Not a pile of impressively named modules. (504 such files were deleted in
  one commit; the lesson is institutionalized here.)
- Not benchmark optimization. A score that improved because the test leaked
  into training is a defect, not a win.
- Not autonomous weight mutation. The system may propose, simulate, train
  isolated children, and recommend promotion. It may never silently rewrite
  the active checkpoint or alter sealed evaluations.
- Not architecture worship. Core stays boring (dense 180M decoder) until
  measurement — not fashion — says otherwise.

## The loop everything must serve

```
task
  ↓ attempt (real Core execution)
  ↓ verifier: FAIL                      ← success is decided ONLY here
  ↓ controlled single-variable interventions
  ↓ evidence: which change flips the outcome
  ↓ missing-capability hypothesis (with competitors)
  ↓ TrainingProposal (smallest justified intervention)
  ↓ isolated child checkpoint
  ↓ development evaluation (replay bank; selection happens ONLY here)
  ↓ sealed OOD evaluation (frozen before training, never imported by it)
  ↓ capability delta: gained / retained / regressed
  ↓ promote (scoped) / reject / SPECIALIST
  ↓ preserve evidence + lineage
```

## The measured reality (as of 2026-08-22)

Facts, with evidence:

- **Substrate**: V4, 180,093,312 params. Raw next-token training left it
  unable to use supplied information at all (nonce knowledge 0/5 at steps
  5k/20k/30.4k, both protocols — `output/probe_v2_*.json`).
- **Targeted SFT works**: ~940 corrective examples + 9 GPU-minutes gave
  single-fact context binding that transfers across five untrained
  protocols (4/15 → 13/15 — `output/ood_child_sft.json`).
- **Narrow training steals capability**: a prose-only follow-up gained its
  target and destroyed protocol transfer (13/15 → 6/15). Gradients were NOT
  in conflict (cos = +0.17) — the cause was starvation + format narrowing;
  the damage concentrated in the embedding/head (2× block drift)
  (`output/grad_conflict.json`).
- **Balanced replay preserves capability, replicably — but robust selective
  binding has NOT been demonstrated**: the accumulation child is ≥ its anchor
  on every axis of two independent sealed suites (retention accumulation:
  supported). Selective binding itself did not survive independent
  replication (paired-CF 0/26 on both models on OOD-4) — the substrate
  currently learns selection *formats*, not a format-independent selection
  *operation*. Keep the two claims strictly separate: retention transfer is
  real; target-capability transfer is not yet. This negative result is
  preserved, not hidden.
- **Training past step 20k was harmful** (capability peaked at 20k,
  collapsed by 30.4k; diversity 1.00 → 0.57). Step-20k lineage is the base.
- Three evaluator/trainer bugs were caught only by behavioral or
  counterfactual checks, never by loss: the label off-by-one (loss → 0.0003
  while emitting whitespace), the CF-scored-against-wrong-gold defect, and
  the stateless GPU decode that fabricated "the the the" degeneracy.
  **Conclusion institutionalized: training loss is not behavior.**

## Non-negotiable rules

1. The verifier is the only source of success. Completers return raw text.
2. Hidden labels never touch diagnosis or curriculum generation
   (`ObservedFailure` vs evaluator ground truth, structurally separated).
3. Sealed OOD suites are frozen (SHA-pinned) before training, never imported
   by training code, evaluated once; the moment one influences a curriculum
   it is downgraded to DEVELOPMENT_ONLY and a successor is frozen.
4. Counterfactual pairs are byte-exact single-value replacements, asserted
   at freeze and at run. Paired scoring is the anti-self-deception metric.
5. Evidence taxonomy: `BehavioralImprovementObservation` (parent→child
   diffs) can generate hypotheses; only `VerifiedInterventionExperience`
   (controlled single-variable runtime flips) supports causal claims.
6. Checkpoint selection is multi-objective: target gain AND parent-relative
   retention floors (each protected capability may drop ≤ 0.10 from its own
   baseline). A child that trades one capability for another is a
   SPECIALIST, never a successor.
7. Promotion is always scoped (EXPERIMENTAL / CONTEXT_BINDING /
   CONNECTOR_READY / ACTIVE). "PROMOTE" without a scope is invalid.
8. Every result carries a receipt: commit, dirty flag, checkpoint +
   parameter SHA (one canonical implementation), suite SHA, decode policy,
   seed, device, raw outputs. No receipt, no evidence.
9. Minimal-intervention ladder: no change → runtime/context → memory →
   policy → small targeted training → large training. Escalate only on
   evidence. **Learning when NOT to train is a first-class capability.**
10. Keep training small (~50 optimizer updates produced the accumulation
    result). Never bet big before cheap discriminating experiments.

## Where things live

- `anra_core/` — the boring, tested Core (executor, state, strict loader).
- `connector/runtime.py` (+ `anra.run`) — the one reference loop.
- `connector/experiments/cognitive_credit/` — intervention battery, probes.
- `connector/experiments/ood{,2,3,4}_battery/` — sealed lineage of suites.
- `connector/experiments/capability_bank.py` — replay/dev data (structural
  split, audited) — allowed for training and selection.
- `connector/experience.py` — evidence contracts, proposals, lineage.
- `training/sft_context_binding.py`, `training/sft_accumulate.py` — the
  corrective trainers (causal eligibility, parent-relative floors).
- `output/EVIDENCE_MANIFEST.json` — every artifact labeled
  VALID / SUPERSEDED / INVALID / INTERMEDIATE / DEVELOPMENT_ONLY.
- Branches: `core-vnext` = stable reference; `core-exp` = this research.

## The current open question (exactly one dominant)

Query-conditioned value selection is **learnable but not yet behaviorally
dominant**. The clean replication (tp-grouped-queryswap-replication-002)
proved grouped counterfactual query-swap training installs a large,
cross-vocabulary query-conditioned preference (paired group-level delta
+2.486 nats on a frozen fresh fixture; rank-1 34%→53%; corrected greedy
30%→37%; RESULT C-). Decomposition shows two remaining regimes: 56/75
failures are selection misses, 19/75 are realization losses where the
correct value wins the likelihood ranking but loses the token-by-token
argmax commit. A runtime constrained-decode intervention flipped 20/20 of
its applicable failures with zero regressions — proof the preference is
real and query-keyed. Next: preregister ONE regime-targeted objective
(candidate margin vs decode-commitment pressure) using the v2 extraction
fixture as protection gate.

## Standard of success

> Capability acquisition is useful. Capability accumulation is powerful.
> **Causal capability accumulation is the research program.**

A result counts only if: the diagnosis justified the intervention, the
intervention added the predicted capability, existing capabilities survived,
and the improvement transferred somewhere training never saw. An-Ra must
stay harder to fool than the people (and agents) building it.
