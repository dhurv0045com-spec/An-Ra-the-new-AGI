# ESOES — An-Ra Cognition-First Core Research

This branch is a design/research branch for the next An-Ra Core generation. It starts from the current `core-vnext` evidence but does **not** assume that the existing V4 architecture, tokenizer, parameter count, data recipe, or training path should survive unchanged.

## Why this branch exists

The recent evidence is strong enough to justify rethinking the foundation:

- ordinary PGE continuation training produced real weight/optimizer movement and significantly better held-out language-model loss;
- that same training did not create the tested query-conditioned binding, exact context use, or composition capabilities;
- targeted SFT and EXP interventions created or exposed capabilities that generic PGE did not;
- runtime/Connector interventions can repair some failures, but permanently compensating for a weak Core is not the desired end state;
- therefore the next Core should be designed so that cognitive primitives are part of foundation training and are measured throughout training, rather than patched in afterward.

The central idea is:

> **Outer cognition discovers useful computations; future Core training progressively internalizes them.**

## Branch rule

No large training run should begin from this branch until the cognition specification, benchmark, data curriculum, architecture search, optimization contract, checkpoint gates, and falsification criteria have been explicitly reviewed and frozen.

Loss is necessary evidence, not sufficient evidence.

## Files

- `docs/esoes/EVIDENCE_AND_CONTEXT.md` — the evidence base from V4, `core-exp`, and `core-vnext`, including negative results and caveats.
- `docs/esoes/V5_COGNITION_FIRST_BLUEPRINT.md` — the working blueprint for a cognition-first Core/training path.
- `docs/esoes/OPEN_QUESTIONS.md` — unresolved design questions that must be answered by experiments rather than preference.

## Current north-star question

> **What training experiences, architecture, and optimization path cause a small Core to acquire robust binding, contextual state, composition, counterfactual sensitivity, and reliable realization internally — and how do we prove those abilities generalize rather than reflect templates or external repair?**

## Working method

**Build → Measure → Understand → Improve**

For every major design choice:

1. state the cognitive objective;
2. state the competing hypotheses;
3. change as little as possible;
4. run the cheapest discriminating experiment;
5. measure raw internal selection separately from output realization;
6. preserve negative results;
7. scale only after the smaller experiment gives a reason to scale.

This branch is intentionally allowed to ask whether existing An-Ra assumptions are wrong.