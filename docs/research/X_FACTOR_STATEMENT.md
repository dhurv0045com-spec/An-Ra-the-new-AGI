# X-FACTOR: Cognitive Credit Assignment Through Controlled Interventions

> **X-FACTOR:** Connector-owned cognitive credit assignment. On failure, hold
> everything fixed, change one variable (knowledge, plan, decode policy, tool
> adapter, context packing), rerun Core, and diagnose from which change
> flipped the verifier. The diagnostician never sees the planted cause.

**Status: methodology validated; substrate currently too weak (see below).**

## What replaced what

The original prototype (`anra_core/ablation.py`, removed) contained a fatal
methodological flaw: `arms_for()` branched on `item.planted_class`, so the
intervention generator used the hidden answer while constructing the
experiment that supposedly discovered it. Its 100% oracle score was label
leakage, not evidence.

The replacement lives in `connector/experiments/cognitive_credit/` and
enforces separation structurally:

- `ObservedCase` — everything the diagnostician may see (task, corpus, plan
  candidates, tools, initial attempt). No field derives from the cause.
- `HiddenGroundTruth` — evaluator-only (planted family, gold solution,
  gold knowledge, gold plan).
- `build_interventions(case: ObservedCase)` — the type system admits no
  hidden data. A focused test
  (`tests/test_cognitive_credit.py::test_hidden_label_flip_cannot_change_interventions`)
  proves permuting the hidden label cannot change the generated battery.
- Interventions are realistic: knowledge arms retrieve from a provided corpus
  (and may fail); plan arms come from the system's own candidate list; decode
  arm changes real sampling policy; tool arm toggles a real adapter; context
  arms repack the same information.
- Diagnosis preserves uncertainty: `multiple_plausible`, `unresolved`, and
  evidence-gated `model_limitation` (only when the full battery ran and
  nothing flipped).

## Files

| File | Role |
|---|---|
| `case.py` | ObservedCase / HiddenGroundTruth / Attempt types |
| `suite.py` | 20 cases, 4 families, fault injection separated from surface |
| `interventions.py` | one-variable-at-a-time battery from observed data only |
| `diagnose.py` | flip-pattern classifier with first-class uncertainty |
| `runner.py` | battery execution, baselines A/B, repair success, metrics |
| `run_real.py` | real-checkpoint entry point |
| `capability_probe.py` | substrate preconditions (P1–P5) |
| `tests/test_cognitive_credit.py` | no-leakage proof + oracle validation |

## Results on the trained V4 (step 30400)

All five substrate preconditions fail (0/5 each): the model cannot use
in-context facts, follow stated plans, echo words, or report tool results —
even when the gold content is placed directly in its context. Intervention
diagnosis is therefore not yet runnable on this checkpoint; the experiment
is validated end-to-end with oracle physics (20/20) and is ready to rerun
once a checkpoint passes the capability probe.
