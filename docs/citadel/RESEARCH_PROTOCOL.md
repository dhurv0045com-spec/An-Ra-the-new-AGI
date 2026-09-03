# RESEARCH_PROTOCOL.md

Citadel's binding operating rules. Source: the CITADEL first-bootstrap instruction, tightened by
what the audit showed actually goes wrong (stale greens, floor-limited substrates, biased scorers).

## 1. Standing question

> What is currently preventing this small An-Ra Core from acquiring stronger transferable internal
> cognition per parameter and per training token — and what is the cheapest controlled experiment
> that distinguishes the leading explanations?

## 2. Labels

Every claim in Citadel documents carries exactly one label:

- **FACT** — verifiable from the repository (command or artifact path given).
- **MEASUREMENT** — a number read from an artifact or produced by a receipted run.
- **INFERENCE** — reasoning from facts/measurements; assumptions stated.
- **HYPOTHESIS** — falsifiable prediction pending test.
- **SPECULATION** — marked as such or not written.

Documentation is never proof. Claims trace to experiments.

## 3. Verdicts

`DEMONSTRATED` (preregistered, powered, replicated — used sparingly) / `SUPPORTED` /
`TENTATIVE` / `INCONCLUSIVE` / `CONTRADICTED` / `IMPLEMENTATION_FAILURE` / `NOT_TESTED`.
A broken or unvalidated implementation yields IMPLEMENTATION_FAILURE, never HYPOTHESIS_REJECTED.

## 4. Receipts are immutable; ledgers are authoritative

If a receipt's verdict field disagrees with the ledger, the ledger wins (see N10/N17: two stale
greens were found inside otherwise-immutable receipts). Receipts are never edited; corrections
are new ledger entries or superseding receipts. Results are emitted programmatically, not typed.

## 5. Experiment rules

- One primary intervention per experiment; everything else fixed (same checkpoint, architecture,
  optimizer, token budget, evaluation) whenever possible.
- Preregistration before result inspection, in the `experiments/<ID>/PLAN.md` format (see C0 for
  the canonical instance): question, existing evidence, primary + competing hypothesis, why,
  independent variable, fixed variables, models/data with hashes, controls, metrics, statistics,
  success/failure thresholds, confound checks, compute budget, stop condition, possible outcomes.
- Baselines where meaningful: current model, trivial heuristic, shortcut, random/untrained,
  ablation, previous implementation — chosen for meaning, not count.
- Implementation validation before any scientific interpretation of failure (§18 of instruction:
  verify the intervention actually occurred — parameters/gradients/tokens/hashes for training;
  batch entry/mixture/packing/leakage/cursor for data; execution/signal/gradients/non-identity
  for architecture).
- Possible outcomes are written before the run, each stating what belief should change.

## 6. Compute ladder

```
static validation → unit test → synthetic probe → tiny CPU run
→ small CUDA run → short controlled training → multi-seed replication → expensive training
```

Every escalation is justified in the experiment log. If a cheap test can kill an idea, kill it
cheaply. Local device reality (receipted): RTX 4050 Laptop 6 GB CUDA + 16-thread CPU; the
tournament's full powered run cost 0.076 GPU-hours — most discrimination happens far below
"expensive training".

## 7. Data discipline

Data is a first-class variable. For every dataset/generator: what capability pressure it creates,
what shortcut exists, how the shortcut is detected, what transfer should occur if the intended
mechanism is learned, and what held-out transformation tests that transfer. Capability per
training token is the metric — never token count. Cymek's mixture→packing→cursor contracts are
reference contracts: experiment with alternatives inside Citadel, never silently redefine
production behavior; better contracts are promoted with evidence, not force.

## 8. Architecture discipline

No module because it "sounds cognitive". Architecture changes require a mechanistic hypothesis
tied to measured evidence and a parameter-matched control. Architecture comes after diagnosis.

## 9. Sealed evaluation discipline

Existing ESOES contamination controls are respected. Development evaluation and promotion
evaluation stay conceptually separate. No tuning against sealed outcomes; any outcome-guided
decision consumes the sealed fixture (ESOES `e0_cognition/sealed.py` policy). Current recorded
state: **no T2 sealed fixture exists; none consumed** — development-tier artifacts only.
If a fixture is ever consumed, that is recorded here immediately.

## 10. Branch discipline

- Citadel descends only from `origin/esoes` (base `85f44b7`). No merges from triquetra/cymek.
- Porting code from another branch requires a provenance record: source branch, source commit
  SHA, source files, reason, whether behavior was modified, and which experiment depends on it.
- Small, meaningful commits (`docs(citadel): …`, `research(citadel): …`, `test(citadel): …`,
  `exp(citadel): …`). Negative results are never rewritten or erased.

## 11. Reproducibility minimum

Every experiment records: branch SHA, model spec hash, checkpoint hash (if any), data/fixture
hashes, config, seed, package/environment, device, token count, command, output artifacts,
evaluation command — machine-readable receipt in addition to Markdown wherever possible.

## 12. Citadel is not

A rewrite, a V6, an architecture dump, a benchmark-gaming branch, a production deployment
branch, or a place where experiments change five variables or failed experiments disappear.
A negative result that destroys a bad idea is useful research; a clever module with no
controlled evidence is not.
