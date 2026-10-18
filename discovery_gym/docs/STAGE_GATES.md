# Stage Gates

The stages are conditional. A later stage cannot be justified by code volume, a passing test suite, a compelling explanation, or an earlier stage's development result.

## DG0 — Instrument validity

**Question:** Are the environment, evaluator, provenance, split firewall, legality checks, and baseline interface valid?

**Pass:** all critical audit checks pass; costs and commitments are exact; the evaluator recomputes outcomes; public and evaluator artifacts are separate; shortcut controls are measured; the protocol and suite identities are recorded.

**Fail:** any hidden information reaches a policy, any illegal action is applied, any cost is unaccounted, any duplicate/isomorphic program crosses a split, or the evaluator cannot reproduce a result.

**Inconclusive:** a required artifact is missing, a source identity is unbound, or a result cannot be recomputed. Missing evidence is not a pass.

**Current status:** PASS for the executed local deterministic smoke. This is an instrument result only.

## DG1 — Learnability

Only after DG0 passes may a small, randomly initialized model consume training-world experience. Compare it with no-learning, random, fixed, uncertainty, no-experiment, and transparent information-gain controls on fresh development worlds, and treat the strongest observed control as the predeclared comparator. Freeze the model initialization, encoding, optimizer, step budget, data manifest, and development metrics before opening confirmation data.

**Pass:** the learner improves on a fresh development signal over no-learning controls, with checkpoints and exact cost accounting.

**Fail:** the learner cannot fit the signal, relies on a shortcut, or gains disappear after structural controls.

**Inconclusive:** underpowered development data, missing checkpoints, invalid comparisons, or a task that transparent policies already solve.

## DG2 — Prospective experiment choice

Freeze the DG1 protocol, model identity, policy identity, confirmation worlds, metrics, and practical effect threshold. Commit each policy action and prediction before exposing its consequence. Compare the learned policy with the strongest relevant transparent baseline at equal action and inference cost.

**Pass:** positive paired benefit over the strongest relevant baseline on untouched confirmation worlds, with uncertainty clustered by independent world and a predeclared margin.

**Fail:** no benefit, a shortcut explanation, invalid episodes, or a benefit caused by unequal budgets.

**Inconclusive:** wide uncertainty, protocol drift, or insufficient independent worlds.

## DG3 — Structural transfer

Evaluate each declared axis separately: rule combination, graph shape, variable role, intervention affordance, and rendering. Renaming or reseeding is not a transfer axis. A pass on one axis does not imply broad transfer.

## DG4 — Sequential discovery and retention

Only after DG2 and DG3 pass. Acquire multiple skills in sequence, then test fresh retention, new-skill formation, interference, and equal-cost replay references. A controller that refuses new skills cannot receive retention credit.

## DG5 — Internalized discovery

Future preregistered study. Remove external guidance only after assisted performance is characterized. Report assisted and unassisted results separately. Scripted generation or orchestration is not internalized learning.

## Default thresholds

DG0 mechanical thresholds are exact: evaluator agreement `1.0`, firewall pass, legal/cost audit pass, and structural split disjointness. DG1–DG5 scientific margins are deliberately not invented here; they must be proposed with rationale, frozen before confirmation, and labeled prospective.
