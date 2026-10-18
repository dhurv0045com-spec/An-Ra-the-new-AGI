# Discovery Gym Handoff

## Identity and isolation

- Working branch: `codex/anra-discovery-gym`
- Isolated worktree: `C:\Users\ankit\AppData\Local\Temp\opencode\anra-discovery-gym`
- Base ref: `refs/remotes/origin/Gandiva`
- Base SHA: `49a3717ae9dc81138f55dc2398c7615f8c80b4b1`
- Canonical owner: Gandiva/BRAMASTRA research family
- Active checkout and all other worktrees were left untouched.
- Refined contract/generator/protocol identities are versioned `v2` because intervention safety and provenance validation changed.
- No merge, reset, clean, stash, branch switch, download, model load, GPU/TPU run, or local optimizer update was performed. One scoped commit was pushed to `origin/codex/anra-discovery-gym` at owner request.

## Overlap decision

The existing `bramastra_lab/discovery` component is a direct precursor for Boolean query discovery, and Gandiva/BRAMASTRA already owns the broader investigation, prediction, transfer, and retention research program. It is not replaced or copied. Discovery Gym extends that owner with a causal-program environment, prospective action/prediction commitments, structural split firewall, explicit shortcut audits, and DG stage gates. The exact overlap table is in `docs/OVERLAP_AUDIT.md`.

## Research question and falsifier

**Question:** Can a from-scratch model use a limited, costly sequence of legal experiments to discover hidden causal rules, predict withheld outcomes, and transfer those rules to structurally new worlds better than strong transparent baselines at equal experiment and inference cost?

**Falsifier:** DG0 fails if hidden rule/target/seed/evaluator/future information reaches a policy, if a direct target query is legal, if costs or state transitions are wrong, if a renamed/isomorphic program crosses a split, or if the independent evaluator cannot reproduce outcomes. A later learned-policy claim fails if it cannot beat the transparent information-gain control on untouched worlds at equal cost.

## Deliverables

| Deliverable | Files | Acceptance status |
|---|---|---|
| Charter and research question | `CHARTER.md`, `README.md`, `docs/CHARTER.md` | Complete for DG0 scope |
| Formal world/action contract | `docs/CONTRACT.md`, `contracts.py`, `environment.py` | Implemented and tested |
| Generator and provenance | `generator.py`, `rules.py`, `suite_manifest()` | Deterministic, hashed, split-aware |
| Independent evaluator | `rules.py`, `evaluation.py` | Direct recomputation and tamper test |
| Evidence firewall | `firewall.py` | Public-key/value and commitment-order checks |
| Baseline interface and definitions | `contracts.py`, `baselines.py`, `docs/BASELINES.md` | Random, fixed, uncertainty, information gain, no-experiment, shortcut, and oracle controls |
| Metric contract | `evaluation.py`, `docs/METRICS.md` | Success, prediction/calibration, cost, invalidity, abstention, regret, and explicit unavailable metrics |
| Prospective protocol template | `docs/PROTOCOL_TEMPLATE.json`, `protocol.py` | DG1 template and frozen DG0 protocol |
| Result/checkpoint identity | `receipts.py`, `schemas/*.json`, `docs/RESULT_SCHEMA.md` | Bound hashes; DG0 explicitly has no checkpoint |
| Shortcut/leakage audit | `audits.py`, `docs/AUDITS.md` | Executed PASS on deterministic smoke |
| Stage gates | `docs/STAGE_GATES.md` | DG0–DG5 pass/fail/inconclusive boundaries |
| CLI/workflow | `cli.py`, `__main__.py`, `pyproject.toml` | DG0 and workbench commands work |
| Practical workbench | `workbench.py`, `docs/WORKBENCH.md`, `tests/test_discovery_workbench.py` | CSV/JSON ingest, graph hypotheses, stability, plans, readiness, replay ledger |
| Bounded plan execution | `workbench.py` executor layer, `cli.py` `workbench-execute`, `docs/WORKBENCH.md` | Executor protocol, callback/local executors, cost/failure/useful accounting, ledger-verified reports |
| Owner runbook | `docs/RUNBOOK.md` | CPU/static reproduction and future hardware boundary |
| Resource status | `docs/RESOURCE_STATUS.md` | Host capacity captured; no hardware qualification claimed |
| Handoff | this file | Current scoped continuation package |

## Executed validation

The following were run in the isolated worktree:

- `python -m compileall -q discovery_gym`: passed.
- `python -m pytest tests/test_discovery_gym.py tests/test_discovery_workbench.py -q --tb=short`: 21 passed.
- `python -m discovery_gym dg0 --output C:\Users\ankit\AppData\Local\Temp\opencode\dg0-recovered-final --seed 1701 --worlds-per-split 6 --budget 8`: exit 0.
- DG0 audit: PASS across generator/split integrity, public firewall, legality/cost accounting, independent recomputation, shortcut controls, and confirmation firewall.
- Final smoke receipt SHA-256: `e3ee3ccd6c53f760df4dea6f5ec599651d94e2994f0bfc6f63f8301715f796ba`.
- Protocol SHA-256: `9398ec6a4a26bd51dc680b2b5bd65e4e81db3fb68c7d06c571257abc9432edd8`.
- Suite-manifest SHA-256: `8204c655742b38d24e1340ba66d9241315cf669cb7969bb68f2f1836d0b33fae`.
- Source-closure SHA-256: `941980d2e580ac3f9ea5132734a9afef710838732cb14df0df983ac590545091`.
- Smoke suite: 24 generated worlds, 192 episodes, eight policy identities; public transcripts contain zero `success`/evaluator-label fields.
- Current transparent controls: information gain 16/24, no-experiment 17/24, uncertainty-seeking 17/24, and oracle 24/24. These are instrument diagnostics, not model evidence.
- Existing BRAMASTRA discovery/environment/inquiry focused tests: 33 passed.
- Workbench end-to-end checks: CSV/JSON/JSONL identity, duplicate-group splits, exact graph recovery, bootstrap stability, plan prioritization, readiness, inventory reconciliation, from-scratch mechanism fitting/cross-validation, majority/full-conditional/graph predictive benchmark, CLI fit/benchmark/session/execute/result/replay, executor success/failure/local-simulation accounting, and tamper detection passed.
- Bounded execution check: `workbench-execute` on a 3-column fixture with `--budget 4 --max-recommendations 1` executed 1 recommendation for actual cost 1, skipped the remaining 2, and left a session with 1 ledger event and ledger status PASS.
- Workbench smoke artifact: `C:\Users\ankit\AppData\Local\Temp\opencode\workbench-demo-recovered-plan.json`; audit PASS, graph hypothesis `c -> a`, `c -> b`, and four budgeted recommendations totaling cost `8`.
- Host resource probe: Ryzen 7 170 (8C/16T), about 15.3 GiB RAM, about 180.3 GiB free on C:, RTX 4050 reported 4 GiB; none qualified or used.

The full repository test suite, external hardware qualification, accelerator training, sealed evaluation, and any scientific learning comparison were not run. `make lint`, `make typecheck`, `python -m ruff`, and `python -m mypy` were attempted; `make`, `ruff`, and `mypy` are unavailable in this runtime. No lint or typecheck success is inferred.

## Review answers

- **Does an existing branch already implement part of this project?** Yes. Gandiva/BRAMASTRA implements the Boolean discovery precursor and public environment primitives. Discovery Gym is a scoped extension, not a competing project.
- **Can evaluation distinguish causal discovery from memorization?** DG0 can reject structural duplicates, hidden truth, direct target queries, and shortcut ceilings. It cannot yet prove model learning; DG1–DG3 must do that with frozen prospective protocols.
- **Can simple transparent baselines solve the task?** Not conclusively: the current information-gain control succeeds on 16/24 worlds, while no-experiment and uncertainty-seeking each reach 17/24. The strongest observed control, not a preferred control, is the DG1 comparison target.
- **Do train and test worlds differ structurally?** Yes in the generated suite: structure families are disjoint and hashed. The worlds are not externally authored or sealed.
- **Does every result bind exact identities?** The DG0 receipt binds source closure, generator/evaluator, suite, protocol, policy, model-none, and seed identities.
- **Can the evaluator reproduce outcomes without model labels?** Yes. It re-evaluates accepted events from the hidden rule and target assignment; a forged evaluator-label field is detected in tests, and public submit feedback contains no success label.
- **Are costs fair?** All attempted actions, including rejected/repeated actions, are charged and reported. A later model must add inference and training costs.
- **What would falsify the central hypothesis?** A learned policy that cannot beat the strongest equal-cost transparent control on fresh worlds, or gains explained by shortcuts, memorization, leakage, or unequal budgets.
- **What is the smallest missing external resource for the next stage?** Owner-supplied tabular fixtures and one external experiment executor/owner identity. The workbench can now ingest, analyze, plan, execute a bounded plan through a named executor, account for its cost and failures, and replay those inputs without claiming causality; the only bundled executor is a local mechanism simulator, so real interventions still require an outside owner.
- **Does any output invite a broader claim than the evidence permits?** No. The strongest claim is deterministic DG0 instrument validity.

## Biggest blockers

1. No from-scratch model has been trained; policy learning, belief update, and transfer are unmeasured.
2. The finite rule catalog is transparent and small; the strongest transparent control changes with the action budget, so DG1 must freeze and report the complete control suite rather than rely on one preferred baseline.
3. Confirmation and transfer are fresh generated structure namespaces, not independently authored or sealed external worlds.
4. The local environment lacks `ruff`/`mypy` modules; the full repository verification status must be recorded honestly.
5. The repository's active FINAL-K8 assignment and resource ledger remain separate; Discovery Gym does not consume or alter them.

## Single next action for DG1

Validate the practical workbench on one owner-supplied tabular fixture and one explicitly owned external experiment record using the `docs/ROADMAP_100X.md` scorecard: inspect the data, freeze the dataset identity, generate and challenge the graph hypothesis, execute only the approved highest-priority recommendation through a `CallbackExecutor` bound to the named owner, append its actual result, replay the session, and report useful experiments per interaction, cost to target competence, recovery quality, and transfer. Then freeze a one-hypothesis DG1 learner protocol only after the workbench utility receipt is reviewed.

## Claim ceiling

Current work supports only: **a deterministic, provenance-bound causal discovery instrument plus a practical, replayable discrete-data workbench that produces observational hypotheses and experiment plans.** It does not support a claim that An-Ra or any model learned to discover, predict, transfer, retain, or improve itself, and it does not support an AGI claim.
