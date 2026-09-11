# ARK-020 V2 — PRE-EXECUTION AUDIT + RED TEAM

**Audit date:** 2026-09-12 · **Verdict: READY FOR OPERATOR COLAB CUDA RUN**
(gated by the launcher's cell-0 scan + cell-1 test/compile gates at run time)

## 1. V1 defect disposition (mission deliverable §3/§4)

| # | suspected defect | live verification | disposition |
|---|---|---|---|
| 1 | V4 integration-boundary mismatch | **CONFIRMED** — `run_ark020.py:760,774` passed `{"A": template}` (nested) and raw factset splits where `V4.acquire_parent`/`select_dose` unpack flat templates (`t['p']`) and semantic rows `(f, q, a)`. Would crash at the first parent call. | repaired: flat + semantic objects; contract tests now execute the real V4 call chain on CPU |
| 2 | skill C not inferable | **CONFIRMED** — V1 sealed keys never trained; prompt contained only `k next` fragments, no mapping. Information-theoretically impossible. | repaired: C replaced by two-hop composition (TASK_VALIDITY_ANALYSIS.md); validity tests prove derivability, balance, no-direct-pair |
| 3 | phase order seeds unused | **CONFIRMED** — `run_ark020.py:792,796` used only `PHASE_ORDER_SEEDS["B"]`; C/D seeds never referenced. | repaired: per-phase seeds bound into arm + checkpoint identity; seed-independence tests |
| 4 | global-vs-phase confirmation | **CONFIRMED** — `run_ark020.py:485` stored global steps; the phase offset distorted the 1.5× ratio (a 3× slowdown read as 1.095×). | repaired: phase-relative confirmation is primary; decide() compares median-vs-median per phase exactly as preregistered; golden tests distinguish median rule from per-run rule |
| 5 | exact-resume overclaim | **CONFIRMED** — V1 smoke hashed model+optimizer and telemetry only; no registry/controller/counters/phase identity; no corruption test. | repaired: smoke hashes 12 state classes independently; corruption test proves fail-closed |
| (extra) | smoke used a broken head_dim hack + vocab mismatch vs `restore()` | found during this audit's own test run | repaired: smoke runs production geometry |

False suspicions: none material — all five suspected defects were real. One additional
defect (smoke geometry) was self-caught during V2 testing.

## 2. V4 evidence integration

`experiments/ARK-019/FINAL_RESULT_AUDIT_V4.md` records the externally audited
`GUARDIAN_CONTINUAL_PROXY_CANDIDATE` result with explicit provenance (operator-returned,
externally audited; not byte-level re-audited in this repository — the ZIP is not
committed here). CURRENT_STATE now carries an audit note correcting the earlier
"believed result unverifiable" status. Older historical verdicts untouched.

## 3. Red team (mission §21) — findings

- **Impossible sealed task**: fixed by C replacement; derivability test-enforced.
- **Template/frequency shortcuts on C**: blocked by distinct template, bijective Y
  balance (test), Trace-answers-never-intermediates property (test).
- **Replay stealing task slots**: replay replaces real-text only; real_slots > 0 fail-closed; task slots arm-invariant.
- **Unequal compute**: logged per arm; plastic has least compute per update (conservative direction).
- **Global-step bias**: eliminated (phase-relative primary).
- **Stale V4 artifacts**: parent/dose reuse only via identity-checked files; dose selection rejected unless `status == PASS`.
- **Duplicate telemetry**: `dedupe_trajectory` at completion; per-session PARTIAL receipts.
- **Checkpoint mismatch**: identity includes arm, parent_sha, dose_b, task_hash, cap16x, and ALL THREE phase seeds.
- **Controller reading SEALED**: structural — controller entry points take CONTROL-state only; test pins.
- **Validation overreach**: validation used only for prospective confirmation gates.
- **Median implementation**: shared `median()` with golden tests covering the median-vs-median rule and its distinction from the per-run rule.
- **CAP calibration mismatch**: per matched set, V4-identical LOW-LR shadow, bound into arm identity.
- **Static baseline unfairness**: static arms get identical task exposure; duty accounted.
- **Cherry-picked seeds**: all seeds frozen in preregistration.
- **Task-order confound**: phase order fixed by design and declared a claim boundary.
- **Science substrate destruction**: joint parent gate + per-set science NLL ≤ 5% gate.
- **Packaging omissions**: zip + .sha256 sidecar + manifest + receipts; result carries decision, dose, hashes.
- **Lock/resume races**: 6h advisory lock; read-only resume scan; one-writer rule stated in launcher.

## 4. Test coverage summary (36/36 PASS locally, CPU)

- **Pure core** (task splits, compose well-formedness, determinism, registry, controller transitions, allocation).
- **Skill-C validity** (derivability, no-direct-pair, frequency balance, both hops in render, order modes real).
- **Phase seeds** (distinctness, binding).
- **Integration contracts** (V4 signature pins; `bmetrics` and `mixed_update` with V2 objects on the real geometry; `run_dose_pilot` boundary end-to-end with patched constants — real shapes, no loose mocks).
- **Checkpoint + fail-closed** (roundtrip incl. registry/controller/confirmations; 4 corruption mutations rejected).
- **Golden decision logic** (13 scenarios: success, quality-only, failure, static-wins, low-interference, formation B/C/D, predictive-wins, median-rule vs per-run distinction, slow-median failure, matching error, incomplete).
- **CPU integration smoke** (task build → V4 boundaries → B/C/D updates → controller+replay → checkpoint roundtrip → fail-closed → packaging).

## 5. Runtime estimate

From the local CPU contracts (real geometry, ~2–4 s/update) and V4's T4 history
(~0.14 s/update primary): expect **0.2–0.4 s/update** on T4 including replay/cap
variance and the larger battery. Campaign: 140,000 main updates + parents + pilots →
**estimated 9–16 GPU-hours ≈ 3–5 sessions of 225 min**. The in-run calibration cell
(`RUNTIME_CALIBRATION.json`) reports the authoritative estimate; it cannot change the
protocol. If operator time forces a reduction, the preregistered fallback is a
prospective 2-matched-set addendum committed before any outcome data exists.

## 6. Residual risks

- C (two-hop composition) formation speed at 12 slots is unproven pre-execution; the
  formation gate converts failure into `INCONCLUSIVE_FORMATION_INSTABILITY_PHASE_C`.
- 3–5 sessions is a real operator commitment; partial bundles after each session.
- V4 evidence remains provenance-marked until a byte-level bundle audit is possible.
