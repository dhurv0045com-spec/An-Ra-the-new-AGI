# An-Ra Actual Innovation Plan

Date: 2026-05-05

Status: plan-only commit. No implementation starts until this document is saved in git and explicitly approved for execution.

Reviewed baseline:

- `ANRA_FRONTIER_RESEARCH_PLAN.md`
- `ANRA_ABSOLUTE_FRONTIER_MASTER_PLAN.md`
- `README.md`
- `ARCHITECTURE.md`

## 1. Why This Plan Exists

The two existing frontier plans contain strong research direction: Differential Falsification Cognition, Falsifiable Constraint Cognition, verifier-shaped learning, experimental proof graphs, law-of-physics budgets, and cross-domain constraint mapping.

The missing piece is not another broad doctrine. The missing piece is an implementation discipline that forces every "innovation" claim to become:

```text
repo evidence -> gap -> hypothesis -> constraints -> experiment -> verifier -> measured delta -> memory/update
```

This plan turns the research language into a buildable innovation operating loop for this repository.

## 2. New Core Innovation

Name: **An-Ra Innovation Engine**.

Short name: **AIE**.

Claim:

An-Ra should not merely contain self-improvement, memory, tools, and verification as separate subsystems. It should run them as one closed loop that continuously discovers capability gaps, proposes upgrades, executes bounded experiments, verifies measurable deltas, and stores the result as future training signal.

The existing plans describe how An-Ra should think. AIE defines how An-Ra should upgrade itself.

Core loop:

```text
scan system
-> detect capability gap
-> write falsifiable upgrade hypothesis
-> define constraints and budgets
-> choose smallest experiment
-> run verifier
-> accept, reject, or revise
-> write proof graph
-> generate replay/curriculum item
-> promote only if metrics improve
```

This is the practical innovation: self-improvement becomes experimental science, not vague autonomy.

## 3. Non-Copy Guardrails

To prevent this repo from becoming a pile of copied frontier language, every future innovation item must pass these checks:

1. **Repo anchor**: it must point to current An-Ra files, tests, reports, or missing artifacts.
2. **Falsifier**: it must define what result would prove the idea wrong.
3. **Metric**: it must define at least one measurable delta.
4. **Smallest experiment**: it must start with a narrow local proof before broad claims.
5. **Verifier path**: it must name a deterministic or repeatable check.
6. **Memory effect**: it must say what gets written back as failure replay, skill, or proof graph.
7. **Sovereignty effect**: it must preserve An-Ra identity and owner control rather than copying a generic assistant style.
8. **No frontier name-dropping without evidence**: external model comparisons are not accepted unless source, date, task, and limitation are recorded.

## 4. The First Build Target

The first real implementation should be **AIE Foundation**, not a model rewrite.

Why:

- It is cheaper than training.
- It improves every later training and agentic loop.
- It gives An-Ra a measurable innovation protocol.
- It can be tested without external APIs or large checkpoints.
- It connects existing modules instead of inventing an unrelated layer.

Planned files for the first implementation pass:

```text
innovation/__init__.py
innovation/schema.py
innovation/gap_scanner.py
innovation/hypothesis.py
innovation/verifier_contract.py
innovation/scoreboard.py
scripts/run_innovation_cycle.py
tests/test_innovation_engine.py
```

No implementation of these files should start until this plan is committed.

## 5. AIE Data Objects

### 5.1 Capability Gap

Purpose: represent what is missing or weak.

Required fields:

```json
{
  "gap_id": "stable id",
  "source": "registry|test|report|manual|memory",
  "repo_anchor": "file path or report path",
  "capability": "learning|verification|memory|identity|autonomy|runtime|repair",
  "current_state": "what exists now",
  "desired_state": "what should be true",
  "risk": "low|medium|high",
  "evidence": []
}
```

### 5.2 Innovation Hypothesis

Purpose: turn a gap into a falsifiable upgrade proposal.

Required fields:

```json
{
  "hypothesis_id": "stable id",
  "gap_id": "linked gap",
  "claim": "If we change X, metric Y should improve under verifier Z.",
  "constraints": [],
  "falsifier": "what result rejects the claim",
  "expected_delta": {},
  "owner_control_boundary": "what must remain human-approved"
}
```

### 5.3 Experiment Contract

Purpose: define the smallest safe proof.

Required fields:

```json
{
  "experiment_id": "stable id",
  "hypothesis_id": "linked hypothesis",
  "change_scope": [],
  "commands": [],
  "verifiers": [],
  "rollback_plan": "manual git revert or no-op because plan-only",
  "promotion_rule": "what passes"
}
```

### 5.4 Observation

Purpose: store what happened, including failure.

Required fields:

```json
{
  "observation_id": "stable id",
  "experiment_id": "linked experiment",
  "timestamp": "ISO-8601",
  "passed": false,
  "metrics": {},
  "unexpected": [],
  "next_revision": "accept|reject|revise"
}
```

### 5.5 Capability Delta

Purpose: decide whether the repo actually improved.

Required fields:

```json
{
  "delta_id": "stable id",
  "before": {},
  "after": {},
  "metric_change": {},
  "accepted": false,
  "reason": "why accepted or rejected"
}
```

## 6. Innovation Score

Every proposed upgrade receives a score from 0 to 100.

Formula:

```text
score =
  20 * repo_leverage
+ 20 * verifier_strength
+ 15 * learning_value
+ 15 * novelty_inside_repo
+ 10 * implementation_smallness
+ 10 * safety_and_owner_control
+ 10 * failure_replay_value
```

Definitions:

- `repo_leverage`: improves multiple existing An-Ra layers.
- `verifier_strength`: has deterministic tests, schema checks, benchmarks, or tool outcomes.
- `learning_value`: creates training, replay, or memory signal.
- `novelty_inside_repo`: is not already implemented here.
- `implementation_smallness`: can be built without a risky rewrite.
- `safety_and_owner_control`: preserves approval boundaries.
- `failure_replay_value`: failed attempts become useful data.

Promotion threshold:

- 80+ can be implemented first.
- 60-79 needs a smaller experiment.
- Below 60 remains research only.

## 7. First Ten Innovation Candidates

These are not implementation tasks yet. They are candidate experiments for AIE to score after the foundation exists.

| Rank | Candidate | Gap Filled | First Verifier |
| ---: | --- | --- | --- |
| 1 | Innovation schema and scoreboard | Turns vague upgrades into measurable experiments | `pytest` schema tests |
| 2 | Local repo gap scanner | Finds missing checkpoints, reports, stale tests, unverified docs | registry/status snapshot |
| 3 | Falsification ledger | Stops confident unverified claims | JSON schema and unit tests |
| 4 | Failure-to-curriculum recycler | Converts failed tests and rejected hypotheses into training samples | generated sample validation |
| 5 | Experimental proof graph | Links hypothesis, action, observation, and correction | graph integrity tests |
| 6 | Verifier contract layer | Normalizes SymPy, pytest, schema, and future tool checks | mock verifier tests |
| 7 | Sovereign disagreement evals | Tests non-sycophancy under false premises | identity/eval fixtures |
| 8 | Constraint isomorphism search | Makes cross-domain analogy disciplined | constraint mapping tests |
| 9 | Law-of-physics budget stubs | Blocks impossible robotics/chip/nano claims early | dimensional/budget fixtures |
| 10 | Quantum-chip dry-run demo contract | Forces talk into artifacts and checks | dry-run transcript validation |

The first implementation should only take rank 1 through rank 3 unless the user approves a wider pass.

## 8. Thirty-Day Execution Plan

This plan is intentionally narrower than the existing master plan. It focuses on creating the machinery that can safely generate later innovations.

### Days 1-2: Save and Freeze Plan

Deliverables:

- Commit this file.
- Do not change code.
- Confirm the next implementation scope with the owner.

Acceptance:

- Git commit exists with only the innovation plan.

### Days 3-5: AIE Foundation

Deliverables:

- Add `innovation/schema.py`.
- Add typed data objects for gaps, hypotheses, experiments, observations, and deltas.
- Add JSON serialization and deterministic IDs.
- Add unit tests.

Acceptance:

- `pytest tests/test_innovation_engine.py` passes.
- No network or model dependency.

### Days 6-8: Repo Gap Scanner

Deliverables:

- Scan `runtime/system_registry.py` manifest.
- Scan `scripts/status.py` output shape.
- Detect missing V2 artifacts, missing reports, failing tests, TODO/FIXME markers, and stale claims.
- Emit ranked gaps.

Acceptance:

- Scanner produces stable JSON.
- Tests use fixtures, not live environment assumptions.

### Days 9-11: Hypothesis and Scoreboard

Deliverables:

- Convert gaps into upgrade hypotheses.
- Score each candidate with the AIE formula.
- Save a report in `output/v2/innovation_scoreboard.json`.

Acceptance:

- Every score has an explanation.
- Every high score has a falsifier.

### Days 12-14: Verifier Contract

Deliverables:

- Create a common verifier result object.
- Support local verifiers first: schema, pytest command result, file existence, report metric comparison.
- Keep optional science tools behind availability checks.

Acceptance:

- Missing optional tools do not fail the base system.
- All verifier outcomes include pass/fail, metric, and evidence.

### Days 15-17: Falsification Ledger

Deliverables:

- Add a ledger for claims, assumptions, verifiers, and falsifiers.
- Connect ledger entries to innovation hypotheses.
- Export training-ready failure replay records.

Acceptance:

- A rejected hypothesis produces a useful replay item.

### Days 18-20: Experimental Proof Graph

Deliverables:

- Store gap -> hypothesis -> experiment -> observation -> delta edges.
- Keep storage dependency-light: JSON first, SQLite only if needed.
- Add graph integrity checks.

Acceptance:

- A full innovation cycle can be replayed from disk.

### Days 21-24: Memory and Training Hooks

Deliverables:

- Convert accepted and rejected experiments into training examples.
- Support the DFC/FCC format already described in the existing plans.
- Do not train yet unless separately approved.

Acceptance:

- Generated samples validate against schema.

### Days 25-27: First Safe Demo Contract

Deliverables:

- Define a dry-run quantum-chip demo contract.
- Use placeholder/mock verifiers if Qiskit or Verilog tools are missing.
- Require real verifier outputs before any public claim.

Acceptance:

- Demo can distinguish verified, inferred, assumed, and unknown claims.

### Days 28-30: Review and Promotion

Deliverables:

- Run status, targeted tests, and AIE report.
- Decide what to promote.
- Write an implementation summary with accepted, rejected, and revised hypotheses.

Acceptance:

- No claim of improvement without metric evidence.

## 9. Immediate Implementation Boundary

After this plan is committed, the next approved coding pass should only implement:

```text
innovation/schema.py
innovation/scoreboard.py
tests/test_innovation_engine.py
```

That gives the repo a real innovation skeleton without touching training, model weights, inference, memory, or UI.

Do not start with:

- changing `anra_brain.py`
- adding special tokens
- training a model
- downloading datasets
- adding external dependencies
- modifying identity behavior
- claiming benchmark wins

Those moves come later, after AIE can score and verify them.

## 10. Definition of Done for Real Innovation

An innovation is real only when all of these are true:

1. It is linked to a concrete An-Ra gap.
2. It has a falsifiable hypothesis.
3. It has a smallest experiment.
4. It runs through a verifier.
5. It produces a before/after delta.
6. It writes failure or success into memory/training format.
7. It preserves owner control and safety boundaries.
8. It can be explained from repo evidence, not copied authority.

Anything less is research text, not innovation.

## 11. First Commit Rule

This document must be committed before implementation starts.

Suggested commit message:

```text
Add actual innovation execution plan
```

