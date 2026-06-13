# Contributing to AN-RA

AN-RA grows by strengthening canonical owners, not by collecting alternate implementations.

## Setup

```powershell
pip install -e ".[dev]"
python -m scripts.verify_structure
python -m pytest tests -q
```

Optional ML and memory dependencies may be installed through the project extras or `make install-ml` where supported.

## Before Editing

For every behavioral change, identify:

1. Canonical owner.
2. Public interface and schemas.
3. Caller and persisted state.
4. Metric and evidence artifact.
5. Typed failure and recovery behavior.
6. Rollback and migration path.
7. Tests.

If the current abstraction is weak, improve it at its source and migrate callers.

## Architecture Rules

- Use `anra/anra_paths.py` for repository and artifact paths.
- Use package imports; never mutate `sys.path`.
- Do not use wildcard imports.
- Check feature flags at the real call site.
- Do not silently downgrade, start fresh, or claim an unavailable backend.
- Do not infer one metric from an unrelated report.
- Keep checkpoints immutable after promotion.
- Keep robotics in simulation/shadow mode unless a separate physical promotion exists.
- Register T-01 through T-26 integrations in `runtime/technology_registry.py`.
- Log meaningful architecture and contract changes.

## Adding Capability

1. Extend the canonical owner.
2. Add or update typed contracts.
3. Persist evidence atomically.
4. Add focused tests and integration reachability.
5. Run the smallest relevant suite.
6. Run the full suite before delivery.
7. Update operator/developer documentation and the engineering log.

## Verification

```powershell
python -m pytest tests -q
python -m training.train_unified --mode status --model-size 3b
git diff --check
```

A blocker list from the status command is valid evidence. Do not manufacture local artifacts to make it green.

## Research Vision Lock

Every contribution must preserve:

- owner-shaped data as the center of gravity;
- CIV, ESV, HAL, SSG, and promotion boundaries;
- verifier-backed checkable reasoning;
- memory, replay, and falsification as compounding mechanisms;
- measurable improvement before auto-application;
- explicit distinction between implemented, measured, and promoted.

## Commit Messages

```text
type(scope): short description
```

Types: `feat`, `fix`, `test`, `docs`, `refactor`, `perf`, `chore`.

Examples:

```text
feat(training): persist tokenizer migration provenance
fix(promotion): verify manifests without creating a signing key
test(memory): add deterministic fusion benchmark coverage
```
