# X-factor week 1 — planted suite and oracle

**Date:** 2026-08-20
**Branch:** `core-vnext`
**Code:** `anra_core/ablation.py`, `tests/test_failure_ablation.py`

## What shipped

A Connector experimenter, not a new neural module.

- Pack format: `<k> … </k> <plan> … </plan> <q> … </q>`
- Battery: baseline + `k_add` / `k_swap` / `k_retrieve` / `plan_change` / `decode_change` / `tool_change` / `truncated_k` / `empty_k`
- Classer: unique flip → class; ties and empty-K-only → `model_limitation`; representation errors → `stop_do_not_learn`
- Update router: one store per class; knowledge writes only for `missing_knowledge` / `wrong_knowledge`
- Decode arm uses `CoreExecutor.fork_state` on a shared prefill
- 80 planted items, 10 per class (8 classes; `representation_failure` is Core-typed, not planted)

## Oracle result (no Core weights)

Command: `python -m anra_core.ablation --oracle`

Tests: `7 passed` in `tests/test_failure_ablation.py`

| Metric | Value | Meaning |
|---|---|---|
| n | 80 | planted items |
| diagnosis accuracy | **1.0** | loop is well-posed |
| false knowledge-write rate | **0.0** | no junk memory on non-knowledge classes |
| always-`model_limitation` | 0.125 | constant-classifier baseline (10/80) |
| per-class accuracy | 1.0 each | no confused arms under planted physics |

This is **not** evidence that V4 can diagnose failures. The oracle scores pack contents, not generated tokens. It proves the experiment can be failed by a real Core.

## What was not run

No V4 checkpoint is in this repo (`*.pt` gitignored). Week 2 is:

```powershell
python -m anra_core.ablation --checkpoint path\to\anra-v4-current-full-resume.pt
```

Expect accuracy near the 0.125 constant baseline if the checkpoint ignores `<k>` tags (see V4 behavior baseline). That result is the SFT gate in `X_FACTOR.md` §5/§10.

## Falsification still open

- Core diagnosis accuracy ≤ 0.40 → Connector attribution is theater; train or stop.
- False knowledge-write ≥ 0.10 → disable memory writes.
- Next-trial Δsuccess ≤ 0 vs do-nothing → labels are not actionable.
