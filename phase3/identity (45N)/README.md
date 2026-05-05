# 45N - Identity Layer

**Layer 08/19: `identity`**

45N is the Phase 3 identity injector. Together with `identity/civ.py`, `identity/esv.py`, and `identity/civ_watcher.py`, it keeps An-Ra from drifting into a generic assistant voice as more capability is added.

## Current Role

```text
prompt
  -> compact identity context
  -> generation runtime
  -> response cleanup
  -> drift-sensitive memory/eval hooks
```

Identity is not decorative style. It is a structural guardrail: the system should learn from stronger examples without surrendering authorship.

## Main File

| File | Role |
| --- | --- |
| `identity_injector.py` | Runtime identity context and response cleanup |
| `README.md` | This operator/developer map |

Related mainline files:

- `identity/civ.py`
- `identity/esv.py`
- `identity/civ_watcher.py`
- `training/finetune_anra.py`

## Runtime Behavior

If the explicit identity file is missing, the injector falls back instead of crashing. That is intentional: identity support should degrade visibly, not break the whole runtime.

You may see:

```text
[IdentityInjector] WARNING: identity file not found. Using fallback.
```

That warning means source is present, but the richer identity artifact should be restored or regenerated.

## Verification

Use:

```bash
python scripts/status.py
python anra.py --phase3-status
python -m training.train_unified --mode eval
```

Identity improvements should be judged by:

- fewer generic refusals or canned self-descriptions
- stable An-Ra voice under pressure
- better coding and reasoning without teacher-style capture
- fewer contradictions about project purpose and owner priorities

## Boundary

Do not solve identity by stuffing giant prompts into every call. Keep identity compact, testable, and compatible with memory, evaluation, and training.
