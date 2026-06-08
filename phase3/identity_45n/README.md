# 45N — Identity Layer

**Component 08/19 · `identity` (runtime injection)**

Training teaches voice. **45N enforces it at generation time** — inject owner anchors before the model runs, clean outputs after, without silently rewriting the whole stack.

Core identity math (CIV, ESV, HAL) lives in `identity/` at repo root. This folder is the **runtime injector** wired into `generate.py` and the master system.

---

## What it does

```text
load identity file (owner exchanges / anchors)
  → select relevant anchors for prompt
  → prepend / shape context
  → model generates
  → post-generation cleanup (drift phrases, policy)
```

---

## Key file

`identity_injector.py` — `IdentityInjector`, health check, anchor selection.

**Identity data:** resolved via `anra_paths.get_identity_file()` (not hardcoded paths).

---

## Operator

```bash
python anra.py --phase3-status
```

On startup you may see: `[IdentityInjector] Loaded N exchanges...` — that is normal when the identity file exists.

**Missing file:** degrades gracefully; generation continues with reduced anchor pressure.

---

## Relationship to CIV / ESV / HAL

| Layer | Where | Role |
| --- | --- | --- |
| CIV / ESV / HAL | `identity/*.py` | Values, emotion, hormones, drift detection |
| 45N injector | this folder | Runtime prompt shaping |
| CIVGuard | `identity/civ.py` | Hidden-state drift (training checkpoints) |

45N is the **operator-facing** identity surface at inference. CIVGuard is the **training-time** alarm.

---

## Do not

- Hardcode identity paths — use `anra_paths`
- Bypass sovereignty when promoting identity-tuned checkpoints
- Confuse 45N with the full 65% identity **training** bucket (that is `v2_data_mix`)

Deep dive: root [`WALKTHROUGH.md`](../../WALKTHROUGH.md) § Identity.
