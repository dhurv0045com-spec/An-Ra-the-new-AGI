# An-Ra research documents

Branch: `core-vnext`.

## X-FACTOR

**Failure-ablation over forked Core state.** On failure, hold one of `(knowledge, plan, decode, tools)` fixed and change another. The unique flip is the cause. Update exactly one store, or nothing. Do not ask the 180M model why it failed.

Start here: **[X_FACTOR_STATEMENT.md](X_FACTOR_STATEMENT.md)**

These documents override README marketing. This branch is a 180M dense Core plus a Connector experimenter.

| Document | What it is |
|---|---|
| [X_FACTOR_STATEMENT.md](X_FACTOR_STATEMENT.md) | **The X-factor, one page** |
| [X_FACTOR.md](X_FACTOR.md) | Full 12-section answer |
| [X_FACTOR_CODEMAP.md](X_FACTOR_CODEMAP.md) | Code evidence: demonstrated / stub / absent |
| [X_FACTOR_WEEK1.md](X_FACTOR_WEEK1.md) | Planted suite + oracle (loop well-posed; Core not scored) |

Related engineering (not research claims):

- `docs/engineering/AN_RA_CORE_ARCHITECTURE_SPEC.md`
- `docs/engineering/V4_BEHAVIOR_BASELINE.md`
- `docs/engineering/STATE_SEMANTICS.md`

Run the oracle (no checkpoint, no GPU):

```powershell
python -m anra_core.ablation --oracle
python -m pytest tests/test_failure_ablation.py -q
```

Run against a V4 checkpoint when you have one:

```powershell
python -m anra_core.ablation --checkpoint path\to\anra-v4.pt
```
