# An-Ra research documents

Branch: `core-vnext`.

These documents are the X-factor investigation. They override README marketing. Implementation on this branch is a 180M dense Core plus a Connector experimenter.

| Document | What it is |
|---|---|
| [X_FACTOR.md](X_FACTOR.md) | Full answer: diagnosis, mechanism, experiment, what to stop |
| [X_FACTOR_CODEMAP.md](X_FACTOR_CODEMAP.md) | Code evidence: demonstrated / stub / absent |
| [X_FACTOR_WEEK1.md](X_FACTOR_WEEK1.md) | Week-1 result: planted suite + oracle (loop is well-posed; Core not yet scored) |

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
