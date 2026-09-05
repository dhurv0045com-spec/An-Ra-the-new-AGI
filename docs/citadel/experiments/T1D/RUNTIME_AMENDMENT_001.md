# T1D RUNTIME_AMENDMENT_001 — audited runtime repin (implementation only)

Status: **PREREGISTERED — NO RESULTS**. Amends no scientific content in
T1D/PLAN.md (arms, gates, budgets, data, thresholds unchanged). No T1D results
exist anywhere. This records a runtime-identity change with its evidence.

```text
OLD_CYMEK_SHA:
298c91ac04f756f0833a7edcf63e73af3d5af688 (T0-certified pin)

NEW_CYMEK_SHA:
28bf57a0d299a2c13a99fe0046616c00a1b8530c (= origin/cymek HEAD, verified)

CHANGED FILES BETWEEN THEM (26 total):
.gitignore, anra_v5/cli.py (whitespace + 1 subcommand entry), pyproject.toml
(packaging includes), artifacts/cymek/* (receipts), e0_cognition/binding_* +
shortcut_suite + tests (E0 eval harnesses), v5_data/{accounting,acquire,
contamination,data_status,near_dedup,normalize,p35a_slice,pipeline,quality,
readers,registry} (NEW data-pipeline/registry files), v5_evaluation/
generator_qualification.py, v5_tokenizer/adapter.py (+encode_batch, additive)

T1D-CRITICAL CHANGED FILES:
none. anra_v5/miniature_run.py, v5_model/*, v5_contracts/model_spec.py,
v5_objectives/causal_lm.py, v5_training/{optimizer,checkpoint,state,
distributed}, v5_data/{pack,manifest,mixture,cursor}, v5_tokenizer identity
semantics: all byte-identical (empty diff on every listed path).

SEMANTIC VERDICT:
identical for every T1D + PRE50M code path. The delta is additive scaffolding
(data foundry/registry, E0 harnesses, packaging) plus receipt refreshes.

WHY SCIENTIFIC PREREGISTRATION REMAINS VALID:
no model, objective, optimizer, checkpoint, packing, cursor, or tokenizer
behavior touched by T1D changed at all; all arm contracts reference behavior,
not SHAs, and every receipt records the exact runtime SHA it ran against.
```

T1D/PRE50M now pin `28bf57a`. The exact-SHA bootstrap (§2 fix) makes future
HEAD movement a non-event: pinned experiments resolve their SHA regardless.
