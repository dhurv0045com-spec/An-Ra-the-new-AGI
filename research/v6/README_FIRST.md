# ELENDRIL V6 — Research handoff bundle

**Status:** Design package, not trained results and not a GPU-ready notebook.

1. `ELENDRIL_V6_GRAND_EXPERIMENT_PROTOCOL.md` — prospective experimental design, hypotheses, resource budget and pass/fail/next-step rules.
2. `ELENDRIL_V6_EXPERIMENT_REGISTRY.json` — machine-readable immutable draft registry for 15 panels, arm allocation, model profiles, data split and result schema.
3. `ELENDRIL_V6_KAGGLE_T4X2_BLUEPRINT.ipynb` — schema-valid Kaggle notebook blueprint; default cells tested to run with NO network, NO GPU, NO writes and NO campaign. It deliberately refuses training until a new V6 source commit is pinned and code and real GPU qualification pass.
4. `ELENDRIL_CODEX_V6_GRAND_EXPERIMENT_MANDATE.md` — full implementation order to Codex. This is what to send with the three attachments.

The current reference Elendril SHA is `0cd200730d43547294cdc9bf8540a478130cce9c`. It predates V6; **Codex must not repin to this exact SHA as a live V6 runner.**

The V6 science is centered on learned action selection, evidence revision, OOD robustness, diagnosis of failures and conditional memory/continual learning. It is a scientific portfolio: 15 panels, not a promise that 15 large neural models can train fully in 6 hours.

The notebook must be recreated after Codex implements and tests V6, then qualified for native Kaggle GPU T4x2, per-device memory, true new-VM durable checkpoint recovery, and owner authorization. The final GPU run will create a separate `ELENDRIL_V6_RESULTS.zip`; that results ZIP **does not exist yet**.
