# ELENDRIL V8-X1 Implementation Handoff

## 1. Status Summary
- **Execution State:** `CPU_QUALIFIED_GPU_PENDING` (no S20 19.87M parameter GPU run executed locally).
- **Planthir / Palantiri Integration:** `x1_s20` registered as a first-class `PlanthirBackend` (`doctor` `GREEN_SIGNAL`, `init -> validate -> run -> verify-bundle -> replay-bundle` verified on CPU).
- **H01 Status:** `HISTORICAL_WEIGHTS_NOT_AVAILABLE` (legacy V8 D0/D1 checkpoints are not mounted locally; original archive contract audits are preserved).

## 2. Implemented Architecture & Protocol Modules
- `anra_v5/elendril/v8/evaluation.py`, `anra_v5/planthir/observatory.py`, `tests/test_elendril_v8x_instrument.py`: V8 `obs_cache` identity repair (`PublicInput` full-key identity) + independent trace transition verification + legacy preservation.
- `anra_v5/elendril/v8x/`:
  - `contracts.py`: V5 S20 backbone (`19,871,616` params at context `640`), `X1ModelSpec`, `<=1%` capacity matching (`R-noedge`, `P-capR`, `F-cap`).
  - `data.py`: `D1-H` (`x1-d1h-v1`, 12 families `X01`–`X12`), `D2` (`x1-d2-pairs-v1`, 4 linked views), 4 renderers + heldout, `<= 640` token bound across all tool states, non-anticipating teacher, `0` prefix collisions, `0` structural signature overlap.
  - `protocol.py`: exact 42 core S20 jobs (`A0, A1, A2, A3, D0, D1, D2`) + 12 optional `U` extension jobs matching `research/v8-x/X1_DESIGN_REGISTRY.json` 1-for-1.
  - `models.py`: S20-backed `X1TypedModel` wrapping `v5_model/core.py` (`B0, J, F, F-cap, P, P-capR, R, R-noedge, U-1/2/4, U2-untied`) with deterministic component-level shared initialization.
  - `objectives.py`: `standard`, `PAIR-CE`, `PAIR-EQ`, `PAIR-EQ-CF` with exact global denominator accumulation.
  - `training.py`: `X1S20Trainer` transactional checkpointing (`64`-update full-resume rotation, deduplicated `u0000` shared backbone + `0/592/1184/1472/2368` inference milestones).
  - `evaluation.py` & `analysis.py`: `X1S20EnvironmentKernel`, `X1S20ObservablePolicy`, batched GPU rollout, lesion/intervention battery, depth scaling (`1..5`), `U` test-time sweep (`1/2/4/8`), and 10k base-world cluster bootstrap.
- `anra_v5/planthir/adapters/x1_s20_adapter.py`: `X1S20Backend` wired into `backend.py`, `checkpoint_intelligence.py`, `causal.py`, `design.py`, `agent_tools.py`, and `__main__.py`.
- `tools/elendril_v8x_{worker,campaign,export}.py`, `tools/analyze_elendril_v8x_results.py`, `tools/build_elendril_v8x_notebook.py`: isolated 2-GPU scheduling via Planthir `IndependentWorkerScheduler`, 330-min cap + 45-min reserve, 2 GiB disk reserve, ZIP security verification, offline intake, and deterministic notebook generation.

## 3. CLI & Notebook Launch Workflow (Kaggle 2xT4)
- **Pinned Runtime Commit (`SOURCE_SHA`):** `8e62ca0f2046c0f07910fdd20abee40508631227`
- **Generated Notebook SHA-256 (`notebooks/ELENDRIL_V8_X1_KAGGLE_T4X2.ipynb`):** `8c59dab4bf4c3f021f0c029816b20f5ff1b379e0345332c3ddd39c3907bd82ca`

```bash
python -m tools.elendril_v8x_campaign prepare --source 8e62ca0f2046c0f07910fdd20abee40508631227 --session A0 --work-dir <DIR>
python -m tools.elendril_v8x_campaign validate --source 8e62ca0f2046c0f07910fdd20abee40508631227 --session A0 --work-dir <DIR> --cpu-fixtures
python -m tools.elendril_v8x_campaign admit --source 8e62ca0f2046c0f07910fdd20abee40508631227 --session A0 --work-dir <DIR> --devices 0,1 --max-minutes 480
python -m tools.elendril_v8x_campaign run --source 8e62ca0f2046c0f07910fdd20abee40508631227 --session A0 --work-dir <DIR> --devices 0,1 --max-minutes 480 --output-dir <OUT> --owner-authorized
```
- Notebook: `notebooks/ELENDRIL_V8_X1_KAGGLE_T4X2.ipynb` (owner-armed `OWNER_AUTHORIZED = True`; automatically loops `SESSIONS = ["A0", "A1", "A2", "A3", "D0", "D1", "D2"]` on 2xT4 GPU within available wall-clock and disk budget with zero edits).
- Byte-identical notebook rebuild verified from `git archive 8e62ca0f2046c0f07910fdd20abee40508631227`:
  ```bash
  python -m tools.build_elendril_v8x_notebook --source 8e62ca0f2046c0f07910fdd20abee40508631227 --output notebooks/ELENDRIL_V8_X1_KAGGLE_T4X2.ipynb
  python -m tools.build_elendril_v8x_notebook --source 8e62ca0f2046c0f07910fdd20abee40508631227 --output notebooks/ELENDRIL_V8_X1_KAGGLE_T4X2.ipynb --check
  ```
