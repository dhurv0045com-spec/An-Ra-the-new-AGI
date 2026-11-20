# ELENDRIL V8-X2 Implementation Handoff

## 1. Qualification & Release Status
- **Current Stage:** `CPU_QUALIFIED_READY_FOR_OWNER_CANARY` (no S20 19.87M GPU training or confirmation executed locally).
- **4-Axis Status:** `execution=PENDING`, `delivery=PENDING`, `scientific=UNRUN`, `causal=UNRESOLVED`.
- **§0 Independent Replay Blocker:** FIXED and verified via fresh-subprocess public CLI regression (`init -> validate -> run -> verify-bundle -> replay-bundle`) on stock `x1_s20`, `elendril_x2`, deliberate input tamper, wrong backend, unregistered plugin backend (`MISSING_BACKEND_PLUGIN`), legacy non-V8 missing reference (`INSUFFICIENT_REPLAY_EVIDENCE`), and legacy V8 compatibility.
- **Target Hardware Gates Still Pending Owner Execution:** live Kaggle 2xT4 canary + `replay-bundle`, dual-lane physical device/peak-VRAM profiling, full S20 waves, and locked confirmation scoring.

## 2. Implemented Architecture & Palantiri Integration
- **Palantiri Replay Blocker & Runtime Fixes:** `anra_v5/planthir/{cartridge,checkpoint_intelligence,replay,__main__}.py`, `tests/test_planthir_replay_blocker_regression.py`, plus stateful `reset_episode()` Gate 5 parity fix and dict-state / CLI replay alignment (`--expected-backend`, `--worlds-json`).
- **Repaired Legacy X1 Worker Commit (`98b9e3318bfa3c02024685bd8668323065f62811`):** replaced the `X1S20ObservablePolicy._S` stub in `anra_v5/elendril/v8x/evaluation.py` with a valid `V8ModelSpec` so `V8BytePolicy` (`B0`) works without altering any original scientific settings (`tools/elendril_v8x_worker.py` preserved untouched), compatibility-mapped from Run 1 (`8e62ca0f2046c0f07910fdd20abee40508631227`).
- **Registered X2 Science Plugin (`PLANTHIR_BACKEND_MODULES=experiments.elendril_x2`):**
  - `experiments/elendril_x2.py`: registers `ExperimentModule(backend_id="elendril_x2")`, split-aware `REGISTERED_X2_DATASETS`, compact `tr_t13_*` canary rendering, and `context_length=640` defaults.
  - `experiments/x2/models.py`: all 5 isolated X2 pairs (`P2`/`P2-dense`, `R2`/`R2-pool`, `U-P`/`U-P-par`, `DM`/`DM-add`, `PS`/`PS-ctrl`) + `P`/`R` on S20 (`19,871,616` params, `context_length=640`), exact §11 delta-memory math, episodic state reset/clone, and surgical lesions.
  - `experiments/x2/tasks.py`: fail-closed `encode_x2_text(..., allow_truncate=False)` (`<=640` bytes across all 13 distinct production datasets verified by `verify_x2_dataset_dispatch_preflight`), compact T13 op-codes (`<=391` bytes at `k=64`), strictly held-out T13 `q_expr` queries, deterministic `digest()` streams, hierarchical balanced cohort matching, and exact T14 DP.
  - `experiments/x2/lineage_runner.py`: explicit lineage routes (`frozen-probe`, `task-adapt`, `horizon-continue`, `scratch-single`), `X1AdaptationWrapper` (`0.0` prediction parity), continuous `TrainingState` `2368 -> 3552 -> 4736` horizon continuation, and `0`-optimizer-step frozen probe verification.
  - `experiments/x2/registry.py`: `192`-row obligation ledger (`174` mandatory + `18` conditional, `174` unique worker units), Run 1 `54/54` legacy `original_legacy_spec_sha256` compatibility mapping (`8e62ca0f -> 98b9e331`), confirmation gate (never auto-selecting `DM`), and `evaluate_x2_completion_audit`.
  - `experiments/x2/campaign.py`: `ElendrilX2Campaign` wrapping `CampaignOrchestrator` & `export_session_archives`, fail-closed route validation, cross-session protected parent checkpoint retention (`ELENDRIL_V8_X2_PARENT_CHECKPOINTS.zip` + descendant-safe weight pruning), global `X2_COMPLETION_AUDIT.json` + `X2_REPORT.md`, and curated A0 rescorer hook.
- **Strictly Gated X3 Package (`experiments/x3/{__init__,factorial,interventions,runner}.py`):**
  - `GATED_PENDING_X2_COMPLETE` (`0.0` GPU seconds spent until `x2_audit["x2_complete"] is True`); outputs strictly separate `ELENDRIL_V8_X3_ANALYSIS.zip`, `X3_GATE_STATUS.json`, `X3_METRICS.json`, `X3_REPORT.md`, and `X3_CROSSWALK.json`.
  - Paired graph factorial (`<=638` bytes, zero `A..P` uppercase prose pollution) + shortcut baselines + Geiger pointer interchange intervention supporting both `X1TypedModel` and `X2TypedModel`.
  - `tools/build_elendril_x2_notebook.py`: deterministic zero-edit Kaggle 2xT4 notebook generator (`notebooks/ELENDRIL_V8_X2_KAGGLE_T4X2.ipynb`, covering `FULL_X2_SESSIONS`).

## 3. Source Pins & Kaggle 2xT4 Launch Workflow
- **Commit 1 — Repaired Legacy X1 Worker (`LEGACY_X1_SHA`):** `98b9e3318bfa3c02024685bd8668323065f62811` (compatibility-mapped from Run 1 `8e62ca0f2046c0f07910fdd20abee40508631227`)
- **Commit 2 — Pinned X2/X3 Runtime Commit (`SOURCE_SHA`):** `bcde233f213c77342757a4777fd4e0f8730db589`
- **Commit 3 — Generated Notebook (`notebooks/ELENDRIL_V8_X2_KAGGLE_T4X2.ipynb`):** SHA-256 `90b707f165f8252ce11eaae2fffe8465615bde653cb9b498a1e44263faeea7d4` (`18,637` bytes; git tip commit records this exact pinned notebook and handoff metadata)
- **Owner Kaggle Run (Zero Edits, Combined 10h Session + Complete Results & Parent Checkpoints):** upload `notebooks/ELENDRIL_V8_X2_KAGGLE_T4X2.ipynb`, select **GPU T4 x2**, enable **Internet**, click **Run All**. Preflight runs `doctor`, full 13-dataset dispatch check, `elendril_x2` canary + `replay-bundle`, and repaired `B0`/`J` smoke checks. Attach Run 1 `ELENDRIL_V8_X2_ANALYSIS.zip` (and subsequent runs' `ELENDRIL_V8_X2_ANALYSIS.zip` + `ELENDRIL_V8_X2_PARENT_CHECKPOINTS.zip`) as `/kaggle/input` to verify and resume automatically (`X1-A3-S20-B0-standard-s7` onward) within `600m` session / `660m` outer / `45m` reserve (`results_only=True`). X3 executes **only** after global `X2_COMPLETE` passes.
