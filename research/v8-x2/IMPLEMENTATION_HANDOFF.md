# ELENDRIL V8-X2 Implementation Handoff

## 1. Qualification & Release Status
- **Current Stage:** `CPU_QUALIFIED_READY_FOR_OWNER_CANARY` (no S20 19.87M GPU training or confirmation executed locally).
- **4-Axis Status:** `execution=PENDING`, `delivery=PENDING`, `scientific=UNRUN`, `causal=UNRESOLVED`.
- **§0 Independent Replay Blocker:** FIXED and verified via fresh-subprocess public CLI regression (`init -> validate -> run -> verify-bundle -> replay-bundle`) on stock `x1_s20`, `elendril_x2`, deliberate input tamper, wrong backend, unregistered plugin backend (`MISSING_BACKEND_PLUGIN`), legacy non-V8 missing reference (`INSUFFICIENT_REPLAY_EVIDENCE`), and legacy V8 compatibility.
- **Target Hardware Gates Still Pending Owner Execution:** live Kaggle 2xT4 canary + `replay-bundle`, dual-lane physical device/peak-VRAM profiling, full S20 waves, and locked confirmation scoring.

## 2. Implemented Architecture & Palantiri Integration
- **Palantiri Replay Blocker Fix:** `anra_v5/planthir/{cartridge,checkpoint_intelligence,replay,__main__}.py`, `tests/test_planthir_replay_blocker_regression.py`.
- **Registered X2 Science Plugin (`PLANTHIR_BACKEND_MODULES=experiments.elendril_x2`):**
  - `experiments/elendril_x2.py`: registers `ExperimentModule(backend_id="elendril_x2")` exposing all 5 hooks (`model_factory`, `dataset_factory`, `loss_fn`, `kernel_factory`, `policy_factory`), grouped D2 loss + scalar gradient parity, and complete §0.3 CPU fixture.
  - `experiments/x2/models.py`: all 5 isolated X2 pairs (`P2`/`P2-dense`, `R2`/`R2-pool`, `U-P`/`U-P-par`, `DM`/`DM-add`, `PS`/`PS-ctrl`) + `P`/`R` on S20 (`19,871,616` params, `<=5%` head overhead, `<=2%` active-control parity, byte-identical shared backbone init per seed), exact §11 delta-memory math, episodic state reset/clone, and surgical lesions.
  - `experiments/x2/tasks.py`: Independent Generator B + Renderer B + reference interpreter + 2x2 canonical/narrative factorial + shortcut baselines; D1-H selective-edit + untouched-sibling + stale-late-lure panel; T13 FST decisive §11 witness pair + 32/8/12 family banks + 512-law bounded symbolic version-space solver; T14 costly-info 6-stratum panel + §11 price crossover (`+0.18` cheap vs `-0.15` expensive) + exact public-belief DP; stateful truth-firewalled `X2EnvironmentKernel` & `X2ObservablePolicy`.
  - `experiments/x2/registry.py`: exact 42-job X1 core inventory verification, full 126-base-equivalent X2 registry + separate non-base ledgers (6 T14 routing prep, 48 T13 transfer parents), test-first reader cards, first discriminating report, and Holm-corrected 4-axis qualification report reusing Palantiri primitives.
  - `experiments/x2/campaign.py`: `ElendrilX2Campaign` wrapping `CampaignOrchestrator` & `export_session_archives`, dual-source routing (`8e62ca0f` legacy X1 vs `elendril_x2`), verified hash/step/spec resume skip list, dual-clock (`480m` outer / `330m` session + `45m` reserve) admission, and curated A0 rescorer hook.
  - `tools/build_elendril_x2_notebook.py`: deterministic zero-edit Kaggle 2xT4 notebook generator (`notebooks/ELENDRIL_V8_X2_KAGGLE_T4X2.ipynb`).

## 3. Source Pins & Kaggle 2xT4 Launch Workflow
- **Pinned X2 Runtime Commit (`SOURCE_SHA`, Commit A3):** `596c5e3dd07cb30bd636b68250d11d960adf67ae`
- **Pinned Legacy X1 Worker Commit (`LEGACY_X1_SHA`):** `8e62ca0f2046c0f07910fdd20abee40508631227`
- **Generated Notebook (`notebooks/ELENDRIL_V8_X2_KAGGLE_T4X2.ipynb`) SHA-256:** `2b7b4953c50b245c97679fb411e33ff1896890bf03413d1c8bc6683f2972e5a9`
- **Owner Kaggle Run (Zero Edits, Combined 10h Session + Single Complete Results ZIP):** upload `notebooks/ELENDRIL_V8_X2_KAGGLE_T4X2.ipynb`, select **GPU T4 x2**, enable **Internet**, click **Run All**. **Run 1 requires NO `/kaggle/input` attachment** (A0 is already committed at `Experiment Result/V8-X1/A0/`) and runs up to 43 jobs (`A1..U1[0]`, covering Sessions 1 + 2 combined) within `600m` session / `660m` outer / `45m` reserve (`9.25h` usable training). Completed jobs automatically prune bulky `.bin` / optimizer / `checkpoint_full.zip` weight blobs while preserving 100% of episodes, mistakes, trajectories, summaries, probes, Tier-2 `.npz` capsules, specs, and `result.zip`, refreshing `ELENDRIL_V8_X2_ANALYSIS.zip` atomically after every completed wave (`results_only=True`). **Run 2+** attaches Run 1's output ZIP as `/kaggle/input` to auto-resume the rest.
