# `research/planthir/agent.md` — Planthir / Palantiri / Planter Quick Mirror

> **Canonical Guide:** See root [agent.md](../../agent.md) for full details, Path 1 vs Path 2 recipes, Fast Test Budget rules, and the Gotchas Cheat-Sheet.

## Quick Summary for Agents (`anra_v5.planthir` / `anra_v5.palantiri` / `anra_v5.planter`)

1. **Cross-Branch User Command Protocol:** When told *"Use the technology planthir (or planter / palantiri) from the branch `<branch>` (e.g. `hitchhiker`, `Elendril`) to perform the experiment"*:
   - Create an experiment branch (`git checkout -b exp/<short-name>`).
   - Pull from `origin/<branch>` if `anra_v5/planthir` is absent (or older) in the current branch.
   - Verify with `python -m anra_v5.planthir doctor` (`green_signal: true`) and `python -m anra_v5.planthir calibrate` (`6/6`).
2. **Choose Workflow Path:**
   - **Path 1 — Varying Hyperparams / Existing Arms (`spec.json` only):** `init --backend <id>` -> `validate` -> `run` -> `inspect-run` / `resume` -> `verify-bundle --bundle-dir ...` -> `replay-bundle --bundle-dir ...` (`explain-error --code <CODE>` on failures; never edit `.py` engine files).
   - **Path 2 — New Architecture / Environment / Loss (`ExperimentModule` in 1 standalone file):** Implement `model_factory`, `dataset_factory`, `loss_fn`, `kernel_factory` + `policy_factory` using `ExperimentModule` + `GenericExperimentTrainer` + `CampaignOrchestrator` (`export_session_archives`, `build_pinned_kaggle_notebook`). Make **zero edits** to `anra_v5/planthir/` core files (`PLANTHIR_BACKEND_MODULES` auto-loads plugins).
3. **Fast Tests & Core Gotchas:**
   - Use `IndependentWorkerScheduler(..., in_process_runner=...)` in pytest instead of 6+ cold `torch` subprocesses (`<15s`). Child CUDA pinning automatically rewrites `--device cuda:N` -> `cuda:0`.
   - `v5_model.core.ModelSpec` uses `vocabulary` (no `.logits()`; call `core(tokens, *packed_layout(segments, torch_module=torch))`). `IdentityBindings` uses `optimizer_spec_sha256`.
   - Keep `.zip`/`.npz`/`.bin` out of Git (`audit_tree.py`) and limit `progress.md` entries to `<= 5` non-empty lines.
