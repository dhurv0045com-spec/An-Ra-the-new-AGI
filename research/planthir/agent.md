# `research/planthir/agent.md` — Planthir Usage & Reference (Synced Mirror)

> **Canonical Guide:** See root [agent.md](../../agent.md) for the primary copy-paste installation commands, `anra-planthir-spec-v1` templates, CLI workflow, and Python extension recipes.

## Quick Summary for Agents Inside `research/planthir/`

1. **Install / Copy into any branch:** Check out `anra_v5/planthir`, `anra_v5/elendril`, `v5_contracts`, `v5_evaluation`, `v5_model`, `v5_training`, `pyproject.toml`, `agent.md`, and `research/planthir` from `origin/planthir`, then run `python -m anra_v5.planthir doctor`.
2. **Declarative CLI workflow:**
   - `python -m anra_v5.planthir doctor` (`green_signal: true`)
   - `python -m anra_v5.planthir schema`
   - `python -m anra_v5.planthir calibrate` (`6/6` faults detected)
   - `python -m anra_v5.planthir init --backend x1_rd2 --out spec.json` (or `--backend v8`)
   - `python -m anra_v5.planthir validate --spec spec.json --canary`
   - `python -m anra_v5.planthir run --spec spec.json --work-root runs/exp1 --canary`
   - `python -m anra_v5.planthir inspect-run --work-root runs/exp1`
   - `python -m anra_v5.planthir resume --spec spec.json --work-root runs/exp1 --canary`
   - `python -m anra_v5.planthir verify-bundle --bundle-dir runs/exp1/result_bundle`
   - `python -m anra_v5.planthir replay-bundle --bundle-dir runs/exp1/result_bundle --output runs/exp1/REPLAY_RECEIPT.json`
   - `python -m anra_v5.planthir explain-error --code <CODE>`
3. **Rules & Honesty:** CPU canary enforces `<= 100,000` parameters, `<= 32` samples, `<= 8` updates. CUDA production runs require a clean git tracked tree matching `"source_commit"`. Do not fabricate productivity savings (`anra_v5.planthir.productivity` reports `STUDY_OBSERVATIONS_PENDING` until paired logs are provided) and keep `.zip`/`.npz`/`.bin` out of Git per root `AGENTS.md`.
