# `research/planthir/agent.md` — Palantiri / Planthir Quick Mirror

> **Canonical Guide:** See root [agent.md](../../agent.md) for the one-sentence user prompt protocol, cross-branch `origin/palantiri` installation commands, `anra-planthir-spec-v1` JSON templates, and Python extension recipes.

## Quick Summary for Agents (`anra_v5.palantiri` / `anra_v5.planthir`)

1. **One-Sentence User Command Protocol:** When told *"Use the technology palantiri from the branch. Change the branch name into something different from this branch. Make this experiment with the help of the technology palantiri"*:
   - Create a new branch (`git checkout -b exp/<short-name>`).
   - Pull from `origin/palantiri` if `anra_v5/palantiri` is absent.
   - Verify with `python -m anra_v5.palantiri doctor` (`green_signal: true`) and `python -m anra_v5.palantiri calibrate` (`6/6`).
2. **Declarative CLI workflow:**
   - `python -m anra_v5.palantiri doctor`
   - `python -m anra_v5.palantiri schema`
   - `python -m anra_v5.palantiri calibrate`
   - `python -m anra_v5.palantiri init --backend x1_rd2 --out spec.json` (or `--backend v8`)
   - `python -m anra_v5.palantiri validate --spec spec.json --canary`
   - `python -m anra_v5.palantiri run --spec spec.json --work-root runs/exp1 --canary`
   - `python -m anra_v5.palantiri inspect-run --work-root runs/exp1`
   - `python -m anra_v5.palantiri resume --spec spec.json --work-root runs/exp1 --canary`
   - `python -m anra_v5.palantiri verify-bundle --bundle-dir runs/exp1/result_bundle`
   - `python -m anra_v5.palantiri replay-bundle --bundle-dir runs/exp1/result_bundle --output runs/exp1/REPLAY_RECEIPT.json`
   - `python -m anra_v5.palantiri explain-error --code <CODE>`
3. **Rules & Honesty:** CPU canary enforces `<= 100,000` parameters, `<= 32` samples, `<= 8` updates. CUDA production runs require a clean git tracked tree matching `"source_commit"`. Do not fabricate productivity savings (`STUDY_OBSERVATIONS_PENDING` until paired logs are provided) and keep `.zip`/`.npz`/`.bin` out of Git per root `AGENTS.md`.
