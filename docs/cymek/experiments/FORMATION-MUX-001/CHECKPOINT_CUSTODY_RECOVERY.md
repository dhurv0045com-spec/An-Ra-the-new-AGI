# FORMATION-MUX-001 checkpoint custody record and recovery request

Scope: artifact custody only. This record does not revise S5 results, the
preregistrations, RETURNED_BUNDLE_AUDIT.md, or any historical verdict. The
absence of `resume.pt` from the returned bundle was already documented in
`observed/S5_KAGGLE_2026-09-15/POSTMORTEM.md`; this record adds the exact
machine-readable recovery inventory.

## Machine-readable inventory

`CHECKPOINT_CUSTODY.json` (schema `anra-formation-checkpoint-custody/v1`,
raw-file SHA-256 `3152a365681caadd7f64df5eab2866faec7d8b5766eb3d2dd9f4240b0b69408a`)
binds the original bundle and enumerates every declared checkpoint identity.

## Verified facts

- Bundle `FORMATION_MUX_001_RESULTS.zip` SHA-256
  `859489d9babc532a4ed1e785af4e4993541b93730f7166dff2946a26cc996bc5` — matches the
  recorded audit identity and the `.sha256` sidecar; no duplicate member names.
- The bundle contains **30 `CHECKPOINT_RECEIPT.json` files declaring 30 distinct
  checkpoint SHA-256 hashes**, and **zero checkpoint-named members** (`.pt`/`.ckpt`/`.bin`).
- 24 receipts are official arms (seed bundles 73011–73014; CS-MECH-002 M0–M3,
  REP-FORM-003A R0/R1); 6 are `ENGINEERING_ONLY` (seeds 99101–99104 plus two
  engineering CS-MECH-002 rows); they are engineering evidence only, not official scientific arms.
- `README_S5.txt` states: "Exact-resume checkpoints stay in Kaggle Output."
- The bundle has **no `SOURCE_COMMIT.txt`** and **no `EXPORT_VERIFIED.json`**
  receipt. The inspected unversioned operator defines a separate export, but
  its existence in current source does not establish that this S5 run executed
  it. Export history and availability outside this ZIP remain unknown.
- In the inspected source, receipt hashes bind serialized checkpoint bytes produced by
  `_save_checkpoint` (`anra_v5/formation_mux_train_v2.py:194–210`); the S5
  worker (`formation_mux_train_v5.py`) wraps the same saver. The
  `resume.pt` path per arm is inferred from inspected source, not declared
  inside the receipts (`path_basis` field marks this).

## Recovery request

The full official-arm recovery set comprises 24 checkpoints; a narrower
diagnostic requires its corresponding arms. Endpoint checkpoints alone do not
recover intermediate weight trajectories. Recovery requires the original
bytes from the Kaggle Output session(s) associated with this returned S5 bundle, each verifiable against
its declared `sha256` in the inventory. Retraining, regeneration, or a
placeholder cannot substitute: new bytes would be new evidence, not recovery.

Per-row recovery targets (full hashes) are in `CHECKPOINT_CUSTODY.json` under
`rows[].declared_checkpoint_sha256` with `checkpoint_path_candidate`.

## Limits

- Only checkpoint receipts and the README payload were opened; no evaluation
  rows were read and no model bytes were loaded.
- Whole-ZIP digest matches; other payload schemas were not revalidated here.
- No global or exhaustive local absence claim. Filename searches returned no
  `resume.pt` in this audit worktree or `C:/Users/ankit/cyr012-evidence`.
  Downloads contained 37 `.pt` candidates; these were not hash-compared and
  therefore do not establish local presence or absence of the requested bytes.
  Search-tool exclusions and other filenames/archives remain outside scope.
- The S5 execution source is not independently bound to this bundle (no
  `SOURCE_COMMIT.txt`); hash semantics come from the inspected source tree.
- No scientific endpoint or mechanism conclusion is changed by this record.
