# SFT Drive handoff: the exact ready state

The V4 SFT notebook is ready to run only when one shared folder contains the
same parent checkpoint and the same audited SFT manifests. The folder name is
`ANRA_T4_TRAINING_HOME`. Share that one folder with **Editor** access and add a
shortcut to each Colab account's `My Drive`.

## Required folder contents

```text
ANRA_T4_TRAINING_HOME/
├── anra-v4-current-full-resume.pt
├── anra-v4-current-full-resume.json
└── sft-v4/
    ├── sft-v4-train.jsonl
    ├── sft-v4-train.manifest.json
    ├── sft-v4-validation.jsonl
    ├── sft-v4-validation.manifest.json
    ├── sft-v4-test.jsonl
    ├── sft-v4-test.manifest.json
    ├── sft-v4-source-receipts.json
    ├── sft-v4-pilot-audit.json
    └── anra-v4-sft-lineage.json  # created by the first notebook run
```

The parent checkpoint must be the real V4 **full-resume** checkpoint, not an
`fp16` inference file, a `.partial` file, a chunk directory, or an old V3
checkpoint. Its JSON sidecar must describe the same byte size and global step.
Do not rename it after the notebook has created the lineage manifest.

The JSONL files and their manifests are produced by
`scripts/build_sft_v4_dataset.py` (or the bounded pilot helper
`scripts/prepare_sft_v4_pilot_from_reasoning.py`). A raw `.txt`, a pretraining
token pack, or a single conversational JSONL file is not a substitute: the
trainer needs the signed train/validation split and its source receipt.

## What must *not* be in Drive

- `training-signing-keys.json` or unrelated legacy private keys.
- Duplicate checkpoint copies, `.partial` uploads, old V3/V2 artifacts, or
  chunk caches. They consume storage and can make account discovery ambiguous.
- A second SFT folder. There is one canonical writer and one canonical folder.

For the trusted-account workflow, place the owner-created
`anra-sft-manifest-signing-key.json` in the shared `sft-v4/` folder. The
notebook loads its `key` field automatically. A Colab Secret named
`ANRA_MANIFEST_SIGNING_KEY` remains supported as a fallback when the shared
file is unavailable. Anyone who can edit this folder can forge SFT evidence,
so share it only with accounts you control.

## Your only operator actions

1. Put the exact parent checkpoint and its sidecar in the folder above.
2. Build and upload the audited SFT artifacts; ensure both train and validation
   splits contain all eight categories and at least three independent
   `split_group` values.
3. Share the folder with Editor access, add its shortcut to the account running
   Colab, and select **Runtime → Change runtime type → T4 GPU**.
4. Open `notebooks/AN_RA_T4_SFT_V4.ipynb` and press **Run all**. The default is
   a bounded 15-minute pilot and saves every 200 optimizer steps or 15 minutes.
5. Inspect `sft-v4/latest_sft_report.json`. Only after reviewing its parent vs
   child validation loss and sample behavior should you create the signed full
   approval and run full mode.

The notebook clones branch `iterate500` and pins reviewed commit
`fa2e77b8db06e511f1ff436db39d624b72b02801`. It records an eight-prompt
behavior probe every 500 optimizer steps as well as validation loss, so a
falling loss cannot hide a collapsed response pattern. If the notebook cannot
fetch or verify that commit, stop and update the source before training.

## Readiness check

The folder is ready when the notebook's asset cell prints a writable
`training_home`, a non-negative `parent_step`, and the preflight reports
`passed: true`. A failure is an asset, permission, hash, or lineage problem;
do not bypass it by changing a filename or enabling `allow_unregistered`.
