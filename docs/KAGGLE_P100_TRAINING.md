# Protected Kaggle P100 training

Use `notebooks/AN_RA_KAGGLE_P100_PROTECTED_TRAINER_V4.ipynb` to let one
Kaggle P100 continue the canonical 181M-parameter V4 foundation. This is a
sequential checkpoint baton, not internet-based distributed training.

## Prepare the private input Dataset

Create one **private** Kaggle Dataset containing a snapshot folder with the
files required by the checkpoint's next token window:

```text
ANRA_T4_TRAINING_HOME/
├── anra-v4-current-full-resume.pt
├── anra-v4-current-full-resume.json
├── v4_phase_a_cont_170m_to_500m_seed1301.tar.gz
└── training-signing-keys.json       # optional compatibility path
```

For a checkpoint below 170M tokens, include the two original 170M archive
parts instead. The notebook verifies that exactly one checkpoint lineage is
attached and checks the pointer size and SHA-256 before loading anything.

Prefer Kaggle Secrets named `ANRA_MANIFEST_SIGNING_KEY` and
`ANRA_EVIDENCE_SIGNING_KEY`. Keeping `training-signing-keys.json` in the
private Dataset is supported only for parity with the existing trusted-account
workflow. Never make the Dataset or notebook public while it contains a key.

## Run

1. Create or import the protected notebook.
2. Attach the private input Dataset.
3. In notebook settings, select **GPU P100** and enable Internet.
4. Confirm no Colab or other Kaggle canonical trainer is active.
5. Press **Run All**.

The notebook uses a micro-batch of one, gradient accumulation of eight, a
480-minute session budget with a 30-minute drain reserve, and protected saves
every 200 optimizer steps or 60 minutes. It reads the saved optimizer step and
Phase A token count, selects the correct immutable pack, and resumes instead
of starting a new model.

## Export the baton

Kaggle inputs are read-only, so the result is published under:

```text
/kaggle/working/ANRA_KAGGLE_EXPORT/
├── anra-v4-current-full-resume.pt
└── anra-v4-current-full-resume.json
```

When the session finishes, save a private notebook version or download that
folder. Verify the JSON size and SHA-256, then replace the matching checkpoint
pair in the shared Drive training home before another worker starts. Do not
copy only the `.pt`; the JSON pointer is part of the handoff contract.

Kaggle output persistence is not the canonical vault. The handoff is complete
only after the verified pair exists in the shared owner-controlled storage.
