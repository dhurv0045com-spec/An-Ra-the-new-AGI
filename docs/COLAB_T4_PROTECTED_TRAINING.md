# Protected Colab T4 training

This is the operator guide for continuing the canonical 181M-parameter An-Ra
V4 model with `notebooks/AN_RA_T4_PROTECTED_TRAINER_V4.ipynb`.

## What this method does

One T4 is the canonical trainer. It restores the newest verified full-resume
checkpoint, continues the deterministic V4 token window, and protects a new
checkpoint every 200 optimizer steps or 60 minutes, whichever occurs first.
After a disconnection, another authorized Colab session can run the same
notebook and resume from the last protected optimizer boundary.

Separate Colab machines do not synchronize gradients over the public internet.
Never run two notebooks with `WORKER_ROLE = "canonical_trainer"` simultaneously.

## Drive layout

Share this one folder with every explicitly authorized trainer account using
Editor access:

```text
My Drive/
└── ANRA_T4_TRAINING_HOME/
    ├── anra-v4-current-full-resume.pt
    ├── anra-v4-current-full-resume.json
    ├── v4_phase_a_170m_seed1301.tar.gz.part00
    ├── v4_phase_a_170m_seed1301.tar.gz.part01
    ├── AN_RA_T4_PROTECTED_TRAINER_V4.ipynb
    ├── anra-v4-recovery-signing-keys.json
    └── training-signing-keys.json
```

The two data-pack parts are the training data in this handoff: approximately
140.3 MB compressed. The current full-resume checkpoint is approximately
2.02 GB because it contains model, optimizer, scheduler, scaler, RNG, and
sampler state. The complete folder is approximately 2.16 GB.

Keep at least 4.5 GiB free during training because checkpoint replacement
briefly requires the current checkpoint and one hidden in-progress upload. At
rest, `ANRA_T4_TRAINING_HOME` contains exactly one portable full-resume `.pt`
checkpoint and one tiny JSON record of its verified step and digest. The `.pt`
filename stays stable across updates, so Drive keeps one checkpoint object
instead of putting old multi-gigabyte checkpoint names into the Bin.

Colab's local scratch outbox remains content-addressed for validation and retry,
but those internal chunks are never published into Drive. A replacement is
fully written, size-checked, SHA-256-checked, and atomically promoted before the
previous `.pt` is removed. A failed upload is deleted while the previous
complete checkpoint remains usable.

For another Gmail account, share `ANRA_T4_TRAINING_HOME` with that account as
an Editor. In the other account, open **Shared with me**, right-click the
folder, and choose **Organize → Add shortcut to Drive**. Open the notebook
inside that folder and press **Run all**. Every authorized trainer then reads
and updates the same checkpoint visible to the owner. Do not copy the folder
or its checkpoint into the second account.

## Before pressing Run all

1. Open the notebook from the owner account or an authorized Editor account.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Keep `WORKER_ROLE = "canonical_trainer"` only on the active trainer.
4. Press **Run all** and approve the Google Drive mount.

The notebook refuses to train without a real T4, a clean checkout, compatible
V4 checkpoint, verified data pack, owner-private signing keys, and signed token
window.

The notebook accepts assets only when they are directly inside one mounted
`ANRA_T4_TRAINING_HOME`. It uses the stable current checkpoint and its verified
step record, copies it to local scratch, and verifies the copy's SHA-256 before
training. It retains a one-time compatibility path for an older step-named
checkpoint, but new runs never create step-named Drive checkpoint files.

## Taking over after a disconnection

Do not start a replacement while the earlier trainer is still running.

1. Open the same notebook from an authorized account.
2. Confirm the shared folder is mounted with Editor access.
3. Confirm the persistent signing-key file is visible.
4. Change `WORKER_ID` to a unique value.
5. Select a T4 and press **Run all**.

The notebook verifies the portable checkpoint after copying it to scratch and
signs a continuation. The next Drive checkpoint does not replace the previous
one until verification succeeds.

## Other T4 roles

Use `WORKER_ROLE = "verify_only"` for source, pack, and checkpoint validation
without advancing weights. Evaluators and data builders must never receive the
canonical-writer role.

Use only compute legitimately granted by the provider. For true simultaneous
multi-GPU acceleration, use one low-latency host with locally connected GPUs
and the repository's DDP/FSDP path.
