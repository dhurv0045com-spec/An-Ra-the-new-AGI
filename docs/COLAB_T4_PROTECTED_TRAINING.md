# Protected Colab T4 training

This is the operator guide for continuing the canonical 181M-parameter An-Ra
V4 model with `notebooks/AN_RA_COLAB_T4_PROTECTED_TRAINER_V4.ipynb`.

## What this method does

One T4 is the canonical trainer. It restores the newest verified full-resume
checkpoint, continues the deterministic V4 token window, and protects a new
checkpoint every 200 optimizer steps or 60 minutes, whichever occurs first.
After a disconnection, another authorized Colab session can run the same
notebook and resume from the last protected optimizer boundary.

The notebook reads Phase A token progress from the checkpoint and selects the
immutable catalog window containing the next token. The original pack covers
0-170M tokens; the continuation pack covers 170M-500M. Restarting inside either
range resumes that range's saved sampler cursor, while crossing 170M selects the
continuation automatically. The 181M number is the model's parameter count,
not a token limit.

Separate Colab machines do not synchronize gradients over the public internet.
Never run two training-enabled copies against the same canonical folder simultaneously.

## Import into Google Colab

1. Upload `notebooks/AN_RA_COLAB_T4_PROTECTED_TRAINER_V4.ipynb` to Google Drive or open it directly in Colab.
2. In Colab, choose **Runtime → Change runtime type → T4 GPU**.
3. Run all once with `RUN_TRAINING = False`; this is a read-only preflight and needs no Kaggle account, dataset, checkpoint, or Drive training folder.
4. Review the printed status. A clean preflight ends with `PREFLIGHT COMPLETE` and never starts training.
5. Only after the canonical assets pass should an authorized trainer set `RUN_TRAINING = True` and run the session.

The notebook catches environment, clone, Drive, asset, and trainer failures and prints the exact diagnostic instead of raising a traceback. It cannot validate credentials, GPU availability, or Drive permissions until it runs in a real Colab runtime.

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
    ├── AN_RA_COLAB_T4_PROTECTED_TRAINER_V4.ipynb
    ├── anra-v4-recovery-signing-keys.json
    └── training-signing-keys.json
```

The two original data-pack parts contain the first 170M-token window. The
276,647,985-byte `v4_phase_a_cont_170m_to_500m_seed1301.tar.gz` archive contains
the unique 170M-500M window. Its SHA-256 is
`330ac75a2dbe20a3bd6608adf1c7325fd4a8bf7e81be546ccd8e5f93877f4888`.
Upload it once directly into `ANRA_T4_TRAINING_HOME`; do not extract it in
Drive. The current full-resume checkpoint is approximately 2.02 GB because it
contains model, optimizer, scheduler, scaler, RNG, and sampler state.

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

1. Open `AN_RA_COLAB_T4_PROTECTED_TRAINER_V4.ipynb` from the owner account or an authorized Editor account.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Leave `RUN_TRAINING = False` for the first run and review the preflight results.
4. Set `RUN_TRAINING = True` only when Drive, the checkpoint, the signing identity, and the data pack pass every check.
5. Press **Run all** and approve the Google Drive mount.

The default preflight is non-destructive and does not require Kaggle, a GPU, Drive assets, or a checkpoint. The training path refuses to start without a real GPU, a clean checkout, a compatible V4 checkpoint, verified data-pack bytes, owner-private signing keys, and a signed token window.

The notebook accepts assets only when they are directly inside one mounted
`ANRA_T4_TRAINING_HOME`. It uses the stable current checkpoint and its verified
step record, copies it to local scratch, and verifies the copy's SHA-256 before
training. It retains a one-time compatibility path for an older step-named
checkpoint, but new runs never create step-named Drive checkpoint files.

If Colab reports `OSError: [Errno 107] Transport endpoint is not connected`
while staging the checkpoint, the notebook removes the partial scratch copy,
remounts Drive, rediscovers the canonical training home, and retries up to two
times. The scratch checkpoint is promoted only after its exact byte size and
SHA-256 match the pre-disconnection source. Digest mismatches and ordinary
filesystem failures do not retry and cannot replace a verified local copy.

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

## Safe validation and handoff

Keep `RUN_TRAINING = False` for source, Drive, checkpoint, and pack validation without advancing weights. Only the active handoff session may set `RUN_TRAINING = True`; do not run two training sessions against the same canonical folder.

Use only compute legitimately granted by the provider. For true simultaneous multi-GPU acceleration, use one low-latency host with locally connected GPUs and the repository's DDP/FSDP path.
