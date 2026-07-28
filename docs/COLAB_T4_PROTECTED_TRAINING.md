# Protected Colab T4 training

This is the operator guide for continuing the canonical An-Ra V4 model on a Google
Colab T4. The notebook is
`notebooks/AN_RA_T4_PROTECTED_TRAINER.ipynb`.

## What this method does

One T4 is the canonical trainer. It restores the latest verified full-resume
checkpoint, trains on the remaining part of the deterministic 170M-token V4
window, and protects new checkpoints in Google Drive every 100 optimizer steps
or 15 minutes, whichever comes first. If Colab disconnects, another authorized
session can run the same notebook and continue from the latest protected
optimizer boundary.

Separate Colab machines do not synchronize gradients. Public-internet gradient
averaging is slower and less reliable than this checkpoint-baton design. Never
run two notebooks with `WORKER_ROLE = "canonical_trainer"` at the same time.

## One-time Drive layout

The account used by the first trainer must contain:

```text
My Drive/
└── AnRa/
    └── cluster/
        ├── resume-step3-837f7721-64m/
        │   ├── manifest.json
        │   └── 33 verified *.chunk files
        ├── v4_phase_a_170m_seed1301.tar.gz.part00
        ├── v4_phase_a_170m_seed1301.tar.gz.part01
        └── checkpoint-vault/              # created during training
```

The prepared baseline is 2,168,037,221 bytes. The compressed 170M-token pack is
147,119,403 bytes. Keep at least 7 GiB free; 10–12 GiB free is preferable.

## Before pressing Run all

1. Open the notebook from the account that owns `My Drive/AnRa/cluster`.
2. Choose **Runtime → Change runtime type → T4 GPU**.
3. Open Colab's key icon and create two private secrets:
   `ANRA_MANIFEST_SIGNING_KEY` and `ANRA_EVIDENCE_SIGNING_KEY`.
4. Give both secrets long random values of at least 32 characters and grant the
   notebook access to them.
5. Keep `WORKER_ROLE = "canonical_trainer"` for the active trainer.
6. Press **Run all**, then approve the Google Drive mount.

The notebook refuses to train without a real T4, the Drive assets, both secrets,
a clean Git checkout, valid file hashes, a compatible full-resume checkpoint,
and a signed remaining-token window.

## Taking over after a disconnection

Do not start a replacement while the previous trainer still runs. Once it has
ended:

1. Open the same notebook in an authorized account.
2. Ensure the shared `AnRa/cluster` folder is available at the same My Drive
   path. If it was shared to the account, add a My Drive shortcut named `AnRa`.
3. Use the same two signing-secret values.
4. Change `WORKER_ID` to a unique name.
5. Select a T4 and press **Run all**.

The notebook reads `checkpoint-vault/canonical.json`, verifies every
content-addressed chunk, reconstructs the latest checkpoint locally, and signs a
new continuation. The old checkpoint remains immutable.

## Other authorized T4 roles

Set `WORKER_ROLE = "verify_only"` to verify the source, data pack, and latest
checkpoint without advancing the model. Evaluators and data builders should use
immutable checkpoint snapshots and must never receive the canonical-writer role.

Google forbids using multiple accounts to evade Colab resource restrictions.
Only use sessions and compute units legitimately granted to each account. For a
true multi-GPU speedup, use one host with multiple locally connected GPUs and
the repository's `torchrun` DDP/FSDP path.
