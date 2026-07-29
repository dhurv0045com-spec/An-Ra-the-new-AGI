# Protected Colab T4 training

This is the operator guide for continuing the canonical An-Ra V4 model on a Google
Colab T4. The notebook is
`notebooks/AN_RA_T4_PROTECTED_TRAINER_V4.ipynb`.

## What this method does

One T4 is the canonical trainer. It restores the latest verified full-resume
checkpoint, trains on the remaining part of the deterministic 170M-token V4
window, and protects new checkpoints in Google Drive every 200 optimizer steps
or 60 minutes, whichever comes first. If Colab disconnects, another authorized
session can run the same notebook and continue from the latest protected
optimizer boundary.

Separate Colab machines do not synchronize gradients. Public-internet gradient
averaging is slower and less reliable than this checkpoint-baton design. Never
run two notebooks with `WORKER_ROLE = "canonical_trainer"` at the same time.

## One-time Drive layout

Keep one live folder and share it with every explicitly authorized trainer
account using **Editor** access:

```text
My Drive/
└── AnRa/
    ├── cluster/
    │   ├── checkpoint-vault/                 # real folder, never a ZIP
    │   ├── v4_phase_a_170m_seed1301.tar.gz.part00
    │   ├── v4_phase_a_170m_seed1301.tar.gz.part01
    │   └── AN_RA_T4_PROTECTED_TRAINER_V4.ipynb
    └── private/
        └── training-signing-keys.json
```

The full resume state is about 2.17 GB. The compressed 170M-token pack is
147,119,403 bytes. Keep at least 7 GiB free; 10–12 GiB free is preferable.
The vault intentionally retains the newest two protected resume generations.
After a replacement is protected, older manifests and unreferenced chunks are
removed immediately. Failed partial uploads are garbage-collected instead of
remaining in Drive.

For a second Gmail account, open the shared folder in Drive and choose
**Organize → Add shortcut to Drive**. The notebook searches canonical MyDrive
paths, MyDrive root, shortcut targets, and Shared Drives. All authorized
trainers therefore write to the same `checkpoint-vault`; the owner sees every
new canonical checkpoint immediately. Do not upload a compressed
`checkpoint-vault` archive and do not make independent vault copies.

If a data-pack part or the signing-key file is shared with the account but is
not exposed by the mounted shortcut, the notebook requests Drive authorization,
finds the exact shared file, downloads it to temporary Colab storage, and
verifies its expected size and SHA-256 before using it. The checkpoint vault
itself must remain a real writable shared folder; API fallback is never used to
create a second vault.

## Before pressing Run all

1. Open the notebook from the owner account or an authorized Editor account
   with a My Drive shortcut to the shared training folder.
2. Choose **Runtime → Change runtime type → T4 GPU**.
3. Keep `WORKER_ROLE = "canonical_trainer"` for the active trainer.
4. Press **Run all**, then approve the Google Drive mount.

The campaign normally uses the mounted `training-signing-keys.json`. A takeover
account must be granted access to that file explicitly. Signing keys never use
the cross-account Drive API fallback. If and only if every cloud checkpoint is
incomplete and the notebook falls back to the verified step-3 local rehearsal,
it creates one persistent `anra-v4-recovery-signing-keys.json` in the mounted
Drive and reuses it on every later session. The values are never printed.

The notebook refuses to train without a real T4, the Drive assets, both secrets,
a clean Git checkout, valid file hashes, owner-private signing keys, a
compatible full-resume checkpoint, and a signed remaining-token window.

An old canonical pointer naming an incomplete checkpoint cannot block a
verified recovery publication. The notebook scans manifests by descending
global step and resumes the newest candidate whose complete chunk set passes
size and SHA-256 verification. The publisher preserves an audit record before
replacing an invalid pointer. A rescued `anra-v4-emergency-step400.pt` and its
`.pt.json` hash receipt take precedence when present.

## Taking over after a disconnection

Do not start a replacement while the previous trainer still runs. Once it has
ended:

1. Open the same notebook in an authorized account.
2. Ensure the shared folder has Editor access and add its shortcut to My Drive.
   The shortcut can have any name; the notebook resolves its target.
3. Share `training-signing-keys.json` only with the authorized takeover account
   and add a My Drive shortcut to it (or its `private` parent folder), so Colab
   can see it through the mounted filesystem.
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
