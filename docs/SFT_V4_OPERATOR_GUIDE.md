# V4 SFT: from prepared data to one Colab button

SFT is a **child stage** of the V4 foundation model. It changes how the model
answers instructions; it does not continue the raw 170M-token pretraining
window and it never replaces the foundation checkpoint.

## The two data tracks

Continue foundation data separately when needed:

```powershell
.\.venv-cuda\Scripts\python.exe scripts\download_training_data.py --profile 30gb
```

For SFT, collect licensed conversational JSONL sources. Each record needs an
auditable source, license, category, and either a `messages` conversation or a
`prompt`/`answer` pair:

```json
{"messages":[{"role":"user","content":"Explain recursion."},{"role":"assistant","content":"Recursion is when a function calls itself..."}],"category":"instruction_following","source_id":"owner-reviewed-v1","split_group":"owner-reviewed-batch-01","license":"CC-BY-4.0"}
```

The train split must contain all eight categories: instruction following,
dialogue, code, mathematics, decomposition, tool contracts, uncertainty, and
correction. `split_group` identifies related conversations; all records in the
same group remain in one split, preventing source leakage into validation. It
defaults to `source_id` if omitted. Do not mark sources approved until their
terms and quality have actually been reviewed. A canonical corpus needs at
least three independent groups so it can form train, validation, and test
without leaking a group between them.

Every record needs a category. When a downloaded file is dedicated to one
category, put that category in its source-registry entry and the builder will
attach it automatically; otherwise retain a `category` field on each record.

To automate the download, copy
`configs/sft_v4_sources.example.json` to a private source registry, replace
every placeholder with an HTTPS URL, exact SHA-256, and reviewed license, then
run:

```powershell
.\.venv-cuda\Scripts\python.exe scripts\download_sft_v4_sources.py `
  --registry C:\data\sft-v4-sources.json `
  --output-dir C:\data\sft-raw
```

The downloader resumes incomplete files and rejects every source whose final
digest differs from the registry. Its receipt becomes part of the data audit.

For the first bounded pilot, the repository also contains an existing
SmolTalk-derived instruction subset. Prepare a receipt-bound 8-category pilot
from it without downloading a second copy:

```powershell
.\.venv-cuda\Scripts\python.exe `
  scripts/prepare_sft_v4_pilot_from_reasoning.py `
  --output-dir C:\data\anra-sft-v4
```

This emits an audit report with the source revision, labeling rules, sample
prompts, hashes, and category counts. It is suitable for the first 15-minute
pilot; replace or expand it with owner-reviewed sources before treating the
result as a production SFT curriculum.

Build immutable SFT artifacts locally:

```powershell
.\.venv-cuda\Scripts\python.exe scripts\build_sft_v4_dataset.py `
  --input C:\data\sft-source.jsonl `
  --output-dir C:\data\anra-sft-v4 `
  --source-receipts C:\data\sft-raw\sft-v4-source-receipts.json `
  --approve-quality --approve-licenses
```

The builder accepts only files named in the hash-verified receipt. A tiny
owner-local smoke-test may use `--allow-unregistered-inputs`, but that flag is
recorded in the manifest and is not suitable for a canonical SFT campaign.

Copy the generated `sft-v4-train.jsonl`,
`sft-v4-train.manifest.json`, validation/test artifacts, and the signed
lineage manifest into this one shared Drive folder:

```text
ANRA_T4_TRAINING_HOME/
  anra-v4-current-full-resume.pt       # frozen V4 foundation parent
  anra-v4-current-full-resume.json
  tokenizer_v4_32k.json                # optional local convenience copy
  sft-v4/
    sft-v4-train.jsonl
    sft-v4-train.manifest.json
    sft-v4-validation.jsonl
    sft-v4-validation.manifest.json
    sft-v4-test.jsonl
    sft-v4-source-receipts.json
    anra-v4-sft-lineage.json
```

The first Colab run creates the signed lineage after the parent checkpoint and
SFT data are visible together. It binds their SHA-256 hashes; another account
may read the same folder through a Drive shortcut, but must not copy or rename
the canonical artifacts. Put the owner-created
`anra-sft-manifest-signing-key.json` in `sft-v4/`; the notebook loads its `key`
field automatically. A Colab Secret named `ANRA_MANIFEST_SIGNING_KEY` remains
available as a fallback. Anyone with Editor access can forge SFT evidence, so
share this folder only with accounts you control.

## One-button GPU operation

Open `notebooks/AN_RA_T4_SFT_V4.ipynb` from the shared folder, choose a T4,
and press **Run all**. Its default is `RUN_MODE = "pilot"`: a bounded
15-minute SFT run. Full mode uses a 240-minute session budget. It verifies CUDA, the V4 tokenizer, signed lineage, dataset
hashes, parent checkpoint, and Drive write access before any weight update.

The pilot publishes only:

```text
ANRA_T4_TRAINING_HOME/sft-v4/
  anra-v4-current-full-resume.pt
  anra-v4-current-full-resume.json
  latest_sft_report.json
  ready_to_sft.json
```

The foundation `anra-v4-current-full-resume.pt` remains untouched. SFT saves
every 200 optimizer steps or 15 minutes, whichever occurs first. Full SFT is
deliberately **not** enabled merely by changing a notebook variable. First
inspect `latest_sft_report.json`: it records held-out assistant-token loss for
both the frozen parent and the SFT child, plus their delta, and a deterministic
eight-prompt `behavior_smoke` output review, plus the hash-bound
`ready_to_sft.json` gate. The smoke review and readiness gate must pass before
full approval. Then run this explicit approval command from
the same checked-out source and mounted Drive:

```powershell
python -m training.sft_v4 approve-full `
  --dataset-manifest <TRAINING_HOME>\sft-v4\sft-v4-train.manifest.json `
  --validation-manifest <TRAINING_HOME>\sft-v4\sft-v4-validation.manifest.json `
  --lineage-manifest <TRAINING_HOME>\sft-v4\anra-v4-sft-lineage.json `
  --base-checkpoint <TRAINING_HOME>\anra-v4-current-full-resume.pt `
  --vault-root <TRAINING_HOME> `
  --owner-approval "I reviewed the pilot loss and behavior smoke outputs."
```

That writes a signed approval bound to the exact protected pilot checkpoint.
Only then change `RUN_MODE` to `"full"` and press **Run all** again. The full
session runs for up to 240 minutes and resumes the SFT optimizer, scheduler,
RNG, and example cursor from the SFT checkpoint. Replacing the checkpoint
with a newer checkpoint is expected: approval stays bound to the immutable
pilot, and the verifier accepts only a newer checkpoint carrying the same
lineage, parent, dataset, validation, and assistant-only contract. A different
lineage, older checkpoint, or corrupted payload requires another pilot review.

The `sft-v4` Drive folder is a single-file hot vault. The canonical payload is
`anra-v4-current-full-resume.pt`; protected saves atomically replace that path.
Old step-named files and archived 2+ GiB payloads are pruned automatically while
small lineage/report evidence is retained, so restarting Colab must not create
another full checkpoint copy.

Only one authorized account may run with `WORKER_ROLE = "canonical_trainer"`
at a time. A second account takes over only after the first has stopped; it
opens the same shared folder and resumes the SFT child checkpoint.

## Ready criteria

Before a full SFT run, run the notebook preflight. It must report `passed:
true`, a T4 device, all eight categories, a V4 parent hash, and the separate
`sft-v4` vault destination. If it fails, correct the reported asset or
permission issue; never bypass hashes, signatures, or the single-writer rule.
