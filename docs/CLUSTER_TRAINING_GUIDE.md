# Easy Guide: Train An-Ra with Authorized Colab Accounts

Updated: 2026-07-23  
Purpose: explain the checkpoint-baton cluster in plain language and show how
separate provider-authorized Gmail/Colab sessions help without corrupting one
model.

## The simple idea

Three Colabs are not three pieces of one GPU. Internet-separated Colabs cannot
safely behave like one synchronous multi-GPU machine.

Instead, imagine one protected notebook being passed between workers:

```text
Trainer A updates the model
  → saves and verifies a full checkpoint
  → Trainer B can continue from that exact checkpoint

At the same time:
  Evaluator checks an immutable checkpoint
  Data builder prepares the next token pack
  Archive worker copies and verifies artifacts
```

Only one worker writes canonical model weights. This is the checkpoint baton.

Use only accounts and sessions for which Google has granted GPU access. You
perform login, account selection, Colab authorization, and the initial
**Run all** manually. The cluster does not create accounts, rotate accounts, or
evade provider quotas.

## What each repository does

- **An-Ra repository**: tokenizer, data, model, training, checkpoints,
  evaluation, and model truth.
- **GPU Cluster repository**: coordinator, worker identities, job leases,
  handoff, storage transfer, operator controls, and audit.

They communicate through signed JSON. Keep both exact commits pushed.

## Recommended three-account layout

| Authorized session | Initial role | What it does |
| --- | --- | --- |
| Account/session 1 | `canonical_trainer` | Advances the one canonical checkpoint |
| Account/session 2 | `standby` | Preloads and takes over after a safe handoff |
| Account/session 3 | `data_builder` | Prepares the next deterministic pack |

After data preparation, session 3 can restart as `evaluator`. Roles are not
automatically changed behind the owner’s back.

## Before opening Colab

### 1. Confirm both repositories are clean and pushed

```powershell
git -C "C:\Users\ankit\Downloads\An-Ra-the-new-AGI-1" status --short --branch
git -C "C:\Users\ankit\Downloads\gpu cluster" status --short --branch
```

There should be no uncommitted training-code changes.

### 2. Verify the shared contract

```powershell
Set-Location "C:\Users\ankit\Downloads\gpu cluster"
.\.venv\Scripts\python.exe scripts\verify_anra_integration.py `
  --anra-repo "C:\Users\ankit\Downloads\An-Ra-the-new-AGI-1"
```

Proceed only when `"passed": true`.

### 3. Prepare secrets

Copy `.env.example` to `.env` in the cluster repository and replace every
placeholder with a different random secret:

```powershell
Copy-Item .env.example .env
```

Important coordinator secrets:

- `ANRA_OPERATOR_TOKEN`
- `ANRA_CREDENTIAL_KEY`
- `ANRA_RELEASE_SIGNING_KEY`
- `ANRA_MANIFEST_SIGNING_KEY`
- `ANRA_EVIDENCE_SIGNING_KEY`
- `ANRA_EVIDENCE_KEY_ID`

Never paste secrets into Git, Markdown, screenshots, or ordinary notebook
cells. Use Colab Secrets.

## Start the coordinator

From the GPU Cluster repository:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app `
  --host 0.0.0.0 `
  --port 8000
```

The Colab sessions must be able to reach this coordinator through an HTTPS URL.
For a remote deployment, configure `ANRA_ALLOWED_ORIGINS`, `FRONTEND_URL`, and
`REDIRECT_URI` for that exact origin.

Authorize Google Drive through:

```text
https://your-coordinator.example/api/auth/login
```

Drive is the active exchange vault. The laptop remains the deep archive.

## Prepare a signed An-Ra launch

From the An-Ra repository:

```powershell
$env:ANRA_MANIFEST_SIGNING_KEY = "<same manifest key as coordinator>"
.\.venv-cuda\Scripts\python.exe scripts\create_cloud_launch.py `
  --pack-root "C:\path\to\170m-pack" `
  --output "output\v2\cluster_launch_170m.json" `
  --artifact-path "output/v2/checkpoints/anra-v4-180m.pt" `
  --checkpoint-source scratch `
  --worker-id trainer-1 `
  --runtime-estimate-hours 3 `
  --model-profile anra-v4-180m `
  --stage canary
```

Do not edit the JSON after signing.

## Create the campaign

Use real paths and byte counts for your pack and storage. From the cluster
repository:

```powershell
.\.venv\Scripts\python.exe scripts\bootstrap_reliable_campaign.py `
  --coordinator "https://your-coordinator.example" `
  --operator-token "<operator token>" `
  --campaign-id "anra-v4-foundation-001" `
  --generation 1 `
  --phase "v4-foundation" `
  --expected-checkpoint-bytes 2300000000 `
  --tokenizer "C:\Users\ankit\Downloads\An-Ra-the-new-AGI-1\tokenizer\tokenizer_v4_32k.json" `
  --architecture "C:\Users\ankit\Downloads\An-Ra-the-new-AGI-1\training\v2_config.py" `
  --data-manifest "C:\path\to\pack\pack_manifest.json" `
  --corpus-bytes <ACTUAL_PACK_BYTES> `
  --available-storage-bytes <ACTUAL_DRIVE_FREE_BYTES> `
  --anra-repo "C:\Users\ankit\Downloads\An-Ra-the-new-AGI-1" `
  --jobs "campaign_jobs.rescue.example.json" `
  --drive-folder-id "<drive folder id>" `
  --launch-manifest "train-v4-window-001=C:\Users\ankit\Downloads\An-Ra-the-new-AGI-1\output\v2\cluster_launch_170m.json"
```

Replace the example job ID with the exact ID in the reviewed job JSON. The
bootstrap response reveals the campaign key once. Store it immediately in each
authorized Colab’s Secrets.

## Configure each Colab

Open `worker/AN_RA_CLUSTER_WORKER.ipynb` from the GPU Cluster repository in
each authorized account. Add these names to Colab Secrets:

```text
ANRA_COORDINATOR_URL
ANRA_CAMPAIGN_ID
ANRA_CAMPAIGN_KEY
ANRA_ACCOUNT_EMAIL
ANRA_WORKER_ID
ANRA_GENERATION
ANRA_WORKER_ROLE
ANRA_CAPABILITIES
ANRA_SOURCE_COMMIT
ANRA_CLUSTER_SOURCE_COMMIT
ANRA_MANIFEST_SIGNING_KEY
ANRA_EVIDENCE_SIGNING_KEY
ANRA_EVIDENCE_KEY_ID
```

Values that differ per session:

| Secret | Example for trainer | Example for standby |
| --- | --- | --- |
| `ANRA_ACCOUNT_EMAIL` | account1 address | account2 address |
| `ANRA_WORKER_ID` | `trainer-1` | `standby-1` |
| `ANRA_WORKER_ROLE` | `canonical_trainer` | `standby` |
| `ANRA_CAPABILITIES` | `canonical_training` | `canonical_training,standby` |

The data worker uses role `data_builder` and capability `data_prepare`.
The evaluator uses role `evaluator` and capability `evaluation`.

Select a T4 runtime, confirm Colab granted CUDA, and click **Run all** once.
The notebook clones the pinned commits, mounts Drive, registers the worker,
verifies its signed lease, and runs only allowlisted An-Ra commands.

## The first safe run

Do not begin with hours of training:

1. Let the canonical trainer run for 10–15 minutes.
2. Confirm a `full_resume` artifact is `canonical_verified`.
3. Copy its manifest/hash to the laptop archive.
4. Drain or interrupt trainer 1.
5. Reconcile expired work.
6. Let standby claim the new attempt.
7. Confirm step, tokens, and sampler cursor continue forward.

After this one handoff succeeds, continue toward the 200M-token milestone.

## How to watch the campaign

Use an operator bearer token:

```text
GET  /api/v2/campaigns/current
GET  /api/v2/storage/quota
GET  /api/v2/evidence/export?campaign_id=<id>
POST /api/v2/campaigns/<id>/pause
POST /api/v2/campaigns/<id>/drain
POST /api/v2/campaigns/<id>/resume
POST /api/v2/campaigns/<id>/reconcile
POST /api/v2/releases/rollback
```

Do not trust only the Colab progress bar. Trust the campaign lease, signed
progress journal, durability state, replica receipt, and checkpoint hash.

## If you close your laptop

Training continues only if:

- the coordinator is hosted somewhere that remains online;
- the Colab worker remains connected and provider runtime stays alive;
- Drive remains mounted and reachable.

If the coordinator is running only on the laptop, closing or sleeping it makes
the coordinator unavailable. After repeated failed heartbeats the worker stops
its training subprocess and cannot start unleased work. Recovery begins from
the newest checkpoint that had already become verified, so host the coordinator
somewhere persistent before relying on laptop shutdown.

## Common mistakes

- **Two trainers use the same checkpoint at once:** blocked by canonical lease.
- **A Gmail account has no GPU grant:** that session cannot register as a GPU
  worker; do not automate quota workarounds.
- **Checkpoint visible but not verified:** do not terminate compute yet.
- **Compact checkpoint used for resume:** rejected; obtain `full_resume`.
- **Wrong commit:** reclone the pushed commit instead of bypassing validation.
- **Worker disappears:** reconcile and resume from newest verified artifact.
- **Drive nearly full:** drain, archive, and protect current plus previous
  resume generations before deleting anything.
- **New pack repeats old tokens:** coordinator rejects the repeated signed
  window.

## When separate GPUs really train simultaneously

DDP/FSDP requires GPUs on the same low-latency host. The cluster contains a
fail-closed future launcher, `scripts/launch_same_host.py`, but An-Ra’s runtime
must implement and prove distributed process-group support first. Do not use
public-internet gradient averaging between Colabs.

## Live truth sources

- Cluster contract:
  `C:\Users\ankit\Downloads\gpu cluster\contracts\anra-training-contract.v4.json`
- Cluster README and runbook:
  `C:\Users\ankit\Downloads\gpu cluster\README.md`,
  `ANRA_CLUSTER_RUNBOOK_DRAFT.md`
- Worker behavior: `gpu cluster/worker/campaign_worker.py`
- Coordinator behavior: `gpu cluster/backend/campaign.py`
- Run truth: `/api/v2/campaigns/current`
- Storage truth: `/api/v2/storage/quota`
- An-Ra run contract: the signed schema-4 launch JSON
- Checkpoint truth: canonical pointer, immutable manifest, and replica receipts

Example filenames are instructional. The signed campaign and live coordinator
state are authoritative for an actual run.
