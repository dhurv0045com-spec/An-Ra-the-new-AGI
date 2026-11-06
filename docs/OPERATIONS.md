# Cloud operations

## Launch decisions

**V4 cognitive-action campaign: NO-GO.** The frozen learner did not request evidence on the revision tasks. A supervised query/answer/abstain positive control is the next gate. The preserved V3 notebooks concern graph composition. They do not train the integrated cognitive learner. Use their canary instructions only if the owner separately reopens that portfolio.

| Lane | Owner-run canary | Research campaign |
| --- | --- | --- |
| Colab, one T4 | **V3 only, conditional GO after preflight.** The notebook and its local contracts are qualified. Device parity and Drive recovery require live evidence. | **NO-GO until canary and recovery pass.** Screening target is 1–2 hours, including export margin. |
| Kaggle, two T4s | **V3 only, conditional GO after preflight.** Verify two distinct physical device identities and two usable CUDA devices. | **NO-GO until the screening gate, both device canaries and a saved-output recovery drill pass.** Target is 5–6 hours including export. |

The notebooks use a single shared pinned operator. Open them with their expensive switches disabled. Choose GPU acceleration, permit internet for the pinned source fetch, then inspect the forecast and output destination. Provide an authorized source checkout if the repository requires authentication. Keep credentials in the platform's secret facility. Never embed access tokens in clone URLs or exported logs.

## Colab

Use the one-T4 screening notebook and select a T4 GPU runtime. Mount Drive only when performing the owner-run recovery canary or campaign. The cloud canary checks numerical updates, same-device restart, scoring and persistence. Keep one session responsible for a run ID. Preserve the verified publication receipt and source identity for a restart. Colab dynamically allocates hardware and session lifetime. [Colab FAQ](https://research.google.com/colaboratory/faq.html)

## Kaggle

Import the dual-T4 notebook, select GPU T4 ×2 and enable Internet. The operator schedules independent lineages and assigns each worker one device. Each worker has the VRAM of one T4. Check account quotas in the notebook settings before opt-in. Save a notebook version with outputs after the canary or a scheduled rung. Download or reattach the versioned output and verify every checkpoint component before resuming in a fresh session. Treat `/kaggle/working` as session-local until versioned output has been independently retrieved. Recoverable saved outputs and session-local canary evidence have different custody. [Kaggle notebooks](https://www.kaggle.com/docs/notebooks) · [Kaggle runtime image](https://github.com/Kaggle/docker-python) · [Versioned output retrieval](https://github.com/Kaggle/kagglehub#download-notebook-outputs)

## Common recovery and export

Preserve a single run ID, the same immutable source, runtime identity, fixture, method, seed and budget across a resumed lineage. Each checkpoint records the optimizer and next update cursor. Copy a verified generation before moving a durable pointer. If a receipt or digest fails, stop and preserve both the prior good generation and the failure log. A new session should restore a verified checkpoint into a fresh local directory. Export compact result receipts separately from full checkpoint payloads.

Run only the canary or campaign selected by the owner. Native platform PyTorch and CUDA must pass compatibility preflight. Operator receipts record the exact runtime used. The notebook hardware paths remain unverified until the returned canary logs are reviewed. [PyTorch installation guidance](https://pytorch.org/get-started/locally/)
