# BRAMASTRA K8 campaign results and recovery notes

**Disposition:** the campaign produced useful training and cognition diagnostics, then failed at the E4 architecture proof because the proof created CPU token indices for a CUDA model. This is a code/device-placement failure, not evidence that either GPU failed. The source fix is in this branch. The campaign did **not** complete E4 or E5 and does not demonstrate AGI, RSI, or general cognition.

## Provenance and integrity

The four supplied ZIP files are alternate result, safety, failure, and probe exports for **one** campaign, not four independent runs. All four passed ZIP CRC validation (`testzip() == None`). Their embedded K8 result identity agrees on source closure, configuration, tokenizer, and data bundle:

| Identity | Value |
|---|---|
| Run | `k8-b796a20ecc04` |
| Instance | `b5e0e88d1ca3` |
| Source closure SHA-256 | `01eeada0ab124aa167b1dd7bb2e7ee0b16c0337d4a369ba1953311b44d9a8177` |
| Configuration identity | `5d9cf2155c9f84e034313cf5a605bf806a66bea8babcb23b446622cc1b5c4ade` |
| Tokenizer identity | `62423dd27ba60f027cd0ca8e01e036cb08ed008005da3f0d49edd0a783290012` |
| Prepared data bundle SHA-256 | `d64ff76188b612f5fa2744bd4739420e3f708a0d88c43be4e8fd1f0392f36681` |
| Allocation | 480 minutes; `cuda:0` and `cuda:1` qualified as two workers |

The pre-campaign build report in `engineering/STATUS.md` used source closure
`adb0a9f79b8a80231de5e6534f7604ad838a0967337a9e75831e3aa595d1831a` and data
identity `7b354e0ac63dfd286e85df19ca7b5b076b8a119fd2184193239099acbf5d0bc0`.
Both identities differ from this campaign. The campaign results therefore do
not qualify that verified build, and that build report does not certify the
source/data used by this campaign.

The campaign log reports `CAMPAIGN_FAILED` and `EXPORTED_PARTIAL`, with 395.44 minutes remaining. It therefore consumed about 84.56 wall-clock minutes before E4 stopped execution.

| Supplied archive | Bytes | SHA-256 |
|---|---:|---|
| `k8-b796a20ecc04-results-b5e0e88d1ca3-failed-full-campaign-b63bc0b3.zip` | 135,927 | `e03d877c27968db6167b6738f9990498c3eca5660e4f6e2c5d05f381aa08d7e1` |
| `k8-b796a20ecc04-results-b5e0e88d1ca3-results-933500ac.zip` | 185,644 | `66d55b33568b804c8e2f393418cb6ec7f0f00162f68f8230dfe15a076ecb8f71` |
| `k8-b796a20ecc04-results-b5e0e88d1ca3-safety-results-1ae02d74.zip` | 186,211 | `3ddc91e68664da8d5519e845dccf9683242c910980dee0bc155ba5d1003cda9a` |
| `k8-b796a20ecc04-results-b5e0e88d1ca3-xprobe-049736db.zip` | 185,639 | `3d4ca5dd81ab3d786970e0c8b71203e4d5829fe40c914a7c8ce7038443a97936` |

## What completed

| Phase | Campaign evidence | Interpretation |
|---|---|---|
| E0 | 12/12 updates committed across two qualified CUDA devices | Two-worker admission and the small calibration path ran. This qualifies execution plumbing, not model capability. |
| E1 | 16,000/16,000 updates committed across four jobs (A/B, seeds 1701/1702). Each job evaluated 32 held-out inventory cases: **0/32 successes**, with all 32 generations per job stopping on EOS. | Neither 4,000-update arm demonstrated the target skill. The repeated early EOS is a high-priority target/stop-token and decoding investigation before spending more updates. |
| E2 | Two seeds; 128 cases per mode per seed; 256 observations per mode. See the aggregate table below. | Cognition/control comparison ran, but policy, memory, and workspace generations all truncated. Planner predictions had zero calibration joins. |
| E3 | 16,000/16,000 updates committed across four jobs. Each job recorded two executed tool outputs with receipts; protected data was excluded. Retention was **0/4** in each job. | Proves tool execution and receipt plumbing. It does not establish learned tool competence or retention. |
| E4 | Both jobs failed before the first update; 0 updates and 0 checkpoint receipts. | Device mismatch in the architecture proof; exact cause and code repair below. |
| E5 | Not run; campaign stopped after E4. | No proposer/self-improvement result exists in this run. |
| E6 | Partial export completed. | Exporting a partial result is not equivalent to completing the campaign. |

### E2 cognition outcomes

| Mode | Successes / 256 | Rate | Truncated / 256 | Inference input tokens | Memory tokens |
|---|---:|---:|---:|---:|---:|
| `a-direct` | 71 | 27.7% | 0 | 68,150 | 0 |
| `a-fixed` | 71 | 27.7% | 0 | 270,944 | 0 |
| `a-random` | 71 | 27.7% | 0 | 269,003 | 0 |
| `b-planner` | 71 | 27.7% | 0 | 394,299 | 0 |
| `b-memory` | 0 | 0% | 256 | 1,755,494 | 667,398 |
| `b-policy` | 0 | 0% | 256 | 1,088,096 | 0 |
| `b-workspace` | 0 | 0% | 256 | 1,088,096 | 0 |
| `symbolic` | 208 | 81.25% | 48 | 89,115 | 0 |

The symbolic row is a separate symbolic comparator, **not** a neural-model score and not evidence that the trained model acquired that capability. The learned-control modes were not competitive with it on this benchmark. The memory variant consumed about 667k recorded retrieval tokens while every episode truncated and none succeeded. The planner recorded `joined=0` and a null mean calibration gap, so there is no calibration evidence to support a claim that its predictions improved decisions.

This points to a concrete next engineering bottleneck: cognition must turn observations into bounded, typed state and executable actions, instead of allowing long unproductive generations. Before increasing model size or update count, inspect the traces and EOS/stop handling, measure where each truncation budget goes, and make the same-seed, same-case comparison fail closed when actions or forecasts cannot be verified. Keep the symbolic comparator labeled separately as a reference.

## E4 failure and source repair

Both E4 jobs reported the same error:

```text
Expected all tensors to be on the same device, but got index is on cpu,
different from other tensors on cuda:0 (when checking argument in
method wrapper_CUDA__index_select)
```

`_prove_architecture_on_handle` restored the model on each worker GPU, then sampled probe token IDs with `torch.randint` on its default CPU device. The embedding lookup therefore received CUDA weights and CPU indices. The probe now explicitly uses `next(model.parameters()).device`, and a parameterless model fails with a direct diagnostic. A regression test asserts that the probe sampler receives the model device and that the gate/segment proof still runs.

The E4 reservations in the old run are closed and cannot be retried in place. Re-run with a **fresh unique run ID**. Have the updated normal notebook generate or select a current bundle, validate it, and run source/build verification; do not assume the archived `d64...` bundle matches the current schema. Then qualify E0 on the exact fixed source before starting the full campaign. Use a fresh Kaggle allocation with the full 480-minute wall window; the failed run's 395-minute remainder is below the declared campaign window.

## Download and checkpoint limitation

These are results-only ZIPs. The result-pack contract intentionally excludes `.pt` checkpoint payloads, but the supplied archives embed an `artifact_manifest.json` that still says `complete: true` and lists eight payload files. None of those eight `.pt` files is inside the supplied ZIPs. The archive can be used to inspect results and logs; it **cannot** restore those checkpoints. If you need those weights, preserve or separately download the campaign's `K8-results/checkpoints/` payload directory from Kaggle before ending the session.

The packer in this branch now projects embedded artifact manifests and restore evidence to the files actually in a results ZIP, and marks omitted weights as non-restorable. This corrects future result downloads; it does not modify the four already-downloaded archives.

## Changes and verification on Gandiva

- Fixed E4's probe tensor device placement.
- Added an E4 regression test for model-device sampling.
- Corrected results-only ZIP metadata so embedded manifests do not claim excluded checkpoint weights are present; archive member paths are normalized across Windows and Linux.
- Verified: `python -B -m pytest -q -p no:cacheprovider -o addopts= tests/test_research_results_pack.py tests/test_research_k8_operational.py` — **56 passed, 12 subtests passed**.
- Repacked a temporary copy of the supplied results archive with the corrected packer: ZIP CRC valid, 104 manifest entries represented, all 8 checkpoint payloads explicitly omitted, and artifact/restore metadata marked non-restorable. The supplied ZIPs were not modified.

These checks are CPU-side regression tests. They do not substitute for rerunning E4 on Kaggle or for successful learning results. No new training run was started during this review.

## Current source readiness — 25 September 2026

The repaired Gandiva source was then checked against a newly prepared,
validated full bundle (`20c3d876136ee9023a085cc11ae67c3929a0c2520c830c647081c988a209cfbd`).
The source-bound, zero-update verifier passed all requirements F01–F24, all
seven test groups, and all seven production-interface exercises. It recorded
zero local optimizer updates and `ready_for_owner_experiment: true`. Its source
closure is `7f4a24d1b70639f5231cfec283ad0eb4a68feb977512ecdc58459998a70c0457`
at commit `c6c64c89eb568c44cc1deed683a089580ff29494`; it took 216.61 seconds.
The full machine-readable report is
[`../FINAL_K8/gandiva-post-e4-c6c64c89-20260925-pytest-temp/build_verification.json`](../FINAL_K8/gandiva-post-e4-c6c64c89-20260925-pytest-temp/build_verification.json).

One verifier attempt first hit an environment permission error in pytest's
default Windows temp directory. The affected F04 group passed 67 tests with a
writable temp root, and the full verifier passed on retry. Runtime gates G01–G04
still require live Kaggle E0 qualification on two T4s. This makes the repaired
implementation ready for the owner experiment; it does not guarantee that E0
will qualify, that training will succeed, or that the model has AGI capability.
