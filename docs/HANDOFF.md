# ELENDRIL V5 handoff

Continued `Elendril` from `f516e0bdb8f9137532b35c0c23299f7f9dbdf185`; no reset, bulk branch merge or historical checkpoint overwrite. The working V4 source and paused V3 program are remotely preserved as `elendril-v4-working`. The pre-V3 archive remains `elendril-pre-v3`.

Implemented and pushed source milestones:

| Commit | Change |
| --- | --- |
| `1f630365215166554e4e654f4e114e02700ac54f` | Versioned legacy/UTF8 representation, shared neural backend, action worlds, exact-denominator trainer and decoded recovery. |
| `ade2d5bf69abeb6b07daca162cf1d275a5f91bee` | Separate-process publisher race, Unicode training/resume, scaler/migration corruption tests and staged legacy tensor decoding. |
| `afbf0c45e6fdddc5c6e5eebfc99a3622aeca2141` | Checkpoint CLI, gated action cloud runner, raw-action replay, scientific gates and archive cleanup. |
| `78ce7def7c1b7bdc4a73a3e1bb7022485880ceab` | Canary CPU and CUDA allocation telemetry. |
| `1d0c6aedbdccd05dbc85e57a66fbfe433e75eb12` | Fix undeclared NumPy fingerprint dependency; preserve exact byte identity. |

The notebook/evidence publication follows these source commits. Both active notebooks pin **1d0c6aedbdccd05dbc85e57a66fbfe433e75eb12**, not a moving branch. [Colab](../notebooks/ELENDRIL_ACTION_COLAB_T4.ipynb) · [Kaggle](../notebooks/ELENDRIL_ACTION_KAGGLE_T4X2.ipynb)

The active code adds `representation.py`, `backend.py`, `foundation_training.py`, `action_learning.py` and `checkpoint_tools.py`. Legacy IDs and manifest hashes are unchanged. The UTF8 path has a separate vocabulary/config identity and explicit row migration with optimizer reset. The original frozen policy and new trained-action policy use the same backend/episode contracts. The reference trainer retains old sampling/objective semantics while adding pre-publication decoding. The new trainer supports variable lengths, exact weighted accumulation, finite/ownership checks, explicit curricula/schedules/precision and typed stage migration. Recovery protects PREVIOUS, rejects stale publishers and validates actual tensor shapes/dtypes, optimizer/scaler counters, RNG and cursor before publication.

Final Windows qualification passed **166 tests plus seven subtests in 42.03 seconds**. The new notebook/operator subset passed ten tests including schema, all default-disabled cells with writes/subprocesses forbidden, exact source/generator bytes, physical UUID gates, nonfinite forecasts, recovery challenges and independent replay. Repaired runtime source passed [Linux CPU CI](https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI/actions/runs/42366390091), 164 tests plus seven subtests before the two publication-only tests. No test was disabled or weakened to hide a failure. [Qualification details](FOUNDATION_QUALIFICATION.md)

Observed results and costs:

- Tiny correctness control: four prompts, eight updates, QUERY+EOS 4/4 on the training prompts; loss 3.8082435→0.7462860. Measured update/save/generation wall 0.7044 seconds, zero GPU. The earlier LR .01 attempt failed the unchanged 50% loss-reduction threshold; it remains recorded. This is overfit learnability, not action triage.
- Ten validation worlds: engineered fixed 10/10 with four requests, always-query 10/10 with eight, no-query 6/10. Full raw traces and shortcut/oracle controls are retained. The oracle is explicitly privileged.
- Frozen V4 replay at repaired source: route-memory 4/4, no-memory 0/4, same-token context 1/4, revision 0/4 with zero QUERY. Forty calls/912 input positions. Before/after weights remain `8734cfb9957f510a703c0d04e5f68a305e0af9552238c28f178e848dc9c2e863`.
- Native GQA matched reference forward/gradients in the packed probe, but CPU medians were 3.3545 vs 2.8698 ms (1.169× slower); reference stays default. No GPU speed/memory result is claimed.
- Historical custody: 78 evidence hashes, 63 receipts/519 snapshots, 237 manifests/1,659 components reverified. The current installed wheel passed Unicode/control/inspection checks; the rebuilt NumPy-free wheel additionally passed migration and the exact original-checkpoint fingerprint with Tensor.numpy forbidden. No training occurred during wheel qualification.

[Raw receipts and SHA manifest](../artifacts/elendril/foundation-v5/manifest.json) · [Result chart](../artifacts/elendril/foundation-v5/qualification.png)

Cleanup removed the paused V3 notebook wrappers/generator and three superseded planning/trace documents from the active tree, preserving exact Git objects and their disabled-cell tests. [Removal manifest](ARCHIVED_PROGRAMS.json). The active tree has 204 tracked files, 16 docs and two V5 notebooks, about 15.5 MB including preserved raw evidence; the original pre-V3 tree had 861 files, 228 docs and ten notebooks. Shared low-level cloud verification remains because the action runner imports it. Other branch worktrees and external weights were untouched.

Operator commands:

~~~text
python -m pytest -q
python -m tools.verify_elendril_evidence --checkpoint-base C:/Users/ankit/Downloads
python -m elendril inspect-checkpoint --root <root> --lineage <lineage>
python -m elendril verify-checkpoint --root <root> --lineage <lineage>
python -m elendril resume-dry-run --root <root> --lineage <action-lineage>
python -m elendril compare-checkpoints --root <root> --lineage <lineage> --sha <old> --other-sha <new>
~~~

The tiny canary checkpoint `2a4d108fabaf6b0b8515c94042eabfcf88fa201dc8ff976b0d0db0c35a095eaa` is external at `C:/Users/ankit/Downloads/Elendril-V5-foundation-canary`. Original ELD payloads remain under their sibling custody roots. No giant weights are in Git. Use a separate detached worktree at an exact source SHA for historical reproduction; never reinterpret old weights with UTF8 or overwrite the accepted lineage.

Launch decision: conditional GO for owner device/correctness canaries only. Research campaigns remain **BLOCKED_PENDING_GPU**. Colab plans CE/weighted CE/uncertain-world curriculum × two seeds × 448 updates; Kaggle uses the same arms × five seeds × 896 updates on separate UUIDs and a distinct world seed/split. Caps are 105+15 and 315+30 minutes, not measured runtime promises. All launch switches are OFF. Admission requires actual formation, all-arm timings, same-device exact resume and a genuinely different VM session reattaching durable outputs. Kaggle also needs complete rescored Colab evidence and owner review. [Operations](OPERATIONS.md) · [Frozen decision rules](ALGORITHM_COMPARISONS.md)

Limitations remain substantive. No GPU campaign, real-text language learning, learned action triage, calibrated metacognition, new-skill retention, forward transfer or self-improvement was established. Fresh action weights intentionally test formation and do not replace the working two-read checkpoint. Byte expansion changes context cost and softmax competition; it has no qualified whole-copy-head migration. The verifier is in-process, not a sealed execution sandbox. Older S5 checkpoints remain outside reviewed custody. The next decisive experiment is full multi-action formation under the frozen accuracy/request-cost gates, then a separately qualified warm-start or replay retention comparison. [Next experiment](NEXT_COGNITIVE_EXPERIMENT.md)
