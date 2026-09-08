# CYR-GPU-006 — PREEXECUTION AUDIT

CYR-GPU-006 supersedes CYR-GPU-005 before any GPU execution. The 005 scientific question remains worthwhile, but its Colab harness violated its own frozen execution contract.

| ID | Finding in CYR-GPU-005 | Severity | CYR-GPU-006 correction |
|---|---|---:|---|
| A1 | Cell 1 called `run_campaign(..., mode="full")` without the CUDA device; the runner then defaulted to CPU after merely checking that CUDA existed. | FATAL | Full runner requires CUDA and defaults to `torch.device("cuda")`; notebook passes the CUDA device explicitly. |
| A2 | The orchestrator broke after the first G90 parent, so at most one independent parent entered the fork tournament. | FATAL SCIENCE | All three frozen parent seeds 707/808/909 are attempted; no first-G90 break exists. |
| A3 | Decision logic could call a winner from one parent. | FATAL SCIENCE | A win requires at least two independent contract-valid parents plus replicated paired margins; one parent is always INCONCLUSIVE. |
| A4 | Cell-0 calibration/resolution was discarded; Cell 1 recalibrated independently. | FATAL REPRODUCIBILITY | Cell 0 writes calibrations + resolved plan; Cell 1 loads and passes those exact objects. The scientific runner never recalibrates or re-resolves. |
| A5 | Calibration omitted `optimizer.step()`. | HIGH | Candidate calibration executes the real ProductionTrainingBackend update, including optimizer mutation. |
| A6 | Runtime model ignored expensive candidate-free generation. | HIGH | Calibration benchmarks batched candidate-free generation; resolver includes generation throughput in prospective wall-time estimation. |
| A7 | Candidate-free evaluation was sample-by-sample, token-by-token. | HIGH | Evaluator buckets equal prompt lengths and generates batches in parallel. |
| A8 | Parent displacement tensor was CPU while live model was CUDA, causing a CUDA/CPU subtraction error after the device bug was fixed. | FATAL RUNTIME | Parent vector is constructed from the restored live model directly on the execution device. |
| A9 | `RUN OR RESUME` did not resume campaign work. | HIGH | Stage-level durable resume skips completed acquisitions and completed arms; incomplete arms restart cleanly from the frozen parent/tail. Notebook stores state on Google Drive. |
| A10 | Failure packaging was only on the successful return path. | HIGH | Outer exception/finally path writes `FAILURE.json`, partial evidence bundle and campaign receipt before re-raising. |
| A11 | Parent-equivalence receipt claimed RNG/cursor checks it did not actually perform. | MEDIUM | 006 claims exactly what it verifies: model bytes, optimizer bytes and checkpoint counters. Every arm also begins with the same post-restore RNG seed. |
| A12 | Red team marked `update_norms_measured=true` without direct update-norm measurement. | MEDIUM | 006 records parameter displacement, integrated pre/post-clip gradient norms and Adam moments and explicitly sets direct update-norm measurement false. |
| A13 | Later arms could systematically receive worse timebox conditions. | MEDIUM | Arm order rotates deterministically by parent; identical token targets remain the exposure currency. |
| A14 | XLA previously all-reduced accumulated gradients inside each microstep. | FATAL PRODUCTION | Already repaired before this freeze: one gradient SUM collective now occurs after all accumulation microsteps; CPU oracle remains local evidence only. |

No CYR-GPU-005 GPU evidence exists. The experiment identity is retired rather than silently patched after preregistration.
