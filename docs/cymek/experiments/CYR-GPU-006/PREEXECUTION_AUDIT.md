# CYR-GPU-006 — PREEXECUTION AUDIT

CYR-GPU-006 supersedes CYR-GPU-005 before any GPU execution. The 005 scientific question remained worthwhile, but its Colab harness violated its own frozen execution contract. CYR-GPU-006 itself was still unfrozen when the live Arkenstone Discovery V6 evidence was re-audited, so its transfer design was revised **before preregistration and before execution** rather than creating post-outcome flexibility.

Arkenstone read-only authority used for the revision: `6acd9dcbdd28d00f387ffcd004253a813aca4b66`; validated Discovery V6 ZIP SHA256 `1ec3224075de49b080e229111e4c4f897430c3eca888c487116aff07a65c8d15`.

| ID | Finding | Severity | CYR-GPU-006 correction / status |
|---|---|---:|---|
| A1 | CYR-005 Cell 1 called the full runner without a CUDA device; the runner could default to CPU after merely checking CUDA existed. | FATAL | Full runner requires CUDA and notebook passes `device=DEVICE` explicitly. |
| A2 | CYR-005 stopped after the first G90 parent. | FATAL SCIENCE | All three frozen parent seeds 707/808/909 are attempted; no first-G90 break exists. |
| A3 | CYR-005 could call a winner from one parent. | FATAL SCIENCE | A retention win requires >=2 independent contract-valid parents and replicated paired margins. |
| A4 | CYR-005 discarded Cell-0 calibration/resolution and recalibrated independently. | FATAL REPRODUCIBILITY | Cell 0 writes calibration + resolved plan; Cell 1 passes those exact objects. Scientific runner never re-resolves. |
| A5 | CYR-005 calibration omitted `optimizer.step()`. | HIGH | Candidate calibration executes the real ProductionTrainingBackend optimizer mutation. |
| A6 | Runtime model ignored expensive candidate-free generation. | HIGH | Calibration benchmarks batched generation; resolver includes measured generation throughput. |
| A7 | Candidate-free evaluation was sample-by-sample. | HIGH | Equal-prompt-length batches generate in parallel. |
| A8 | Parent displacement could subtract a CUDA tensor from a CPU tensor. | FATAL RUNTIME | Parent vector is constructed from the restored live model on the execution device. |
| A9 | `RUN OR RESUME` did not actually resume campaign work. | HIGH | Drive-backed stage-level resume skips completed acquisition/retention/transfer states; incomplete stages restart from immutable source state. |
| A10 | Failure packaging existed only on success path. | HIGH | Outer `finally` packages partial evidence and failure receipt before re-raising. |
| A11 | Parent-equivalence receipt overstated RNG/cursor checks. | MEDIUM | Current receipt claims only model bytes + optimizer bytes + checkpoint counters; future-tail equality is a separate receipt. |
| A12 | Red team claimed direct update norms without measuring them. | MEDIUM | Current red team explicitly says direct update norm is not measured; reports displacement, gradient norms and moments instead. |
| A13 | Later retention arms could systematically get worse wall conditions. | MEDIUM | Retention-arm order rotates by parent; transfer candidate/comparator order counterbalances across parents. |
| A14 | XLA previously all-reduced accumulated gradients inside every microstep. | FATAL PRODUCTION | Repaired: collective occurs once after accumulation boundary; CPU oracle is local math evidence only, not TPU certification. |
| A15 | Initial CYR-006 transfer compared a pre-continuation G90 parent with an older post-continuation candidate. Policy effect was confounded with additional same-task age/exposure. | FATAL SCIENCE | Final transfer compares **HYSTERETIC_HIGH_LOW vs LOW_CONTINUE** after equal continuation-token exposure from the same acquired parent. |
| A16 | Initial transfer candidate was chosen only if it won the same CYR arithmetic experiment, creating a post-hoc second-family selection path. | HIGH SCIENCE | Transfer candidate HYSTERETIC_HIGH_LOW and comparator LOW_CONTINUE are fixed prospectively from pre-CYR Arkenstone evidence. Transfer executes independently of arithmetic winner if source-state gates are met. |
| A17 | ARK-013 shows LOW same-task protection does not solve long no-replay cross-task interference. | HIGH SCIENCE | Final transfer adds fixed old-T2 replay while a new robust binding skill is learned and measures old T2 concurrently. Replay fraction is reported in actual tokens. |
| A18 | ARK-014 shows canonical-only non-arithmetic binding can hide fact-order brittleness. | HIGH SCIENCE | Binding training is deterministically order/query augmented; CANONICAL, QUERY_ONLY, ORDER_ONLY and QUERY+ORDER are measured separately. |
| A19 | ARK-014 HIGH-vs-LOW non-arithmetic retention screen had zero failures, so simply repeating that screen is low-information. | MEDIUM/HIGH | Final second-family endpoint is **plasticity of equal-age retained states**, not another forced retention-failure screen. Zero acquisition/event rate is honestly INCONCLUSIVE. |
| A20 | Arkenstone bottleneck graph is stale and still describes ARK-011 as unexecuted. | PROCESS | Raw ARK-011..014 results + validated Discovery V6 bundle + live experiment log are treated as authority; stale summary is explicitly excluded. |
| A21 | Full Cymek CI shallow checkout made exact historical receipt verification fail with `fatal: bad object`. | CI INFRA | Cymek PR workflow now checks out full history (`fetch-depth: 0`). Superseded experiment receipts are being separated from the current exact-head closure semantics before final freeze. |
| A22 | ESOES PR workflow ran repository tests without installing torch/pytest/tokenizers, creating irrelevant red CI. | CI INFRA | Readiness-branch PR workflow now installs its declared regression dependencies and fetches full history. No ESOES branch is modified. |

## Current pre-freeze status

No CYR-GPU-005 or CYR-GPU-006 scientific GPU evidence exists. CYR-GPU-005 remains retired. CYR-GPU-006 remains an **executable candidate**, not a preregistered experiment, until the exact final executable SHA passes the dedicated CYR contract suite, the useful full Cymek suite, freeze/checkout simulation and the final manual red team. Only then may Commit B add the immutable preregistration and readiness receipt.
