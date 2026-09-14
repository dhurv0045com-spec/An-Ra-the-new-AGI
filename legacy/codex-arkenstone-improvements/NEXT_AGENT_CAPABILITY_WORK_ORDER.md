# NEXT AGENT: deliver a visible capability improvement on Arkenstone

Packet: ARK-NEXT-001. Written 2026-09-12. This file is the assignment; prior conversation is not required.

## Owner intent and success

The owner wants a model that shows a meaningful new ability, not another report about faster infrastructure. Work toward intelligence learned from scratch. Deliver a runnable demonstration backed by real model outputs and matched measurements. Do not describe yourself or this project as having achieved AGI.

The immediate target is **robust non-arithmetic binding**: a small neural model reads three key=value facts and answers a query even when the facts are reordered. The owner should be able to see a baseline failure, the candidate's output on the same input, the correct answer, and the aggregate held-out results. Use the existing ARK-014 plan rather than inventing a disconnected benchmark.

This work order asks the next agent to implement and verify the next increment. It does not assert that it is already implemented or that a positive result is guaranteed. If the learned comparison is negative, preserve it and deliver the working demonstration plus the exact unresolved bottleneck.

## Starting point and authoritative reads

Start from the published `codex/arkenstone-improvements` branch in the same repository. Resolve its latest actual commit with Git; do not treat `933d4f3` as the final implementation commit. That SHA is the historical Arkenstone base. Record the resolved start commit and dirty-tree status before editing.

Read, in order:

1. Root AGENTS instructions if present, plus the owner's active instructions. Preserve unrelated work and branch boundaries.
2. [README.md](README.md) and [ARKENSTONE_CLARITY.md](ARKENSTONE_CLARITY.md).
3. [Initial integration handoff](docs/arkenstone/improvement_20260912/HANDOFF.md) and its linked validation evidence.
4. [ARK-014 scientific plan](experiments/ARK-014/PLAN.md).
5. [V6 campaign plan](experiments/COLAB/MASTER_DISCOVERY_V6_PLAN.md).
6. ARK-009 implementation/data construction and the imported V5 results under `experiments/COLAB/`; find the actual definitions with `rg` before reusing them.
7. Shared runtime in `experiments/COLAB/discovery_v6_common.py`, `run_discovery_v6.py`, and ARK-011/012/013 runners.

The existing 50 focused tests and CPU preflight passed. The final sampler microbenchmark was 2.85x faster, with exactly matching index streams; that is not a model-capability gain. No full ARK-012/013 training or CUDA qualification was run in the preceding pass. ARK-014 and the combined V6 scheduler remain unfinished.

## Execution order and ownership

Use one Luna executor for bounded implementation when delegation is useful; the primary agent owns scientific decisions and integration. This is the owner's model preference. Do not spawn a large agent team or a separate user-owned task without an explicit request. Delegate with this packet ID, prerequisite paths, exclusive owned files, CPU allowance and evidence location.

Use a fresh branch/worktree if another task is active. Never edit `.codex-worktrees`, other branches' working trees, or existing evidence snapshots. Do not inspect secrets or rewrite historical experimental results.

Owned paths for this packet:

- New `experiments/ARK-014/run_ark014.py` and narrowly scoped supporting modules under that directory.
- New `tests/test_ark014.py` and `tests/test_ark014_demo.py`.
- New `experiments/ARK-014/demo.py` and a small local report template if needed.
- New `docs/arkenstone/next_capability/` work notes, protocol clarifications and handoff.
- New run directories under `artifacts/arkenstone/ark014/`.
- Small integration edits to `experiments/COLAB/run_discovery_v6.py`, shared runtime, and root README only when required for this packet and reviewed by the primary.

Excluded from this packet: architecture-wide rewrites, production Cymek scheduling, unrelated V5 modules, historical PLAN/result files, a complete V6 campaign scheduler, and autonomous code modification. Follow-on work is listed below; it is not a reason to delay the visible demonstration.

## Step 1: qualify data and objective before training

Implement the frozen ARK-014 contract:

- Six keys, six values, three facts per example; task seed 4242.
- Split by whole semantic fact-set before query/order expansion: 400 train and 100 held-out fact-sets; held-out splits into 50 CONTROL and 50 SEALED by the plan's deterministic hash rule.
- All three queries per fact-set. Report CANONICAL, ORDER_ONLY, QUERY_ONLY and QUERY_ORDER separately.
- Deterministic training-order augmentation is a pure function of acquisition seed, optimizer step, batch position and semantic example ID. A changed order must preserve query/answer semantics.
- Assert zero fact-set overlap across train, CONTROL and SEALED. Record membership, semantic identities, answer checks and order hashes.

Audit the inherited loss before reuse: it supervises the answer BOS token and includes padding for mixed answer lengths. Do not quietly change the historical objective. If ARK-014 requires a correction, write a dated prospective protocol clarification before executing the affected comparison, define the exact mask, and apply it identically to baseline and candidate. Give the new objective its own version/hash. Preserve old receipts.

Acceptance: deterministic repeated generation, semantic equivalence of all order variants, no split leakage, and focused tests that fail if a fact-set crosses a boundary or augmentation changes an answer.

## Step 2: implement the real matched experiment

Follow ARK-014's exact architecture and optimizer contract. All learned weights start randomly initialized. Declare CompactVocab, task generators, symbolic answer computation used for scoring, and any external-tool access as fixed priors. No pretrained model may answer for the trained network.

Use matched seed 2201 for CANONICAL_TRAIN and ORDER_AUGMENTED, with equal update budgets and matched semantic examples. Measure actual supervised positions, examples, steps and elapsed time. Keep semantic minibatch order separate from fact-order augmentation in the receipt.

Qualification uses CONTROL only: three consecutive checks with CANONICAL >=0.90, ORDER_ONLY >=0.85, QUERY_ORDER >=0.85. SEALED never selects checkpoints, changes LR, stops training or chooses which regime advances. If both regimes qualify, fix the plan's preference for ORDER_AUGMENTED before looking at sealed scores.

Implement matched HIGH/LOW retention forks at 1e-3 and 1e-5 for frozen order seeds 7701, 7702 and 7703, 6,000 updates each. Preserve each completed arm immediately. Do not count a partial pair, duplicate identity, mismatched horizon, failed seal at fork or truncated arm as a successful comparison. Report the plan's paired retention endpoint and uncertainty/limitations, including acquisition n=1.

Acceptance: a CPU fake-runtime test demonstrates the controller ignores changed SEALED values; identical snapshot/order/LR reproduces the next update; completed evidence survives an injected later-arm exception; incomplete experiments receive an explicit incomplete status. Use the existing runtime's isolated receipts, source snapshots, monotonic budget checks and failure packaging.

## Step 3: make the improvement visible

Provide one documented local command, intended interface:

```text
python experiments/ARK-014/demo.py --run-dir <completed-run-directory> --output <new-demo-directory>
```

The command must load the actual recorded model checkpoint(s), verify their source/data identities, and produce a small local HTML report plus a machine-readable examples JSON. It must work without a hosted service or a paid API. Use a simple template; do not spend this packet building a dashboard framework.

Show:

- Canonical facts, reordered facts and query.
- Baseline prediction, candidate prediction and ground-truth answer on the same examples.
- Aggregate exact accuracy for all four diagnostics, denominators, seed, update/token budgets, source checkpoint identity and whether the run completed.
- Examples selected by a frozen rule independent of candidate success, such as the first 12 hash-ranked demonstration fact-sets. Do not choose only flattering successes.
- A separate clearly labeled failure section when errors remain. Do not replace failed model answers with symbolic solver output.

The illustrative examples must come from a distinct deterministic demo split or be labeled SEALED measurement examples used only after the frozen run; never feed them back into model selection. If there is no trained checkpoint, the command must report that fact and refuse to fabricate a before/after comparison. An untrained plumbing preview is allowed only when prominently labeled as such.

Acceptance: a reviewer can run the command, inspect real predictions and verify that displayed aggregate metrics agree with receipts. Tests reject missing/mismatched checkpoints and unsupported success badges.

## Step 4: execute only the authorized compute box

Existing authorization covers implementation and bounded local CPU diagnostics. Keep individual correctness/preflight commands within two minutes; stop on a concrete invariant failure instead of expanding the test sweep. The prior local runtime was Python 3.11.15 with PyTorch 2.14.0+cpu; verify the next environment rather than assuming it is identical.

Long training, paid compute and accelerator time are **not authorized by this document alone**. Before a full learned comparison, check the active owner's actual authorization, device and remaining allowance. The historical roughly 100 TPU-hours/week is a planning estimate, not a live entitlement. GPU execution must first pass the new CUDA preflight. Record the exact box in a run manifest.

If a learned run is not authorized or hardware is unavailable, finish the runner, tests, demo command and a pinned reproducible launch command. Report the missing run plainly; do not fill the demo with invented outputs. This is a valid implementation handoff, but not completion of the owner's visible learned-capability target.

The frozen campaign has a 40-minute minimum entry reserve for ARK-014 and a 20-minute reserve before continuation. Distinguish entry gates from between-block gates. The full V6 plan's 240-minute box does not grant permission to launch it.

## Evidence and measurable success

Use a new immutable run ID for each change of code, objective, seed, budget or data. Keep weights/optimizer files and large data out of Git. Store model checkpoints locally or in the actually authorized durable location; commit compact manifests and outcomes. A model/demo pairing must be tied to checkpoint hashes, not just filenames.

The primary success is the frozen `ORDER_ROBUSTNESS_REPAIRED` criterion or the plan's stronger qualified transfer screen, supported by matched full-run evidence. A positive screen with one acquisition seed is not independent replication or AGI. If 'materially improves' needs an operational definition, freeze it prospectively and show both raw deltas and the decision rule; do not invent a threshold after seeing results.

Report engineering and science separately:

| Deliverable | Required evidence |
|---|---|
| Implemented ARK-014 runner | Exact owned files and focused checks |
| Reproducible task and firewall | Full membership/semantic hashes and adversarial split tests |
| Training correctness | Actual forward/backward/update and snapshot replay on the claimed device |
| Visible before/after demo | Real checkpoint predictions, frozen examples and aggregate metrics |
| Claimed capability gain | Matched completed arms, frozen qualification criteria and all failures |
| Runtime improvement claim | End-to-end timed baseline/candidate comparison, separate from sampler setup |

Do not substitute more documentation, more tests, lower training loss or a larger model for the learned-capability criterion. Do not promise 10x: define what would be tenfold, measure it and report the actual result.

## Finish and hand off

Write `docs/arkenstone/next_capability/HANDOFF.md` containing: starting/final source identities; owned paths; exact commands; each acceptance criterion PASS/FAIL/NOT RUN; device and measured time; run IDs and checkpoint hashes; before/after metrics with denominators; failures and limitations; the demo command/output location; and one recommended next experiment.

Update README with the visible demo command and the honest current status. Preserve the existing 50-check validation as historical evidence; run the changed components' checks, not the entire repository repeatedly. Publishing future agent work follows that future session's actual Git authorization. The current owner authorized publishing the already-completed hardening and this work order.

## Later packets, after the visible result

1. Durable checkpoint/restart across process interruption, including optimizer, RNG, exact data cursor and objective identity.
2. Fresh multi-seed non-arithmetic replication if ARK-014 qualifies; otherwise diagnose order sensitivity using the separated diagnostics.
3. A separately preregistered active-information task where a learned policy chooses useful queries, compared with random, fixed-query and privileged-oracle controls. This would test investigation, which the current arithmetic/LR work does not establish.
4. Integrate qualified ARK-011/012/013/014 runners under the frozen V6 order and actual authorized budget.

Start with Steps 1-3. The next useful deliverable is a model behavior a person can inspect, with trustworthy measurements behind it.
