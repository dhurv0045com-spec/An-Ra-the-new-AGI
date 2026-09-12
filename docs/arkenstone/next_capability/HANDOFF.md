# ARK-014 handoff — visible capability increment delivered

Date: 2026-09-13. Packet: ARK-NEXT-001. Executor: primary agent (no delegation).

## 1. Starting and final source identities

- Starting branch/commit: `codex/arkenstone-improvements` at
  `9c09d9d682ce3b2cd60f8e5573c84e04922ec8a8` ("Improve Arkenstone runtime integrity and define
  next capability work order"), clean working tree, isolated worktree
  `C:/Users/ankit/AppData/Local/Temp/arkenstone-improvements-20260912`. The prior handoff's base
  `933d4f3` was treated as historical, per the work order.
- Final commit of this packet: see `git log` on the branch (this file is part of it).
- Unrelated work preserved: the `Arkenstone` worktree (branch `Arkenstone`, modified
  `tests/test_ark020_v3.py`, untracked `_rebuild_v4.py`) and the `BRAMASTRA` checkout were not
  touched. No `.codex-worktrees` paths, no historical evidence snapshots, no secrets.
- Executed-source identity of the recorded run is bound inside the run bundle
  (`sources/experiments/ARK-014/*.py` + `source_sha256` in every receipt):
  - executed `run_ark014.py`: `afe41a52e3ce3c9e84f255c7fa6ccfde104104d8d86e3c520f56e0bbd73a5c7d`
  - executed `ark014_binding.py`: `ca2c230f8b4386feeb28b08678c30ed2d51758d7f56eb7b7b6603a6906f40263`
- The committed source evolved after the run in two review passes, both disclosed here:
  1. Commit 1 (with the run): added the explicit `order_robustness_repaired` summary field and
     the demo's display of it — reporting only.
  2. Commit 2 (architectural cleanup, static-only, no compute executed): removed the redundant
     writer/task construction in `main()`, the duplicate `run_ark001` module load and local
     sha helper in `demo.py`, the unused `sha_json`/`json` imports, the fragile bytecode hash
     in `AUGMENTATION_SPEC`, and the input mutation in `_materially_improves` (now pure);
     unified interrupted-arm evidence (any exception now writes an explicit
     `INCOMPLETE_*` receipt, not only budget exhaustion); replaced the JSON-based
     snapshot-diff in the preflight with full structural snapshot equality. Permutation
     indices, prompts, streams, thresholds and training math are unchanged — the executed
     binding's augmentation derivation (`ca2c230f…`) computes byte-identical results.
  - Final committed identities (commit 2):
    `run_ark014.py` `5850bed96260a224eb0234af5b9bd32d8229864f91d4bb3874147a756fa4377e`,
    `ark014_binding.py` `bddafe447c0ee7b2d875121aaa0233302cf28ccae07ed2a2b887c421d4c371f8`,
    `demo.py` `f5e0befef1535c11078dbb27bee62ee32dc9309185ec33426be6462e9c6172c4`.
  - Verification status of commit 2: compile-checked and statically reviewed only; per owner
    instruction no local CPU/GPU execution occurred in this pass. The 35 focused tests and the
    runtime suites must be re-run on the next compute-authorized session before any new run.
  - For run `ark014-cuda-2201-03` the repair criterion is derivable from the receipt's
    `summary.regime_qualification` and is restated in §5.

## 2. Owned paths

- New: `experiments/ARK-014/ark014_binding.py`, `experiments/ARK-014/run_ark014.py`,
  `experiments/ARK-014/demo.py`, `tests/test_ark014.py`, `tests/test_ark014_demo.py`,
  `docs/arkenstone/next_capability/{PROTOCOL_CLARIFICATIONS,WORK_NOTES,HANDOFF}.md`,
  `artifacts/arkenstone/ark014/ark014-cuda-2201-03/` (receipts + ZIP; `checkpoints/*.pt` stay
  local, excluded from Git by the root `*.pt` ignore),
  `artifacts/arkenstone/ark014/demo-ark014-cuda-2201-03/` (report.html + examples.json, committed).
- Small integration edits: `experiments/COLAB/discovery_v6_common.py` (backward-compatible
  `extra_source_paths` parameter on `ReceiptWriter`), `experiments/COLAB/run_discovery_v6.py`
  (`--campaign ARK-014`, 40-minute entry reserve), `README.md`.

## 3. Exact commands (reproduction)

```powershell
# focused checks (35 total)
python -m unittest tests.test_ark014 tests.test_ark014_demo -v
# existing runtime/campaign checks still pass after the ReceiptWriter extension
python -m unittest discover -s tests -p 'test_discovery_v6*.py' -v

# bounded preflight on the exact device (CPU or CUDA), ~2 minutes
python experiments/ARK-014/run_ark014.py --device cpu --budget-minutes 5 \
    --output-dir <new-dir> --preflight-only

# the frozen matched experiment (CUDA; RTX 4050 used here)
python experiments/ARK-014/run_ark014.py --device cuda --budget-minutes 150 \
    --max-gpu-duty 1.0 --output-dir artifacts/arkenstone/ark014/<new-run-id>

# the visible before/after demo from a completed run (CPU; no hosted service)
python experiments/ARK-014/demo.py \
    --run-dir artifacts/arkenstone/ark014/ark014-cuda-2201-03 \
    --output <new-demo-dir>
```

`--max-gpu-duty 0.8` (the default) paces the GPU to ≤80% utilization per the owner's
instruction; the recorded run used 1.0 after the owner raised the cap. The cap is wall-clock
pacing only and cannot change any scientific quantity.

## 4. Device and measured time

- Device: `cuda` — NVIDIA GeForce RTX 4050 Laptop GPU (6 GB), driver 591.62;
  torch 2.14.0+cu126, Python 3.11.15, Windows. CUDA preflight: **PASS** (run
  `ark014-cuda-preflight-01`; matched fork next-update equality, augmentation purity,
  controller sanity, task anchors all verified on the GPU).
- Frozen campaign `ark014-cuda-2201-03`: 41.3 minutes wall (budget 150), including all
  evaluations, checkpoints and receipts. Measured throughput ≈ 27–33 optimizer steps/s for the
  Micro-128/batch-64 frozen configuration.
- CPU reference: ~7 steps/s (8 threads) — the same frozen box would be several hours on CPU,
  which is why the GPU was used after the availability check the owner requested.
- Two earlier starts were abandoned and deleted (no evidence value): `ark014-cuda-2201-01`
  (stopped to introduce the 80% duty cap) and `ark014-cuda-2201-02` (stopped when the owner
  raised the cap). No results were reported from either.

## 5. Result of the frozen learned comparison (run `ark014-cuda-2201-03`)

Receipt SHA-256: `28f5354d61f9a51c4324fe7228d207618f7ea26d94419c2415170ae76f63f5b8`.
Scale: `PREREGISTERED_FROZEN` (24,000-step acquisition box, 6,000-step retention forks).

**Primary criterion ORDER_ROBUSTNESS_REPAIRED: MET.** ORDER_AUGMENTED qualified on
BIND_CONTROL at step 2000 (onset 1600, 3 consecutive evals over thresholds) while the matched
baseline CANONICAL_TRAIN never qualified in its full 24,000-step budget.

Matched arms (same seed 2201, identical initialization and semantic minibatch stream; the
baseline's 24k-batch stream hash extends the candidate's 2k-batch prefix by construction):

| Arm | Status | Steps | Supervised positions | BIND_CONTROL at last eval |
|---|---|---|---|---|
| CANONICAL_TRAIN | NO_QUALIFICATION | 24,000 | 4,608,000 | CANONICAL 1.000, ORDER_ONLY 0.380, QUERY_ONLY 1.000, QUERY_ORDER 0.380 |
| ORDER_AUGMENTED | QUALIFIED (step 2000) | 2,000 | 384,000 | CANONICAL 0.987, ORDER_ONLY 0.987, QUERY_ONLY 0.987, QUERY_ORDER 0.987 |

BIND_SEALED measured once at qualification (analysis only, n=150 per diagnostic): candidate
CANONICAL 0.987, ORDER_ONLY 0.993, QUERY_ONLY 0.987, QUERY_ORDER 0.993. Sealed values were
measured strictly after the qualification decision and checkpoint snapshot; the controller
never reads them (demonstrated by test with mutated sealed streams).

**Retention screen: INCONCLUSIVE (zero events).** All three frozen orders (7701, 7702, 7703)
forked from the identical qualified snapshot; both LRs completed the full 6,000 steps each
(6/6 arms COMPLETED, all sealed-qualified at fork). Sealed qualification failed **zero** times
in both HIGH (1e-3) and LOW (1e-5) arms — final sealed diagnostics 1.000/1.000/1.000 in all six
arms. Paired failure counts HIGH=0, LOW=0, risk difference 0.0, no discordance. With no
retention failure events there is no directional LR-protection evidence; this is the plan's
`ROBUST_BINDING_ACQUIRED_BUT_TRANSFER_INCONCLUSIVE` branch, recorded as the run's overall
verdict alongside the met repair criterion. Interpretation is bounded: the acquisition box may
simply be too short for high-LR decay to appear (the ARK-011 arithmetic phenomenon needed
specific collapse dynamics), so "LOW did not protect" is NOT supported either.

**Demo aggregates recomputed from the real checkpoints** (final baseline checkpoint at 24,000
steps vs candidate qualification checkpoint at 2,000 steps; n=150 per cell; verified against
receipt values — any mismatch would have refused the demo):

| BIND_SEALED diagnostic | Baseline (CANONICAL_TRAIN) | Candidate (ORDER_AUGMENTED) |
|---|---|---|
| CANONICAL | 1.0000 | 0.9867 |
| ORDER_ONLY | **0.2867** | **0.9933** |
| QUERY_ONLY | 1.0000 | 0.9867 |
| QUERY_ORDER | **0.2867** | **0.9933** |

On the 144 displayed demonstration rows (12 hash-ranked SEALED fact-sets × 4 diagnostics × 3
queries, frozen outcome-independent selection): baseline wrong on 50, candidate wrong on 4.
Failures are shown verbatim in a dedicated section; the symbolic solver never replaces a model
answer.

## 6. Acceptance criteria (work order steps 1–4)

| Criterion | Result | Evidence |
|---|---|---|
| Step 1: deterministic task, zero split leakage, semantic equivalence of order variants | PASS | `tests/test_ark014.py::TaskContractTests` (7 checks incl. ARK-009 anchor reproduction, 400/50/50 membership, per-row symbolic-answer equality, variable-isolation audits); manifest hashes in `ARK-014_TASK_MANIFEST.json` |
| Step 1: loss audited before reuse; no quiet objective change | PASS | `PROTOCOL_CLARIFICATIONS.md` §1 (dated, pre-execution); `ObjectiveAuditTests` (3 supervised positions/row; no padding heterogeneity possible); historical `loss_and_positions` reused unchanged |
| Step 2: matched arms, equal budgets, matched semantic examples | PASS | Same seed/init/generator; augmentation is a pure function consuming no RNG (`AugmentationTests`); receipt records per-arm minibatch stream hash, augmentation stream hash and permutation histogram (near-uniform: 21,063–21,497 per permutation) |
| Step 2: controller qualification on CONTROL only; sealed measurement-only | PASS | `ControllerDisciplineTests` (identical decision streams under three wildly different sealed streams); frozen policy + material-improvement rule published pre-run in `ARK-014_FROZEN_POLICY.json` |
| Step 2: fork replay, snapshot immutability, evidence survival, incomplete statuses | PASS | `MatchedUpdateTests` (same snapshot/order/LR → identical next-update hash; different LR → different); `EvidenceSurvivalTests` (4 completed arms survive an injected 5th-arm failure; `INCOMPLETE_BUDGET_STOP` status with checkpoint); real run: all 6 retention arms completed |
| Step 3: demo shows real predictions, aggregates agree with receipts, frozen examples, failure section | PASS | `artifacts/arkenstone/ark014/demo-ark014-cuda-2201-03/` (report.html + examples.json); `test_ark014_demo` (10 checks incl. missing/mismatched checkpoint refusal, task-manifest tamper refusal, unsupported-badge rejection, failure display) |
| Step 3: no fabricated outputs; refuse without checkpoints | PASS | `DemoRefused` paths: missing RESULT file, missing checkpoint, hash mismatch, manifest drift/tamper, unsupported QUALIFIED claim |
| Step 4: compute box recorded | PASS | `compute_box` in `ARK-014_RESULT.json` (device, GPU name, duty cap, budget, authorization note); CUDA preflight receipt; no paid compute |
| Claimed capability gain | MET (repair) / INCONCLUSIVE (transfer) | §5 above; acquisition seed n=1, not replication; no AGI or universal claim made or implied |

Honest limitations: one acquisition seed (n=1) — a positive repair result is not independent
replication; retention conclusion is "no events observed in 6,000 steps at either LR", not
"LR does not matter"; the sealed split was used for measurement and for the post-run demo
display (labeled, selection frozen, never used for model selection); GPU numerics are not
bit-reproducible across CUDA versions, so a rerun may differ slightly while the qualitative
contrast (0.29 vs 0.99 order robustness) is expected to persist; checkpoints stay local
(`artifacts/arkenstone/ark014/ark014-cuda-2201-03/checkpoints/`, ~100 MB, Git-excluded) —
copy them to durable storage if the machine changes.

## 7. What the owner can see right now

Open `artifacts/arkenstone/ark014/demo-ark014-cuda-2201-03/report.html` in a browser: it shows
the run identity, the met ORDER_ROBUSTNESS_REPAIRED criterion, the aggregate table above with
denominators, and 144 real per-example predictions (baseline vs candidate vs ground truth) on
the frozen SEALED demonstration fact-sets, with every failure listed. The same data is
machine-readable in `examples.json`.

## 8. Recommended next experiment

Multi-seed non-arithmetic replication (work-order follow-up #2): repeat the ARK-014 acquisition
contrast with fresh acquisition seeds (e.g., 2202, 2203) before touching retention, and lengthen
the retention horizon (or start from a deliberately decayed state à la ARK-011) so that the
HIGH/LOW retention comparison has failure events to measure. The current single-seed repair
result predicts, but does not establish, that order augmentation repairs binding across seeds.
