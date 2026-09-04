# AGENT.md — Citadel handover (machine-readable status for the operator's agent)

> Convention (binding): rewritten at the END of every Citadel work cycle,
> committed to the `citadel` branch ONLY, then `git push origin citadel`.
> Other branches (esoes / triquetra / cymek / ...) are read-only audit inputs —
> never modified, never pushed. A CPU/CUDA run is NEVER a TPU result. No
> fabricated device results. Preregistration and results never share a commit.
> Cycle download ceiling: <10 GB total (target <2 GB); this cycle: 0 bytes.

## STATUS

Citadel SHA: `bc09a39` (local; to be pushed this cycle)
Cymek runtime SHA selected: `298c91ac04f756f0833a7edcf63e73af3d5af688`
(= current `origin/cymek` HEAD; T0-relevant surface verified unchanged vs last
audited `105ad22`, so current HEAD is pinned, not the old SHA)

This cycle (urgent Colab hardening): operator pushed 4 PJRT-repair commits that
were audited, not redone (`bb3e763` PJRT detection, `2760b30` backend
modernization, `f460e30` launcher sync, `b31f053` handover). Remaining
deterministic gaps closed: (1) silent `optimizer.step()` fallback →
fail-closed UNSUPPORTED_OPERATION; rendezvous tries `xr` before legacy `xm`;
public `get_device()/world_size()/device_hardware()` helpers (§14), all
drivers switched to them. (2) Missing-module bootstrap: new
`citadel_tpu/runtime_bootstrap.py` (pinned SHA, fetch + detached worktree,
`sys.path` injection, file verification, one clear PRECHECK_ error) and
`python -m citadel_tpu.preflight` (SHAs, platform, PJRT, versions, file +
real-import checks per module, TPU, XLA API compat, READY_FOR_T0, exit 0/1).
(3) T0/T1/throughput drivers resolve the runtime first and record
`citadel_sha` + `cymek_runtime_sha` in every receipt; probe gate added where
missing; Kaggle-only wording removed. (4) `docs/citadel/tpu/TPU_IMPORT_GRAPH.md`
mechanical import audit. (5) Colab notebook rebuilt to cells 0–7 (checkout +
runtime + SHAs, handover, deps, install, preflight gate, probe, T0, export);
Kaggle notebook given the same bootstrap + preflight + T0 gate (its blind
`pip install torch torch-xla` replaced by inspect-first).

## OBSERVED COLAB FACTS (operator-observed evidence, NOT agent-executed receipts)

```text
torch 2.9.0+cpu | torch_xla 2.9.0 | PJRT_DEVICE=TPU | XLA xla:0
hardware TPU (Colab UI: v5e) | xr.world_size() = 1
```

`torch +cpu` tag with XLA TPU hardware is normal (execution backend is XLA).
Failures already seen live: `xrt_world_size` missing (repaired, audited),
`No module named 'anra_v5'` (repaired this cycle via pinned runtime).

## FIXES

```text
XRT world-size incompatibility (audited operator fix, kept)
PJRT bootstrap PJRT_DEVICE=TPU before torch-xla import (audited, kept)
modern device API torch_xla.device()/xr.world_size()/xr.device_type() first (audited, kept)
optimizer_step fail-closed + rendezvous modern-first + stable helpers (this cycle)
runtime module resolution via pinned detached Cymek runtime (this cycle)
import preflight with READY_FOR_T0 gate (this cycle)
notebook refresh behavior: cell 0 fetch+reset citadel, runtime bootstrap prints both SHAs (this cycle)
```

## QUESTIONS FOR OPERATOR

```text
NONE
```

## DOWNLOADS

```text
ITEM | SOURCE/PURPOSE | BYTES | CUMULATIVE BYTES
git fetch (branch refs + missing objects) | sync only | negligible (not metered) | —
(no pip installs, no datasets, no checkpoints, no wheels)
TOTAL_DOWNLOADED_GB = 0.0
```

## TPU STATUS

```text
IMPLEMENTED_NOT_RUN
```

No TPU receipt exists yet; operator screenshots are evidence, not receipts.
Local validation (real detached Cymek tree in Temp, since removed): compileall
clean; preflight prints both SHAs, files PASS, citadel imports PASS, Cymek
imports fail only on missing local torch (expected), TPU FAIL, XLA
UNAVAILABLE, READY_FOR_T0=NO, exit 1; bogus runtime dir → one clear
PRECHECK_RUNTIME_MISSING; T0 with resolved runtime → ABORT_NO_TPU (hardware
gate), never a missing-module error.

## BIGGEST BLOCKER

Need operator to rerun fresh Colab T0 using fixed branch.

## NEXT ACTION

In Colab (TPU runtime): open `notebooks/citadel_colab_tpu.ipynb` from updated
`origin/citadel`, run cells 0–7 sequentially with no edits, transfer back the
exact `TPU_ENVIRONMENT.json` / `TPU_ONE_UPDATE.json` files.

## Commit log (latest first, citadel only)

```text
bc09a39 fix(citadel): harden colab launcher bootstrap
dd8ea7b refactor(citadel): route tpu drivers through pinned runtime
12487fc feat(citadel): resolve pinned cymek runtime for tpu experiments
067e790 fix(citadel): fail-closed optimizer step and modern-first rendezvous; stable xla helpers
b31f053 docs(citadel): record Colab PJRT repair handover
```
