# AGENT.md — Citadel handover (machine-readable status for the operator's agent)

> Convention (binding): rewritten at the END of every Citadel work cycle,
> committed to the `citadel` branch ONLY, then `git push origin citadel`.
> Other branches are read-only audit inputs — never modified, never pushed.
> A CPU/CUDA run is NEVER a TPU result. No fabricated device results.
> Preregistration and results never share a commit. Download ceiling <10 GB;
> this cycle: 0 bytes.

## STATUS

Citadel SHA: `83d8ac3` (local; to be pushed this cycle)
Pinned Cymek runtime SHA: `298c91ac04f756f0833a7edcf63e73af3d5af688` (unchanged)

**T1 FAIL recorded honestly.** Operator ran the frozen T1 on a legit tree
(`935f822` = prior tip + notebook-only addition, audited; all executed code
identical to the reviewed path) and returned the receipt
(`docs/citadel/tpu_receipts/TPU_CALCULATOR_CHECKPOINT.json`). Provenance
verified before acceptance: all 3 data split hashes regenerate byte-identical
locally; token accounting exact (69664 = 76064 − 6400); pin + MINI_SPEC exact.
(I caught and fixed my own one-field transcription slip during recording —
verified clean after.)

## T0

```text
PASS (unchanged, still applicable)
```

## T1

```text
FAIL (per frozen gate; only loss + reload rules passed)
```

Loss 10.10→2.85, dev CE 10.08→2.91, but exact-match 0/500 untrained AND
trained, nulls ≤1.4%, reload hash-identical. Correct reading: strong
loss-learning with zero answer-rule movement — prompt/format modeling without
rule extraction (H2 direction), scale floor not ruled out. The receipt lacked
the discriminator (stop reasons, answer-token CE, samples) — added this cycle
so the next run is diagnosable by construction.

## T1B (batched multi-scale, single session — answers the batching request)

`docs/citadel/experiments/T1B/PLAN.md` PREREGISTERED, code + notebook ready.
Five fixed-budget arms [1k, 2.5k, 5k, 10k, 20k updates ≈ 1M–20M cap tokens,
~39M total ≈ tens of MB text-equivalent] in ONE session: same seed/init/data,
exactly one variable (budget). Per arm: own baseline, DEV info, TEST once,
train-sample memorization lens, nulls, reload hash, stop hist, answer-CE,
full receipt. TEST: 10 new frozen observations total, all reported, Holm over
per-arm claims. Repetition disclosed (max arm = 160 epochs over 4000 rows).
Checkpoint binaries exported for smallest-PASS + final arm (the T1 lesson:
ephemeral checkpoints destroy diagnosability). Est ~75–100 min, ceiling <2 TPU-h.

## DOWNLOADS

```text
ITEM | SOURCE/PURPOSE | BYTES | CUMULATIVE BYTES
(none — no pip installs, no datasets, no checkpoints, no artifacts)
TOTAL_DOWNLOADED_GB = 0.0
```

## QUESTIONS FOR OPERATOR

```text
NONE
```

## BIGGEST BLOCKER

Need operator to execute the preregistered T1B session (`citadel_colab_t1b.ipynb`).

## NEXT ACTION

On Colab TPU with updated `origin/citadel`: run `notebooks/citadel_colab_t1b.ipynb`
end to end (gates → 5 arms in order, no skipping, no edits); transfer back all
`TPU_SCALE_*.json` receipts + the two exported checkpoint binaries.

## Validation this cycle (local, 0 downloads)

6/6 unit tests (Decimal Wilson reference, data invariants, nulls, notebook
references incl. the new notebook); compileall clean; fail-closed intact;
T1-default training path behavior-identical (eval-only additions).

## Commit log (latest first, citadel only)

```text
83d8ac3 feat(citadel): add T1B scale notebook
6b1f9b3 feat(citadel): add diagnosis lens and fixed-budget arms
875a235 research(citadel): preregister T1B batched scale run
db961ae test(citadel): record colab T1 calculator FAIL receipt
```
