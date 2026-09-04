# AGENT.md — Citadel handover (machine-readable status for the operator's agent)

> Convention (binding): rewritten at the END of every Citadel work cycle,
> committed to the `citadel` branch ONLY, then `git push origin citadel`.
> Other branches (esoes / triquetra / cymek / ...) are read-only audit inputs —
> never modified, never pushed. A CPU/CUDA run is NEVER a TPU result. No
> fabricated device results. Preregistration and results never share a commit.
> Cycle download ceiling: <10 GB total (target <2 GB); this cycle: 0 bytes.

## STATUS

Citadel SHA: `a11aa62` (local; to be pushed this cycle)
Cymek runtime SHA: `298c91ac04f756f0833a7edcf63e73af3d5af688` (= `origin/cymek`)

**T0 PASS on real Colab TPU.** Operator returned `TPU_ONE_UPDATE.json`
(`docs/citadel/tpu_receipts/TPU_ONE_UPDATE.json`, commit `a11aa62`).
Validation before acceptance: `citadel_sha` == pushed `525a357` == env
`git_sha` (fresh checkout); `cymek_runtime_sha` == pin; MINI_SPEC param count
exactly 1,647,104 (matches prereg spec); loss 10.1228 ≈ ln(24576) ≈ 10.11
theoretical untrained CE; supervised 1022/1024 (2 BOS excluded, as designed);
before/after param hashes differ; grad norm finite; reload identical;
platform=colab, PJRT, torch 2.9.0+cpu / torch-xla 2.9.0, 1 device, probe_pass.
Minor gap (non-blocking): `PJRT_DEVICE` env value not an explicit env-block
field (`xla_runtime=PJRT` covers it) — add it next touch of the probe.

## PLATFORM

```text
colab
```

## ENVIRONMENT (from T0 receipt, 2026-09-04T18:45:56Z)

TPU detected: true | TPU identity: TPU (Colab) | XLA devices: 1
Python: 3.13.15 | PyTorch: 2.9.0+cpu | torch-xla: 2.9.0
Host: x86_64, 50.5 GB RAM, 221 GB disk free

## DOWNLOADS

```text
ITEM | SOURCE/PURPOSE | BYTES | CUMULATIVE BYTES
(operator-side Colab installs only; nothing downloaded this cycle locally)
TOTAL_DOWNLOADED_GB = 0.0
```

## T0

Status: PASS | Loss: 10.1228 | Grad norm: 0.3731 | Parameter mutation: YES (hashes differ)
Checkpoint save: YES (`9c7870cf…`) | Checkpoint reload: identical YES
Wall time: 6.37 s | Tokens/sec: 160.6 (cold, compile included)

## CALCULATOR

Status: NOT_RUN (UNBLOCKED by T0 PASS — next) | Parameters: MINI_SPEC (~1.6M)
Train examples: 0 | Train tokens: 0 | Updates: 0
Loss start: — | Loss end: — | Untrained test accuracy: —
Final test accuracy: — | Checkpoint hash: — | Reload result: —
Tokens/sec: — | Generator: calculator-canary/1.1 (locally validated)

## 5B DATA

```text
NOT_STARTED
```

`0 / 5,000,000,000`. Sequenced behind T1→T2.

## QUESTIONS FOR OPERATOR

```text
NONE
```

## BIGGEST BLOCKER

Calculator learning not yet demonstrated — T1 is the gate for everything after.

## NEXT ACTION

Run the calculator canary on the same Colab TPU runtime (small progression:
5 → 20 updates → small run per escalation policy); return the exact
`TPU_CALCULATOR_CHECKPOINT.json` file.

## Commit log (latest first, citadel only)

```text
a11aa62 test(citadel): certify one cymek update on colab tpu
525a357 docs(citadel): update colab tpu repair handover
```
