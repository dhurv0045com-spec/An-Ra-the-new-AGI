# EXP-PLANTHIR-002: End-to-End Bounded CPU Causal Observatory Validation

- **Date (UTC):** 2026-11-15T22:29:37.275217+00:00 | **Elapsed:** `12.46s` on CPU
- **Scope Warning:** `CPU tiny bounded canary (<=100k params, 6 updates/arm; mechanism/pipeline validation ONLY, not capability scaling)`
- **Evaluation Worlds:** `16` split-disjoint D1 worlds (`8` multi-hop)

## Cross-Architecture Telemetry, Funnel & Causal Summary (Tiny 6-Update CPU Controls)

| Arm | Params | Overall Acc | Stage1 Info | Stage3 1-Hop | Stage4 M-Hop | Oracle Delta | Probe Final | Eff. Rank | Capsule Bytes |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B0` | 17,680 | 0.000 | 0.500 | 0.000 | 0.000 | +0.000 | 0.071 | 5.20 | 13,017 |
| `J` | 18,307 | 0.062 | 0.500 | 0.100 | 0.000 | +0.000 | 0.071 | 5.13 | 13,429 |
| `F` | 18,340 | 0.000 | 0.500 | 0.000 | 0.000 | +0.000 | 0.143 | 5.09 | 14,151 |
| `P` | 19,397 | 0.000 | 0.500 | 0.000 | 0.000 | +0.000 | 0.143 | 5.11 | 14,778 |
| `F-cap` | 19,412 | 0.000 | 0.500 | 0.000 | 0.000 | +0.000 | 0.143 | 5.09 | 14,147 |
| `U` | 26,756 | 0.000 | 0.500 | 0.000 | 0.000 | +0.000 | 0.143 | 5.11 | 14,930 |

## Scientific Interpretation & Epistemic Firewall Checks

- **Checkpoint Intelligence:** Discovered & restored all 6 architectures from Tier-3 stores; J-vs-F `single_architecture_contrast=True`.
- **Offline Tier-2 Capsule Probing:** Verified SHA-256 tamper-checked `representations.npz` + `diagnostics.json` capsules (< 50 KB each) reproduce representation rank and linear probes without loading model checkpoints.
- **Packaged Result & Checkpoint Archives:** Verified `result.zip` (`1,293,821` B, mode=`result.zip (full_embedded)`) and `checkpoint_full.zip` (`1,295,931` B, `6` lineages SHA-256 restored).
- **Epistemic Discipline:** At 6 CPU updates (`17,680–26,756` params), confirmed/suspected readout-vs-representation gaps and scale uncertainty remain explicitly separated; no unrun S20 GPU scaling capability claims are made.

