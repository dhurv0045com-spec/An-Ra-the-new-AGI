# EXP-PLANTHIR-004: Level-5 5-Gate Cartridge Live GPU Validation

- **Date (UTC):** 2026-11-15T22:28:24.915933+00:00 | **Device:** `NVIDIA GeForce RTX 4050 Laptop GPU` (`cuda:0`)
- **Elapsed:** `51.464s` | **Peak VRAM:** `75.56 MB` | **Updates/Arm:** `60` (milestones `[20, 40, 60]`)
- **All 5 Launch Gates Passed:** `True` | **Evaluated:** `24` D1 worlds (`12` multi-hop)

## 5-Gate Self-Audit & Result Archive Parity

| Gate | Check | Result |
| :--- | :--- | :--- |
| Gate 1 | Instrument Integrity & Evaluator Parity | `True` |
| Gate 2 | Confounder Firewall (Single-Variable Design) | `True` |
| Gate 3 | Gradient Health & Milestone Commits | `True` |
| Gate 4 | Milestone Trajectory + 6 Causal Interventions | `True` |
| Gate 5 | `result.zip` (`4,255,655` B) + `checkpoint_full.zip` (`12,624,167` B) Live Parity | `True` |

## Cross-Architecture Causal & Trajectory Summary

| Arm | Params | Loss (Init -> Final) | Overall Acc | Stage1 Info | Probe Final | Surgical Ablation Delta | Bypass? | Depth Gain (`U`) | Live Parity |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | :--- | ---: | ---: |
| `J` | 92,115 | 2.873 -> 2.517 | 0.000 | 0.542 | 0.190 | +0.000 | `False` | +0.000 | 1.000 |
| `F` | 92,180 | 2.933 -> 2.501 | 0.000 | 0.542 | 0.095 | +0.000 | `False` | +0.000 | 1.000 |
| `P` | 94,293 | 2.860 -> 1.860 | 0.000 | 0.500 | 0.048 | +0.000 | `False` | +0.000 | 1.000 |
| `U` | 121,300 | 2.932 -> 2.509 | 0.000 | 0.542 | 0.095 | +0.000 | `True` | +0.000 | 1.000 |

## Token Fuel Savings & Context Compression (`AI_HANDOFF_BRIEF.json`)

- **Estimated AI Workflow Token Reduction:** `82.0%` (`180,000` tokens vs `1,000,000` from-scratch baseline)
- **Diagnostic Handoff Compression:** `48,283` B full arm JSON -> `5,978` B (`~1,494` tokens, `8.66x` compression)

