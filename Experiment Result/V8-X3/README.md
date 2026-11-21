# ELENDRIL V8-X3 — Cognitive Intelligence Program (Pre-GPU Qualified Template)

- **Qualification Stage:** `CPU_QUALIFIED_READY_FOR_OWNER_GPU`
- **Status Axes:** `execution=PENDING` · `delivery=PENDING` · `scientific=UNRUN` · `causal=UNRESOLVED`
- **Code Source Commit (`CODE_SHA`):** `d12238b0ec26defa724e2403cfb618caa8ad7048`
- **Pinned Kaggle 2xT4 Notebook:** `notebooks/ELENDRIL_V8_X3_KAGGLE_T4X2.ipynb` (`14,356` bytes, SHA-256 `ef7a31ef5ad4b1dfa39ca6d89a0f62cab093a2cb522a71c934cb76a38647059d`, `OWNER_AUTHORIZED=True`, `SESSION_MINUTES=660`, `RESERVE_MINUTES=30`, `ONE_GO=True`)

## 1. Preregistered Program Structure (`57` GPU Jobs + Stage 5 Sealed Confirmation)
| Stage | Study | Arms (`seeds 7, 19, 41`) | Scale | Jobs | Gate / Notes |
|---|---|---|---|---:|---|
| **Stage 0** | `REDGE` (`S0`) | `R`, `R-noedge` on `D2` | `S20` | 6 | Crosses with completed X1 `D1`/`A0` + `A0/D1-H vs D2` CPU sampler audit |
| **Stage 1** | `D3` Algorithm (`S1`) | (`P`, `R`) x (`D3-Hybrid`, `D3-NoSurgery`, `D3-NoAux`) | `S20` | 18 | Tests world diversity + PCGrad gradient conflict surgery |
| **Stage 2** | `ARCH` Cognitive Screen (`S2`) | `P-R-Iter`, `P-R-Iter-Fixed`, `P-R-Iter-NoRel`, `P-R-Iter-AddMem`, `P-R-Iter-NoCopy` on `D3-Hybrid` | `S20` | 15 | Iterative delta working memory + halting + copy/generate vs 4 controls |
| **Stage 3** | `INTEG` Interface & Curriculum (`S3`) | `P-R-Iter` x (`D3-Hybrid`, `D2`) | `S20` | 6 | Dual copy/generate integration across `D3-Hybrid` vs `D2` (`gated_by="S2"`) |
| **Stage 4** | `SCALING` Evidence-Gated (`S4`) | `Base` (`D3-Hybrid`), `P` (`D3-Hybrid`), `P-R-Iter` (`CE`, `D3-Hybrid`) | `S50` (`50.55M` core) | 12 | `GATED_PENDING_S20_QUALIFICATION` (`0.0` GPU s if Stage 2/3 gate fails) |
| **Stage 5** | Sealed `confirm` & OOD (`S5`) | Qualified checkpoints (`pending_confirm_jobs == []` required) | `eval` | 1 bundle | Fails closed (`SEALED_CONFIRMATION_LOCKED`) on partial cohorts |

## 2. Ten-Family Cognitive Battery & Five Learning Modes (`experiments/x3/cognitive_battery.py`, `experiments/x3/models.py`)
- **Families (`C01`–`C10`):** `C01_NOVEL_OP_INDUCTION`, `C02_FST_RULE_COMPOSITION`, `C03_GRID_TRANSFORM_ARC`, `C04_COMPOSITIONAL_SCAN`, `C05_RELATIONAL_GRAPH_CF`, `C06_OPEN_WORLD_DEDUCTION`, `C07_HIERARCHICAL_LISTOPS`, `C08_ADAPTIVE_MEMORY_REVISION`, `C09_METACOGNITION_SELF_CHECK`, `C10_LANGUAGE_MATH_TRANSFER`.
- **Learning Modes (`evaluate_five_learning_modes`):** (1) frozen in-context `{0, 1, 4, 16, 64}` shots, (2) persistent internal working-memory state adaptation, (3) bounded floating-param fast gradient adaptation, (4) tool-free vs oracle separation, and (5) symbolic-to-prose/grid zero-shot transfer.
- **Provenance Separation:** Internal matched analog evaluations stay strictly separated from external JSONL benchmarks (`official_external` vs `interface_adapted_external`).

## 3. CPU Qualification Checks Observed Before GPU Launch
- Full 7-suite pytest regression matrix passed (`44 passed in 172.22s`, `0` failures via `python -m pytest -q`).
- Clean `git archive d12238b0` out-of-repo import verified `57` total GPU jobs (`45` S20, `12` S50) and X2 `U0` before `BII-FROZEN` ordering (`u0_max=41 < bii_min=93`).
- Kaggle dual-GPU execution (`S0`–`S4` training, peak VRAM profiling, full S20/S50 runs, and Stage 5 sealed confirmation) remains **PENDING** owner notebook execution.
