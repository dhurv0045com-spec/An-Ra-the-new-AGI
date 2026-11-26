# ELENDRIL V8-X3 — Cognitive Intelligence Program (Pre-GPU Qualified One-Go Template)

**2026-11-26 owner results.** The full archive verifies 45 S20 jobs and 12 gated S50 jobs. See the [owner-run analysis](OWNER_RUN_01.md) and [audited metrics](OWNER_RUN_01_METRICS.json). The qualification notes below describe the earlier pre-launch state.


- **Qualification Stage:** `CPU_QUALIFIED_READY_FOR_OWNER_GPU`
- **Status Axes:** `execution=PENDING` · `delivery=PENDING` · `scientific=UNRUN` · `causal=UNRESOLVED`
- **Code Source Commit (`CODE_SHA`):** `8485dbb695e5bfae63e4a2eeeb8da391fbe3317a`
- **Pinned Kaggle 2xT4 Notebook:** `notebooks/ELENDRIL_V8_X3_KAGGLE_T4X2.ipynb` (`18,729` bytes, SHA-256 `117b05b2d3d7d9b01c8cf80e56597b855c12bb0f9a67c5a228adee749e2bdf3e`, `OWNER_AUTHORIZED=True`, `ONE_GO=True`, `SESSION_MINUTES=660`, `RESERVE_MINUTES=45`, `PER_JOB_DISK_MB=95`)

## 1. Single-Session One-Go Architecture & Budget (`57` GPU Jobs + Auto Stage 5)
- **Why One Session Works:** `1,024` FP16 AMP updates with `prepack_d3_batches` (`400` train cases across 10 families = `41` epochs) runs ~4–6x faster (`~3.0–4.5h` total across `2×T4`, well inside `11h`), while completed production checkpoints strip redundant finished optimizer states (`optimizer.bin = {}`, `~80.3 MB` S20 / `~204.1 MB` S50 -> `<= 6.1 GB` across all 57 models vs Kaggle's `19.5 GiB` disk ceiling).
- **Level-7 Runtime & Crash Hardening:** `UTF8.is_ascii_stop_token` + `UTF8.decode_lossy` (`valid_utf8: bool`, eliminating the Stage-5 `0xB5` decode crash), FP32 attention/log-softmax reduction casts inside `torch.amp.autocast(..., dtype=torch.float16)`, `try/except BaseException (CRASH_RECEIPT.json)/finally` `/kaggle/working` salvage mirroring, and `palantiri doctor` / `palantiri shield` pre-launch qualification gates.
- **Stages:** `S0` (`REDGE`, 6 S20) + `S1` (`D3`, 18 S20) + `S2` (`ARCH`, 15 S20) + `S3` (`INTEG`, 6 S20) = **45 S20 jobs**, followed by **Stage 4** (`12` evidence-gated `S50` `50.55M` jobs; held cleanly in `gated_s50_job_ids` at `0.0` GPU s if S20 gate fails), followed immediately by **Stage 5** (`X3_SEALED_CONFIRMATION_RESULTS.json`, `X3_OOD_TRANSFER_RESULTS.json`, `X3_FIVE_LEARNING_MODES.json`, plus per-job `dev_mistakes` in `JOB_RECEIPT.json`), all packed into `ELENDRIL_V8_X3_ANALYSIS.zip`.

## 2. CPU Qualification Checks Observed Before GPU Launch
- Level-7 docs & contract suite (`python -m pytest tests/test_planthir_two_agent_manuals.py tests/test_planthir_fp16_autocast_gate.py tests/test_planthir_y4_x3_feedback_upgrades.py -q`) passed `9 passed in 6.74s` on `8485dbb6`.
- Combined Level-7/Level-6/manual/FP16/X3/tree regression suite (`python -m pytest tests/test_planthir_two_agent_manuals.py tests/test_planthir_level7.py tests/test_planthir_level6.py tests/test_planthir_fp16_autocast_gate.py tests/test_planthir_y4_x3_feedback_upgrades.py tests/test_x3_recover_compute_stop_20261124.py tests/test_elendril_tree.py -n auto -q`) passed `32 passed in 90.16s` (`0` failures), and `python -m anra_v5.palantiri shield` returned `LEVEL7_SHIELD_PASS` (`4/4` shields passed).
- Notebook verification (`python tools/build_elendril_x3_notebook.py --source 8485dbb695e5bfae63e4a2eeeb8da391fbe3317a --check`) returned `match=true` (`18,729` bytes) with all `5` code cells passing `ast.parse`.
- Kaggle GPU execution (`S0`–`S5`) remains **PENDING** owner notebook execution (`Run All` on **GPU T4 x2** or single **T4**).
