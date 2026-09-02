# Senora P35 Remote Cluster Execution Runbook

This runbook defines the exact sequence to execute An-Ra's P35 scientific training on authorized remote compute.

**Target Branch**: `senora`  
**Source Commit**: `c6f88cb5a42a8f60ef34d6ff382624ada1b96d1c`  

---

## 1. Prerequisites Checklist

| Prerequisite | Category | Required Path | Status |
|---|---|---|:---:|
| `pyproject_toml` | environment | `pyproject.toml` | **PASS** |
| `p35_model_constructor` | code | `senora/model.py` | **PASS** |
| `p35_training_step` | code | `senora/training_step.py` | **PASS** |
| `p35_experiment_plan` | specification | `artifacts/v5/p35_cms1_plan.json` | **PASS** |
| `remote_preflight_canary` | code | `senora/canary.py` | **PASS** |
| `remote_experiment_runner` | code | `senora/run_experiment.py` | **PASS** |
| `slurm_batch_scripts` | script | `artifacts/v5/cluster_jobs/run_control-substrate-00.sbatch` | **PASS** |
| `execution_manifests` | specification | `artifacts/v5/cluster_jobs/manifest_control-substrate-00.json` | **PASS** |
| `result_classifier_engine` | code | `senora/result_classifier.py` | **PASS** |
| `triquetra_neutral_bridge` | code | `senora/triquetra_bridge.py` | **PASS** |
| `external_corpus_pack_shards` | external_data | `data/packs/v5_tokens/*.bin` | **BLOCKED** |
| `signed_data_manifest` | external_data | `data/packs/data_manifest.json` | **BLOCKED** |

---

## 2. Remote Cluster Setup

On the GPU/TPU cluster node:
```bash
# 1. Clone repository and checkout senora
git clone https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI.git
cd An-Ra-the-new-AGI
git checkout senora

# 2. Install dependencies via uv / pip
pip install -e .
```

---

## 3. Mandatory Preflight Canary (1–2 minutes)

Before launching training, execute the target accelerator canary:
```bash
python -m senora.canary \
    --device cuda \
    --remote-authorized \
    --output logs/canary_receipt.json
```
Assert that `logs/canary_receipt.json` reports `status: "PASS_CANARY_CERTIFIED"`.

---

## 4. Phase P35-A Job Dispatch

Submit the matched treatment and control arms via SLURM:
```bash
sbatch artifacts/v5/cluster_jobs/run_control-substrate-00.sbatch
sbatch artifacts/v5/cluster_jobs/run_cognition-mixture-15-ce.sbatch
```

---

## 5. Automated Result Classification & Next Steps

Upon completion, inspect `output/p35_control/receipt_control-substrate-00.json` and `output/p35_cog_ce/receipt_cognition-mixture-15-ce.json`.

Run result aggregation:
```bash
python -m senora.result_classifier \
    --control output/p35_control/receipt_control-substrate-00.json \
    --treatment output/p35_cog_ce/receipt_cognition-mixture-15-ce.json
```

- If **`ROBUST_POSITIVE`**: submit `artifacts/v5/cluster_jobs/run_cognition-mixture-15-qswap.sbatch`.
- If **`NO_EFFECT`** or **`SYNTHETIC_ONLY`**: halt scientific training immediately.

Neutral causal observation records will be in `output/p35_cog_ce/triquetra_bridge/`.
