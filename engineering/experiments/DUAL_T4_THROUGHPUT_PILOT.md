# Dual-T4 training-throughput pilot

## Purpose

Compare microbatch sizes while two independent workers train concurrently, one
worker on each T4. This pilot identifies a throughput/memory operating point
for the registered BRAMASTRA 8-layer, width-256, 512-token model. It does not
alter the frozen K8 campaign, and it makes no capability or task-quality claim.

## Kaggle run

Use a Kaggle notebook with **GPU T4 x2** enabled and the BRAMASTRA source
available on `PYTHONPATH`. Run:

```bash
python -m bramastra_lab.research.campaigns.gpu_throughput \
  --devices 0,1 --batch-sizes 1,2,4,8,16,32,64 --seconds-per-case 20 \
  --sequence-length 512 --target-utilization 75 \
  --out /kaggle/working/dual_gpu_throughput.json
```

The default run lasts about 140 seconds plus startup. Both workers test each
batch size at the same time. For every condition each starts from the same
fresh initialization and a new AdamW optimizer. The random byte-token workload
isolates model-training throughput; it is intentionally not counted as K8
learning evidence.

## Read the result

Compare `tokens_per_second`, `final_loss`, `finite`, and peak allocated memory
for each batch size on each device. The report also samples `nvidia-smi` once
per second and groups utilization/memory by active batch size. Prefer the
recommended smallest batch size meeting the target with stable memory headroom;
if no candidate reaches 75%, the recommendation names the highest-throughput
measured fallback and reports that the target was missed. Do not select a batch
merely to hit 80% utilization. GPU utilization is a
sampled time-active measure, not a model-quality metric, and can vary by driver
and kernel scheduling. If a condition runs out of memory it is marked `oom`
and later conditions continue where possible.

## Scope boundary

This is a from-scratch performance probe, not speech transcription or a
validation of the frozen K8 objectives. Batch size changes optimizer-update
semantics, so no selected value is applied to K8 automatically. Any adoption
requires a separately registered training comparison with matched data,
initialization, token budget, and held-out task evaluation.
