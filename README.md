# AN-RA `iterate900`

This branch is built for one purpose: **900M-parameter AN-RA experiments on the
frontier profile**.

The trainable model exposed by this branch is the 900M-class frontier model. The
current build is `908,098,891` parameters:

- Hidden size: `1536`
- Layers: `36`
- Attention heads: `16`
- KV heads: `4`
- Context length: `2048`
- HAL: enabled
- ESV, RIM, DSTP, MoD, cognition, agents, memory, verification, runtime, and
  ThirdEye integration: preserved
- Gradient checkpointing: enabled for T4 feasibility

The old public `25m` and `3b` training choices are intentionally removed from
this branch. Use `frontier` only.

## Train

```powershell
python scripts/build_brain.py `
  --data_path training_data/anra_training.txt `
  --checkpoint_path anra_frontier_900m.pt `
  --model-size frontier
```

The unified dispatcher is also locked to the frontier profile:

```powershell
python -m training.train_unified `
  --mode session `
  --model-size frontier `
  --checkpoint_path anra_frontier_900m.pt
```

## ThirdEye

ThirdEye telemetry is enabled by default. It records training dynamics and deep
subsystem signals for:

- embeddings
- attention
- MLP
- normalization
- ESV
- RIM
- MoD
- HAL
- cognition
- language-model head

Run the evidence and activation report:

```powershell
python scripts/evaluate_with_thirdeye.py --profile auto
```

Disable ThirdEye intelligence telemetry only for an explicit baseline:

```powershell
$env:ANRA_THIRDEYE_INTELLIGENCE = "0"
```

## T4 Notes

This branch targets T4 experimentation, but 900M training on a T4 is tight. Keep
the frontier defaults unless you are deliberately testing memory behavior:

- batch size from `V2_1B_TRAINING`
- gradient accumulation from `V2_1B_TRAINING`
- fp16/bf16 according to hardware support
- gradient checkpointing enabled
- fixed data, seed, checkpoint, and evaluator protocol for comparisons

Use the 900M model for real feature experiments. Use ThirdEye reports to compare
sessions, detect regressions, and decide which controlled experiment to run next.
