# BRAMASTRA executable research instrument

A randomly initialized dense Transformer, explicit byte/answer/EOS serialization, paired terminal-supervision experiment, strict free generation, and full local continuation checks. No pretrained weights are loaded.

The default `smoke` profile has 117,312 parameters (two layers, width 64); it is a cheap instrument check, not the proposed 6,493,440-parameter B0 experiment. `--profile b0` constructs the full B0 shape. Neither profile is a trained AGI or a production training system.

## Run

Install the optional dependencies with `python -m pip install -e ".[bramastra]"` in a suitable environment. The committed local experiments used Python 3.11 and CPU PyTorch 2.13.0. A different backend must establish its own numerical and continuation evidence.

```powershell
.\.venv\Scripts\python.exe -m bramastra_lab.experiment --output artifacts/bramastra/my_new_run --seed 601 --steps 600 --max-seconds 90
.\.venv\Scripts\python.exe -m pytest tests/test_bramastra_model.py tests/test_bramastra_experiment.py -q -p no:cacheprovider --basetemp=.codex-test-tmp-bramastra-new
```

Use a new output directory for each run: existing results cannot be overwritten. The time limit bounds training per arm, checked between updates; evaluation, initialization, and checkpoint verification add overhead. A time-stopped or unequal-update comparison is explicitly incomplete. Maximum generation length never comes from the gold answer.

The default comparison trains on the same fixed 16 worlds and evaluates 64 fresh development worlds plus 64 new worlds with an alternate rendering. These are development probes, not independent natural transfer. Complete training-set success can reflect memorization.

The loss is `mean answer CE + terminal_weight * mean EOS CE`, with weights zero and one for the two arms. This deliberately preserves the answer-loss coefficient across arms. The short diagnostic uses a constant learning rate, FP32, and no packing; it is not the proposed B1 schedule or a long training recipe.

## Results and continuation

Each run saves the experiment manifest before model training, exact datasets, a source snapshot, raw predictions and stop reasons, learning curves, hashes, actual exposure counts, a scoped decision, and local full-state checkpoints. Each checkpoint is verified by comparing one uninterrupted update with one restored update, including optimizer moments and sampler state. Evaluation then restores the registered final checkpoint, excluding the extra verification update.

Checkpoints use `torch.load(..., weights_only=True)` and recorded identities. They remain local and are excluded from Git by the repository's existing `*.pt` rule. This is local continuation evidence; remote checkpoint durability, general resume CLI, mixed precision, TPU execution, and multi-device training are not implemented or certified here. The local implementation refuses requested unavailable CUDA rather than silently changing devices.

The research controller is a fixed rule in `decide()`. Its conclusions do not constitute model self-diagnosis. See [the target research loop](../docs/bramastra/RESEARCH_LOOP.md) and [the original experiment plan](../docs/bramastra/EXPERIMENTS.md).
