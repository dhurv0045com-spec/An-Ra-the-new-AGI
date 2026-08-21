# An-Ra V4 high-quality SFT handoff

This is a new SFT lineage for the continued-pretraining parent at step 30,400.
It does not resume or overwrite the older collapsed SFT child.

## Prepared corpus

- Source: `HuggingFaceTB/smol-smoltalk` small-model test shard, revision
  `f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc`.
- Source SHA-256:
  `be6773dcce145f3918ff14237b1f765affa427b0b13f6a02d397e665ac908b9a`.
- License policy: only the newly generated components identified as Apache-2.0
  by the source dataset card are accepted.
- Accepted: 5,455 unique conversations; 4,622 train, 545 validation, 288 test.
- Eight capability categories are present. Rare categories are never silently
  duplicated; their shortfalls are explicit in `sft-v4-quality-audit.json`.
- Code indentation and multiline structured answers remain intact.

The source dataset is intended for small models and its maintainers report that
it excludes advanced mathematics and favors concise conversations. This makes
it a better starting point for An-Ra 181M than relabeling raw pretraining text.
It is still only a candidate corpus: the protected pilot decides whether it is
good enough for this checkpoint.

## One-folder installation

The prepared upload is:

`training_data/sft_high_quality/ANRA_V4_HIGH_QUALITY_SFT_BUNDLE.zip`

Its SHA-256 is:

`6b7069680c61d4fd642167e0b8203530de826deb8c723c43b5cb809c8fe1282a`

Extract the eight files into `ANRA_T4_TRAINING_HOME/sft-v4`, or use:

```powershell
python -m scripts.install_high_quality_sft_bundle `
  --bundle C:\path\to\extracted-bundle `
  --training-home C:\path\to\ANRA_T4_TRAINING_HOME
```

Keep the latest step-30,400 foundation checkpoint at:

`ANRA_T4_TRAINING_HOME/anra-v4-current-full-resume.pt`

Then open `notebooks/AN_RA_T4_SFT_V4.ipynb` in Colab, select a T4, and run all.
It defaults to a 30-minute pilot, starts lineage
`anra-v4-sft-004-step30400-hq`, and uses a light frozen-parent KL anchor. Do not
switch to `RUN_MODE = "full"` until the pilot produces a passing behavior smoke
report and a signed approval bound to that exact child checkpoint.

## Promotion standard

Low loss alone is insufficient. The pilot must retain parent validation
quality, produce diverse outputs on the eight fixed behavior categories, avoid
generic-answer collapse, and save a complete child checkpoint with its lineage,
dataset, parent, optimizer, RNG, and sampler state. Failure means revise the
data or recipe and start another named lineage—not relabel the failed child.
