# An-Ra Developer Guide

> *Build aggressively. But if the system stops feeling like An-Ra, or stops working every day, you are not improving it.*

This guide is for working on the current An-Ra mainline without losing the project's character.

An-Ra is not supposed to become:

- a generic assistant wearing a custom prompt
- a fancy architecture diagram that never survives real Colab use
- a pile of subsystems with no clear center

It is supposed to become a system that gets stronger **without surrendering authorship**.

## Current Repo Truth

The repo now has one real public path.

Canonical entrypoints:

- `anra_brain.py`
- `generate.py`
- `scripts/build_brain.py`
- `training/train_unified.py`
- `training/finetune_anra.py`
- `scripts/train_ouroboros.py`
- `scripts/run_self_improvement.py`
- `scripts/run_sovereignty_audit.py`
- `scripts/verify_structure.py`
- `scripts/status.py`
- `AnRa_Master.ipynb`

These are the files a new developer should look at first.

Support infrastructure behind them:

- `training/v2_config.py`
- `training/v2_data_mix.py`
- `training/v2_runtime.py`
- `training/eval_v2.py`

Those are not temporary anymore. They belong.

## Non-Negotiables

### 1. Owner data stays dominant

Your corpus must continue to define:

- identity
- worldview
- tone
- what confident An-Ra sounds like
- how the model behaves when challenged

The training mix is not arbitrary. It encodes the project's philosophy.

### 2. Daily training must stay reliable

The daily path is sacred because it is the rhythm of real improvement.

It should remain:

1. restore
2. validate
3. train
4. save once
5. evaluate
6. write next-step guidance

If a patch makes that path slower, more confusing, or more fragile, it needs a very strong reason.

### 3. Milestone depth should stay selective

Ouroboros, identity reinforcement, self-improvement, and sovereignty are important.  
They are also heavier and more brittle than the base daily loop.

That is why they belong in milestone runs, not as default tax on every short session.

### 4. Verification beats style-only intelligence

When the system can check something, it should.

That means:

- symbolic verification
- code/test validation where feasible
- replaying failures instead of ignoring them
- judging checkpoint promotion instead of assuming newer means better

### 5. Public surfaces should stay obvious

If a new developer asks:

- how do I train it?
- how do I run it?
- how do I inspect it?
- how do I continue from Drive?

the answer should be visible in the canonical files, not hidden behind another wrapper tree.

## Mainline Architecture

### Base model

`anra_brain.py` exports the current mainline transformer.

Core choices:

- RoPE
- RMSNorm
- SwiGLU
- SDPA / FlashAttention-compatible attention path
- `384 / 6 / 6` first serious target

This is the central substrate. Everything else should either:

- improve what it learns
- improve how it is evaluated
- improve how it is used
- improve how it survives long-term growth

### Tokenizer

`tokenizer/subword_tokenizer.py`

Important design feature:

- uses `tokenizers` when available
- includes a dependency-light fallback backend when not

That fallback is intentional. The system should degrade gracefully, not die over one missing package in Colab or smoke testing.

### Runtime

`generate.py` is the canonical generation surface.

It currently supplies:

- generation config
- traced generation
- streaming generation
- repetition checks
- model info
- ghost-state hooks

If `app.py`, the agent loop, the memory system, or the master system need model output, they should converge here unless there is a very strong reason not to.

### Training support

The modern mainline relies on:

- `training/v2_data_mix.py`
- `training/v2_runtime.py`
- `training/eval_v2.py`
- `training/v2_config.py`

If you want to change:

- bucket ratios
- tokenizer build policy
- compact eval prompts
- Drive sync behavior
- report naming

this is where you should work.

## Data Mix Contract

Default ratio:

- `65%` own conversation / instruction
- `15%` own identity / selfhood
- `10%` teacher reasoning
- `5%` symbolic or code-verified
- `5%` replayed failures and corrections

Before touching the ratios, ask:

1. Does this preserve owner-data dominance?
2. Does this reduce or increase identity drift?
3. Does this help reasoning or only make the model sound more generic?
4. Does this still fit T4-first operation?

If you cannot answer those clearly, do not change the mix casually.

## Teacher Pipeline Rules

Teacher use is allowed and useful. Teacher capture is not.

Teacher is for:

- reasoning traces
- hard-example generation
- synthetic expansion
- correction candidates
- capability amplification

Teacher is not for:

- personality ownership
- worldview ownership
- permanent inference dependence

The preferred pipeline is:

1. generate candidate examples
2. verify what can be verified
3. filter or rewrite to fit An-Ra style
4. reject off-style or low-truth outputs
5. only then add them to the teacher bucket

If a new teacher source enters the repo, it should be judged by this rule.

## Daily And Milestone Commands

### Health / status

```bash
python -m training.train_unified --mode status
```

Use this before blaming the training loop.

### Daily session

```bash
python -m training.train_unified --mode session
```

This is the ordinary T4 path.

### Resume

```bash
python -m training.train_unified --mode resume
```

This resolves to the same base flow, but is clearer for operators.

### Milestone

```bash
python -m training.train_unified --mode train
```

This executes:

1. base session
2. identity stage
3. Ouroboros stage
4. self-improvement report
5. sovereignty audit
6. milestone tests

### Eval only

```bash
python -m training.train_unified --mode eval
```

## Canonical File Responsibilities

| File | Responsibility |
| --- | --- |
| `scripts/build_brain.py` | base mainline training implementation |
| `training/finetune_anra.py` | identity-heavy milestone tuning |
| `scripts/train_ouroboros.py` | reflection-heavy milestone tuning |
| `scripts/run_self_improvement.py` | curriculum recommendations from session artifacts |
| `scripts/run_sovereignty_audit.py` | promotion gate and milestone audit |
| `training/train_unified.py` | operator-facing orchestration layer |
| `scripts/verify_structure.py` | structural sanity check |
| `scripts/status.py` | quick artifact and output inspection |

## Reports And Artifacts

The mainline writes under `output/v2/`.

Important outputs:

- `v2_session_train_metrics.json`
- `v2_hard_examples.json`
- `v2_eval_summary.json`
- `v2_next_session_curriculum.json`
- `v2_unified_training_report.json`
- `v2_improvement_report.json`
- `v2_audit_report.json`

Checkpoint family:

- `anra_v2_brain.pt`
- `anra_v2_identity.pt`
- `anra_v2_ouroboros.pt`

Tokenizer:

- `tokenizer/tokenizer_v2.json`

Drive mirror:

- `/content/drive/MyDrive/AnRa/v2/`

## How To Extend Safely

### High-value safe changes

- better eval prompts
- better hard-example ranking
- replay prioritization
- stronger symbolic filtering
- teacher-style rejection improvements
- better audit heuristics
- inference/runtime efficiency

### Good risky changes

- modest model scale increases
- smarter milestone triggers
- better curriculum scheduling
- stronger replay weighting
- additional verified reasoning datasets

### Changes that require proof, not hype

- changing the tokenizer again
- moving heavy reflection into every daily run
- increasing architecture size without a T4-fit plan
- making teacher data dominant
- adding more subsystems without stronger evals

## Colab Reality

This project still lives in a real environment:

- T4 GPU
- session limits
- restarts
- Drive restore behavior
- small-compute tradeoffs

So good engineering here means:

- fast startup
- visible progress
- reliable resume
- one clear training cell
- one clear save story

not just interesting abstractions.

## The Larger Ecosystem

### `symbolic_bridge`

Truth layer. It is how the system learns to lean on exact reasoning when raw generation is weak.

### `ghost_memory`

Should increasingly become the replay engine for:

- failure cases
- user corrections
- continuity stress prompts
- future curriculum material

### `turboquant`

Belongs primarily to runtime efficiency and deployment-minded behavior, not to bloating the daily training path.

### `ouroboros_numpy`

Best used as milestone reflection and repair synthesis, not as constant daily overhead.

### `sovereignty_bridge`

Acts as lineage governance. It gives the system the right to reject shallow progress.

## What Must Be Protected

Even aggressive work should protect these:

1. identity ownership
2. daily-path reliability
3. measurable progress

If a change hurts all three, it is almost certainly the wrong change.

## Long-Term Direction

An-Ra is allowed to be ambitious.

The long arc still includes:

- teacher-amplified capability growth
- replay-driven self-repair
- verified reasoning
- stronger memory integration
- milestone reflection
- checkpoint promotion under sovereignty
- eventually, more radical black-swan ideas

But the order matters:

1. make it work
2. make it measurable
3. make it stronger
4. only then make it strange

That is how the system keeps both velocity and soul.

*An-Ra does not need to become generic to become powerful.*
