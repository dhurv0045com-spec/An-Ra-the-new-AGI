# 45G - Inference Engine

**Layer 06/19: `inference_runtime`**

This folder is the historical inference engine from the original staged build. It remains useful as the lower-level reference for decoding, checkpoint I/O, evaluation helpers, and standalone generation experiments.

The current canonical runtime is higher up:

- `generate.py`
- `inference/anra_infer.py`
- `inference/full_system_connector.py`
- `anra.py`

## Current Role

| File | Role |
| --- | --- |
| `greedy.py` | Deterministic argmax decoding |
| `sampling.py` | Temperature, top-k, and top-p sampling |
| `inference.py` | Prompt-to-text pipeline, streaming, batching |
| `model_io.py` | Save/load helpers, checkpoint metadata, export helpers |
| `evaluate.py` | Perplexity, speed, memory, and quality reports |
| `run.py` | Standalone historical entry point |
| `test_45G.py` | End-to-end test coverage for the old engine |

## How It Connects Now

```text
generate.py
  -> tokenizer_v3
  -> anra_brain.py
  -> streaming/traced generation
  -> memory, identity, ghost, symbolic, and API layers

history/inference (45G)
  -> standalone reference implementation
  -> decoding and evaluation patterns
  -> historical tests for the staged transformer build
```

Use this folder when you need to inspect or compare the original inference mechanics. Use `generate.py` for the mainline.

## Quick Checks

From this folder:

```bash
python test_45G.py
python run.py --prompt "An-Ra is" --max_tokens 80
```

From the repo root:

```bash
python scripts/status.py
python -m inference.full_system_connector
python anra.py --status
```

## Sampling Modes

| Strategy | Use When |
| --- | --- |
| `greedy` | You need deterministic debugging |
| `temperature` | You want controlled variation |
| `top_k` | You want a hard candidate limit |
| `top_p` | You want nucleus sampling over likely tokens |

Recommended defaults for normal generation are still conservative: `temperature` around `0.7` to `0.9`, `top_p` around `0.9` to `0.95`.

## Boundary

This layer is source-active, but it is not the operator-facing mainline. The production path is the V2 runtime and registry, not the old char-token examples. Keep this folder as a precise reference layer, not as the place to add new product behavior.
