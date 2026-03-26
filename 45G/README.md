# 45G — Inference Engine & Model Management

Final module of the transformer LM build.  
Steps 27–33: from trained weights to generated text.

---

## What this is

The part of the system that makes the model **speak**.

Built on top of everything from 45A–45F (tokenizer, embeddings,
positional encoding, attention, transformer blocks, training loop,
checkpointing). This module adds:

| Step | File | What it does |
|------|------|--------------|
| 27 | `greedy.py` | Deterministic argmax decoding |
| 28-30 | `sampling.py` | Temperature · Top-k · Top-p (nucleus) |
| 31 | `inference.py` | Full pipeline: prompt → text, streaming, batching |
| 32 | `model_io.py` | Save, load, ONNX export, checkpoint versioning |
| 33 | `evaluate.py` | Perplexity, speed, memory, generation quality |
| — | `run.py` | Single command entry point |
| — | `test_45G.py` | Full end-to-end test suite |

---

## Requirements

```bash
pip install torch
# Optional (for ONNX export):
pip install onnx onnxruntime
```

Python 3.9+. No other ML libraries required.

---

## Quickstart — 30 minutes to first generation

### 1. Clone and enter the directory

```bash
git clone <repo>
cd 45G
```

### 2. Run the tests to verify everything works

```bash
python test_45G.py
```

All tests should pass. Expected output ends with:
```
OK
Ran 38 tests in X.XXXs
```

### 3. Generate with an untrained model (instant, no checkpoint needed)

```bash
python run.py --prompt "Once upon a time" --max_tokens 80
```

Output will be nonsensical — the model has random weights.  
That is correct. You need a trained checkpoint for coherent output.

### 4. Generate from a trained checkpoint

```bash
python run.py \
  --prompt "The future of artificial intelligence is" \
  --checkpoint best_model.pt \
  --strategy top_p \
  --temperature 0.8 \
  --max_tokens 200
```

### 5. Stream output token by token

```bash
python run.py \
  --prompt "Tell me a story about" \
  --checkpoint best_model.pt \
  --stream \
  --temperature 0.9
```

---

## All sampling strategies

```bash
# Greedy — always picks the most likely token (deterministic)
python run.py --prompt "Hello" --strategy greedy

# Temperature — full softmax, temperature controls randomness
python run.py --prompt "Hello" --strategy temperature --temperature 0.7

# Top-k — restrict to k most likely tokens before sampling
python run.py --prompt "Hello" --strategy top_k --top_k 40 --temperature 0.8

# Top-p (nucleus) — restrict to minimum set covering p probability mass
python run.py --prompt "Hello" --strategy top_p --top_p 0.92 --temperature 0.8
```

Temperature guide:
- `0.1 – 0.4` → focused, nearly deterministic
- `0.5 – 0.8` → balanced (recommended)
- `0.9 – 1.2` → creative, more varied
- `1.5 – 2.0` → chaotic (useful for brainstorming)

---

## Save and load a checkpoint

```python
from model_io import save_checkpoint, load_checkpoint, CheckpointMetadata

# After training:
meta = CheckpointMetadata(
    model_class="TransformerLM",
    d_model=128, n_heads=4, n_layers=2,
    vocab_size=128, max_seq_len=512,
    epoch=50, train_loss=1.23, val_loss=1.45,
    dataset="shakespeare", notes="first run"
)
save_checkpoint(model, "checkpoints/model_v1.pt", metadata=meta, tokenizer=tok)

# To load:
from run import TransformerLM, CharTokenizer
model = TransformerLM(vocab_size=128, d_model=128, n_heads=4, n_layers=2)
meta  = load_checkpoint(model, "checkpoints/model_v1.pt")
```

### List available checkpoints

```bash
python run.py --list_checkpoints checkpoints/
```

```
File                                   MB  Epoch   ValLoss Date
────────────────────────────────────────────────────────────────────
model_v3.pt                           4.2     50    1.2100 2025-01-15
model_v2.pt                           4.2     30    1.4800 2025-01-14
model_v1.pt                           4.2     10    2.1200 2025-01-13
```

---

## Batch inference

```bash
# Create a prompts file
echo '["Hello world", "The sky is", "Once upon a time"]' > prompts.json

python run.py --batch_file prompts.json --checkpoint best_model.pt --max_tokens 100
# Writes: batch_output.json
```

---

## Evaluation suite

```bash
# Full eval: perplexity + speed + memory + generation quality
python run.py --eval --checkpoint best_model.pt --eval_corpus test_data.txt
```

Or from Python:

```python
from evaluate import run_eval_suite
from inference import InferencePipeline

pipe    = InferencePipeline(model, tokenizer)
results = run_eval_suite(model, tokenizer, test_text,
                         inference_pipeline=pipe,
                         label="v1_baseline")
```

Writes a timestamped JSON report to `eval_reports/`.

### Compare two checkpoints

```python
from evaluate import run_eval_suite, compare_checkpoints

ra = run_eval_suite(model_v1, tok, corpus, label="v1")
rb = run_eval_suite(model_v2, tok, corpus, label="v2")
print(compare_checkpoints(ra, rb, "v1", "v2"))
```

---

## Export to ONNX

```python
from model_io import export_onnx
export_onnx(model, "model.onnx", seq_len=128, vocab_size=128)
```

Then serve with ONNX Runtime:

```python
import onnxruntime as ort, numpy as np
sess  = ort.InferenceSession("model.onnx")
ids   = np.array([[65, 66, 67]], dtype=np.int64)
logits = sess.run(["logits"], {"input_ids": ids})[0]
```

---

## Python API

```python
from run import TransformerLM, CharTokenizer
from inference import InferencePipeline, GenerationConfig
from model_io import load_checkpoint

# Build
tok   = CharTokenizer()
model = TransformerLM(vocab_size=tok.vocab_size, d_model=128,
                      n_heads=4, n_layers=2, d_ff=512)
load_checkpoint(model, "best_model.pt")

# Generate
pipe = InferencePipeline(model, tok, device="cuda")
cfg  = GenerationConfig(strategy="top_p", temperature=0.8,
                        top_p=0.95, max_new_tokens=200)

# Single string
text = pipe.generate("Your prompt here", config=cfg)

# Streaming
for piece in pipe.stream("Your prompt here", config=cfg):
    print(piece, end="", flush=True)

# Batch
outputs = pipe.batch_generate(["Prompt A", "Prompt B"], config=cfg)
```

---

## File structure

```
45G/
├── greedy.py        Step 27 — greedy decoding
├── sampling.py      Steps 28-30 — temperature, top-k, top-p
├── inference.py     Step 31 — full pipeline
├── model_io.py      Step 32 — save / load / ONNX export
├── evaluate.py      Step 33 — perplexity, speed, memory, quality
├── run.py           Master entry point
├── test_45G.py      Full test suite (38 tests)
└── README.md        This file
```

---

## Connection to 45A–45F

`run.py` contains `TransformerLM` — a decoder-only transformer that
mirrors the architecture built in Steps 1–26. If you have the earlier
modules, swap the import:

```python
# In run.py, replace the built-in TransformerLM with:
from your_module import YourTransformerLM as TransformerLM
```

The only contract: `model.forward(input_ids)` returns logits of shape
`(batch, seq_len, vocab_size)`.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `CUDA out of memory` | Add `--device cpu` or reduce `--max_tokens` |
| `Unknown token in prompt` | Prompt uses characters outside the training vocab; they map to id=0 |
| Gibberish output | Model is untrained or undertrained — perplexity will be high |
| Very slow on CPU | Expected; a 128-dim model does ~50k tok/s on a modern CPU |
| `FileNotFoundError` on checkpoint | Check path; use `--list_checkpoints DIR` to see what's available |

---

## What comes next (Phase 2 suggestions)

1. **BPE tokenizer** — replace char-level with byte-pair encoding for a
   10–20× vocabulary reduction and far better generalisation
2. **Larger training run** — 10M+ token corpus, 256–512 d_model
3. **KV-cache** — cache key/value tensors across generation steps for
   5–10× inference speedup at longer contexts

---

*45G — final module. Builder: AN designate 45G.*
