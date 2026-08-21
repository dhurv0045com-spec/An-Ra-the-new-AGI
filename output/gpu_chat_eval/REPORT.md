# Local GPU Evaluation — An-Ra V4, step 20,000 vs step 30,400

Device: RTX 4050 Laptop (6 GB), torch 2.11.0+cu128, fp32 exact profile.
Discipline: sequential execution — load → evaluate → close → free
(`gc` + `empty_cache` + `synchronize`) → verified reserved VRAM (710→24 MiB
and 730→44 MiB) → 15 s spacing → next model. Raw evidence: `report.json`,
`cpu_reference.json`, `divergence.json`, `run.log`. Harness:
`connector/experiments/gpu_chat_eval.py` (executor-validated prefill +
KV-cached incremental decode, BOS-bound).

## Exact errors found by reverse-engineering

1. **CUDA executor bug (fixed, regression-locked)**: `CoreExecutor` stored
   `torch.device("cuda")` (index-less) while cache tensors report `cuda:0`;
   `torch.device("cuda") != torch.device("cuda:0")`, so the first
   `forward_step` after `prefill` raised `state cache storage device drifted`.
   CPU never trips this (index-less both sides); no prior test exercised CUDA.
   Fix: pin the device index in `CoreExecutor.__init__`.
2. **The earlier GPU evidence (`output/ckpt_eval/`) is invalid beyond token
   #1**: its decode loop re-runs `model(next_token)` with a single token and
   no KV cache — every step after the first sees a 1-token context, forcing
   "the the the" / "ditj ditj ditj" degeneracy. The model is weak, but not
   *that* weak. This report supersedes it.
3. Step-20k artifact required the legacy identity binding fixed earlier
   (old-format tokenizer contract, byte-identical vocabulary).

## A — Chat (H:/ANRA: dialogue format, greedy, 20 new tokens)

| Probe | step 20k | step 30.4k |
|---|---|---|
| greeting | " I have been working on a project that I have" | " Hello! How are you today?\nH: Hello! How…" (echolalia) |
| name question | " I have a name that is named after the name" | " Hi. My name is Hi. My name is Hi…" (loop) |
| multi-turn (2nd answer) | " Apples are a healthy fruit.\nH: What color…" | " Apples contain apples.\nH: What color are apples?" |
| persona | " I am a student and I am a student." | " Know NowickiH: Know AquariiH: Know terrarium…" (token salad) |
| reply-one-word | " Reply with one word: hello\nTags: []…" (echoes frame) | " Reply With dermatan: Reply With Aquarii…" |
| capital of France | " The capital of France is the capital of France." | " France lies mostly within France. France…" |

**20k**: grammatical, on-format, conversational English; correct dialogue
turn-taking (it even simulates the next `H:` turn); no real self-identity or
knowledge; instruction extraction fails (echoes the instruction instead of
the answer). **30.4k**: form survives, content degrades — prompt-echolalia,
rare-token attraction, repetition loops.

## B — Continuation

20k: fluent webtext (" located in the city of Lancashire."), code = syntax
noise, math degenerate (" 0.000000 + …"). 30.4k: rare-token salad
(" had Travertine popitem popitem Aquarii Glebes"), the syllogism fails at
both steps (continues the pattern, never concludes "an animal").

## C — Both vocabularies/protocols (nonce items)

| Item | 20k NL | 20k tag | 30.4k NL | 30.4k tag |
|---|---|---|---|---|
| nonce knowledge (MAV-731) | near-miss: echoes fact, "MAV-" prefix | fail | fail | fail |
| verbatim echo (quartz) | **pass** | fail | fail | fail |
| tool result (42) | echoes "42\nTOOL OUTPUT: 42…" (copies frame) | fail | fail | fail |

NL strictly dominates tags at every step. The structured protocol
(`<k>…<q>…<answer>`) is out-of-distribution: next-token entropy after a tag
prompt is 4.9 (20k) / 7.0 (30.4k) vs 0.09–0.47 after natural prompts —
the model is near-certain in-distribution and lost in the tag grammar.
The one skill 20k has (copying in NL) is destroyed by 30.4k.

## D — Tool calling

No checkpoint emits a callable tool request in any tested convention
(`<tool>` tag, chat request, JSON). Given a supplied `<tool_output>42`,
20k-NL copies the surrounding frame ("42\nTOOL OUTPUT: 42") — the value is
visible but not extracted as an answer; 30.4k fails outright.

## E — Sampling health (3 samples, chat prompt)

| temperature | cross-sample distinct (20k) | (30.4k) |
|---|---|---|
| 0.0 | 0.27 | 0.25 |
| 0.4 | 0.71 | 0.59 |
| 0.8 | **1.00** | **0.57** |
| 1.2 | 0.91 | 0.94 |

The 20k distribution is healthy (full diversity at t=0.8). At 30.4k
diversity at t=0.8 collapses to 0.57 — the continuation sharpened the
distribution into repetition basins. Objective sign of training damage.

## F — Next-token forensics

Natural prompts: top-1 probability 0.96–0.999 (entropy 0.01–0.47) at both
steps — the " ` ` " token after a period dominates. Tag prompts: entropy
4.9→7.0 (20k→30.4k). The failure mode is not uncertain gibberish in NL; it
is overconfident continuation that never re-attends to earlier context.

## Verdict — exact deltas

- **Improvements 20k → 30.4k**: none measured. Every metric degrades or ties.
- **Degradations**: echo-NL 1→0 (probe), chat content quality, repetition
  (rep-rate up on 5/8 chat probes), sampling diversity at t=0.8 1.00→0.57,
  rare-token salad, tag-prompt entropy 4.9→7.0.
- **Training implication**: continuation past 20k was net-harmful. Restart
  from step 20k; change the regime (the LR was constant 2e-4, no decay —
  WSD decay is the candidate fix) and gate promotion on this battery.
- **Best current model**: step-20k artifact, natural-language protocol only.

## Note on "both vocabularies"

Both checkpoints embed a **byte-identical tokenizer contract**
(`vocabulary_sha256`, `probe_sha256` verified equal) — the checkpoints do
NOT differ in vocabulary. All measured differences are weights/training.
"NL vs tags" above is the protocol comparison.
