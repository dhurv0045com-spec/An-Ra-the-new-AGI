# An-Ra Phase 3 Integration

Phase 3 is the deep-cognition band of the current `19/19` stack:

| Layer | Component | Role |
| --- | --- | --- |
| 08/19 | 45N Identity | Runtime identity injection and response cleanup |
| 16/19 | 45O Ouroboros | Recursive reasoning and milestone reflection |
| 17/19 | 45P Ghost Memory | Compressed long-memory retrieval |
| 18/19 | 45Q Symbolic Bridge | Verified math, logic, and code reasoning |
| 19/19 | 45R Sovereignty | Audit, benchmark, report, and promotion governance |

## Integrated Pipeline

```text
user prompt
  -> 45Q symbolic pre-check when the prompt is math/logic/code
  -> 45N identity context
  -> 45J memory context
  -> 45P ghost context
  -> 45O adaptive reasoning for hard prompts
  -> generation runtime
  -> 45N cleanup
  -> memory write
  -> sovereignty/audit artifacts when scheduled
```

## Unified Commands

From the repo root:

```bash
python anra.py --phase3-status
python anra.py --symbolic "solve x^2 - 9 = 0"
python anra.py --sovereignty-report
python anra.py --sovereignty-run
python scripts/status.py
```

## Module Status

| Component | Source Status | Runtime Notes |
| --- | --- | --- |
| 45N Identity | Active | Falls back if identity file is absent |
| 45O Ouroboros | Active | NumPy path is CPU-friendly; Torch path is heavier |
| 45P Ghost Memory | Active | Mock embeddings work without downloads; real embeddings are optional |
| 45Q Symbolic Bridge | Active | Best with `sympy`, `scipy`, and `numpy` installed |
| 45R Sovereignty | Active | Best with `psutil`; local daemon/API only |

## Import Reality

The Phase 3 folder names contain spaces and layer labels. For direct experiments, run from inside the component folder or use `anra.py`, which injects the relevant paths.

Examples:

```bash
python anra.py --symbolic "factor 360"
cd "phase3/symbolic_bridge (45Q)" && python demo.py
cd "phase3/ouroboros (45O)" && python test_ouroboros.py
```

## Checkpoint Story

The current V2 checkpoint family is:

```text
anra_v2_brain.pt
anra_v2_identity.pt
anra_v2_ouroboros.pt
```

On a fresh local clone those files may be missing. That means the source stack is present but trained artifacts need to be restored or produced.

## Design Rule

Phase 3 exists to deepen the mainline, not to replace it. Symbolic verification, ghost recall, identity, reflection, and sovereignty should feed the canonical runtime and training loop rather than forking a separate product path.
