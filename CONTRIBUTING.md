# Contributing to AN-RA

AN-RA is a sovereign AGI research platform. These guidelines keep it
coherent as it grows.

## Setup

```bash
git clone https://github.com/your-org/An-Ra-the-new-AGI
cd An-Ra-the-new-AGI
make install
make verify
make test-fast
```

For memory and ML extras (sentence-transformers, FAISS):

```bash
make install-ml
```

## Before Every Commit

```bash
make lint
make typecheck
make test
```

## Architecture Rules

**Registry pattern - mandatory for all new components.**
Any model, memory tier, training algorithm, inference strategy, or identity
module must be registered:

```python
from anra.core.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register("my_new_model")
class MyNewModel(nn.Module): ...
```

**No sys.path manipulation.**
The project is installed with `pip install -e .`. All imports use package paths.
Never add path mutation calls.

**No bare imports.**
Use full package paths:

```python
# WRONG
from model import CausalTransformerV2

# RIGHT
from anra_brain import CausalTransformerV2
from anra.core.model import CausalTransformerV2
```

**No wildcard imports.**
`from X import *` is banned everywhere.

## Adding a New Component

1. Write the implementation in the appropriate module.
2. Register it: `@REGISTRY.register("name")`.
3. Write tests in `tests/test_<component>.py`.
4. Add it to the research roadmap doc if it is a new research direction.
5. Run `make test` to confirm nothing broke.

## Research Vision Lock

Every contribution must preserve:

- Owner-shaped data as center of gravity
- CIV, ESV, HAL, and sovereignty gates protecting identity
- Verifier-backed tasks (not just narrated)
- Memory, replay, and falsification as first-class mechanisms
- Everything registered, switchable, measurable, and testable

See `docs/research/ANRA_BEST_RESEARCH_FOR_INTELLIGENCE_AND_EFFICIENCY.md`
for the ranked technology roadmap.

## Commit Message Format

    type(scope): short description

    Types: feat, fix, test, docs, refactor, perf, chore
    Examples:
      feat(training): add DAPO variant to RLVRTrainer
      fix(model): correct gradient checkpointing closure
      test(memory): add router integration tests
      docs(research): add SparseLoRA fit analysis
