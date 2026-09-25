# Cymek: learning, measurement, and transfer

Cymek studies how a model acquires a capability, whether that capability survives a change in wording or representation, and whether it remains after further learning. This repository is not trying to turn a generic chatbot into AGI by adding impressive-sounding parts. It is building experiments that can show a narrow idea worked, failed, or remains unresolved.

The An-Ra V5 model and training contracts give the experiments a stable subject. The research changes the questions, tasks, measurements, and controlled interventions around that subject, so results can be compared without changing everything at once.

## The task underway now: METRIC-RES-001

The next action is a **checkpoint-only audit of 16 preserved S5 model checkpoints**. It asks whether the S5 model had partial identity/copy skill that the all-or-nothing exact-match score could not see, or whether that skill was genuinely absent. The audit does **no training**: it does not construct an optimizer, run backward passes, or change checkpoint weights.

S5 completed all 24 planned arms, but its multi-token identity scores were at the floor. The corrected interpretation is `INCONCLUSIVE_AT_ZERO_BASELINE`: the mechanism comparisons had too little signal to say whether the treatments helped or hurt. At the same time, one-token termination reached 1.00. That contrast makes measurement resolution the cheapest important question to answer next.

METRIC-RES-001 scores preserved checkpoints with token-level accuracy, longest-common-prefix, per-position diagnostics, and a termination positive control. Its preregistered decision determines what to investigate next; it cannot rewrite S5’s result or authorize a larger training run.

Use the [Kaggle T4 x2 notebook](notebooks/CYMEK_METRIC_RES_001_KAGGLE_T4X2.ipynb), preferably, or the [Colab T4 alternative](notebooks/CYMEK_METRIC_RES_001_COLAB_T4.ipynb)—run one, not both. Attach the original S5 Kaggle Output containing the 16 `resume.pt` files. They are not in the S5 results ZIP. The [experiment instructions](docs/cymek/experiments/METRIC-RES-001/README.md) explain the required input layout and frozen decision rules.

## What is active alongside it

**Signac** is the separate 100M-class engineering lane. Its M102 reference model has exactly 101,790,080 parameters. The [Kaggle TPU notebook](notebooks/SIGNAC_100M_KAGGLE_TPU_PREFLIGHT.ipynb) is prepared for bounded synthetic qualification, but no Kaggle TPU receipt is recorded. Production XLA, target memory and throughput, exact distributed recovery, qualified research data, and an independent evaluator remain unqualified. A successful TPU canary would show engineering readiness, not a Cymek result or permission to begin Signac research training.

Keep the platforms straight: METRIC-RES-001 is a short **T4 GPU checkpoint audit with no training**; Signac is **synthetic TPU engineering qualification**. Neither substitutes for the other. The [current research queue](docs/cymek/research/NEXT_7_EXPERIMENTS_V51.md) defines what follows each Cymek audit outcome, and [Signac readiness](docs/signac_100m/READINESS.md) defines its separate hardware and research gates.

## How the research works

The research loop is simple to state and strict to run: **freeze a question → use identified checkpoints or a fixed training recipe → change one condition at a time → measure the behavior directly → preserve the evidence → choose the next test from the result.**

Generated cognition exercises and recorded experiment findings are kept as distinct, provenance-labeled development streams. Evaluation is candidate-free: the model must produce its own answer. Exact correctness, token-level behavior, valid termination, transfer, and retention are measured separately. Development materials help debug instruments; they do not qualify a production corpus or count as sealed evidence.

For the design decisions—what the system reuses, what changed after S5, and how training, evaluation, and recovery fit together—read [OTHER.md](OTHER.md). The [Cymek queue](docs/cymek/research/NEXT_7_EXPERIMENTS_V51.md) and [Signac training plan](docs/signac_100m/TRAINING_PLAN.md) are the operational references for their respective tasks.
