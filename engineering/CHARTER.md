# Engineering charter

Owner direction recorded: 2026-09-07. Objective: AGI trained from scratch. Chief-engineer role: architecture, algorithms, experiments, data structures, learning methods, delegation and technical acceptance. Execution is delegated to implementation agents.

## What success means

The long-term objective is broad competence, learning unfamiliar tasks with limited experience, effective investigation and planning, and repeated durable improvement. No known engineering checklist is sufficient to guarantee that outcome.

Near-term success is an independently reproducible experimental result that resolves an important uncertainty in that objective. For example: a learned inquiry policy improves success on held-out mechanisms at the same interaction budget, and its selected experiences produce a child model that transfers better without unacceptable forgetting.

Do not quietly replace AGI with a calculator, a symbolic solver, a collection of wrappers or a task-specific benchmark. Those can be instruments or components. State their scope.

## Roles

**Chief engineer:** selects mechanisms worth testing; freezes interfaces and comparisons; resolves ownership; assesses budgets; reviews evidence; accepts or rejects claims; integrates accepted work. The chief writes specifications and review decisions, not the bulk of implementation requested from execution agents.

**Implementation agent:** owns one package and its tests; identifies contradictions early; implements the simplest correct version that answers the assigned question; runs bounded authorized experiments; submits a reproducible handoff.

**Independent reviewer/examiner:** checks leakage, controls, provenance and statistical interpretation; does not tune the candidate using confirmation answers. Independence means restricted information and separate responsibility, not merely a second model name.

## Compute and time

Approximately 100 free Kaggle TPU-hours/week is owner-reported. Verify actual quota/topology before allocating runs. No paid compute is presumed. Count compilation, failed runs, evaluation, upload and restore against the allowance. Use CPU for generators, semantic checks, statistical analysis and small pilots where practical.

A 3–5-hour work estimate describes implementation/review effort with prerequisites available. It is not a required training duration, deadline guarantee or reason to manufacture lines of code. A packet that cannot finish within the estimate should stop at an explicitly useful checkpoint and report remaining work.

## Claims ladder

1. **Specified:** an actionable design exists.
2. **Implemented:** runnable code exists, with known limitations.
3. **Correctness-checked:** focused tests and invariants pass.
4. **Operationally verified:** the actual backend performs and restores the intended computation.
5. **Experimentally supported:** the declared comparison supports a bounded conclusion.
6. **Replicated:** fresh seeds/tasks reproduce the conclusion under declared conditions.
7. **Broadly validated:** independently authored tasks establish broader competence.

AGI is not an automatic eighth checkbox. Broad claims require much stronger evidence than this initial program can provide.

## Decision discipline

Record the question before running the experiment. Define the primary metric, control, sampling unit, budget and failure conditions. Exploratory experiments may guide design; label them development evidence. Freeze a separate confirmation protocol after pilot calibration. A negative result can reject a hypothesis; a broken objective or leaked target cannot.

Keep raw outcomes, rejected candidates and consumed budget. Do not select only the best seed, shift the target threshold after observing results, or call source-code automation learned research ability.
