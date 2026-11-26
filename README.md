# Avendi V8-Y2, AN-RA V8-Y1 & Hitchhiker — Elendril mechanism investigation

**2026-11-27.** [Y5 foundation campaign](research/avendi/Y5_FOUNDATION.md) and [owner readiness and launch](research/avendi/Y5_READINESS.md).

**2026-11-22.** [Avendi V8-Y3 full 88/88 GPU results & audit](Experiment%20Result/Avendi-V8-Y3/y3-full-88of88/RESULTS.md). Clean single-session Kaggle 2×T4 run completed all `88/88` jobs in `4,132.65 s` (`50/50` studies complete, `79/88` jobs & `36/50` studies independently verified, verified `CONFIRM_LOCK.json`). `P4-categorical` achieves `1.00` depth `1..2` + causal interchange (`0.00` at depths `3..4`), `P5` `info` forks reach `0.7857` macro-optimal, and `P7-ark` isolates the cap stability-plasticity trade-off, while `P2`/`P3`/`P7-s20` miss the `0.80` qualification gate.

**2026-11-21.** [Avendi Y3 partial GPU audit](Experiment%20Result/Avendi-V8-Y3/y3-partial-57of88/RESULTS.md) covers 57 of 88 jobs. The margin objective improved four-cell development accuracy at all three seeds, but missed the registered improvement and row-order qualification gates.

**Next owner campaign.** [Avendi V8-Y3 executable Gemini handoff](research/avendi/GEMINI_AVENDI_V8_Y3.md) specifies 50 scientific studies and 54 main training jobs. The plan generator and exact reference fixtures are implemented and CPU-tested. The Y3 learned plugin and owner notebook are pending.

**Latest owner results (2026-11-19).** [Avendi V8-Y2 frozen audit](Experiment%20Result/Avendi-V8-Y2/avendi-frozen-63c10bbd2f/RESULTS.md). All six Y1 S3 checkpoints were restored in the owner run. Every checkpoint solved 0/256 complete one-hop and 0/256 complete three-hop families. F00 seed 7 chose the last rendered value on 1,020/1,024 one-hop cases. The next diagnostic permutes row order on the same facts before P3 acquisition.

**Original protocol and CPU intake (2026-11-18, branch `hitchhiker`).** [Avendi V8-Y2 registry](research/avendi/REGISTRY.json) (`design_sha256: 13f8f46c9065115b9bbdecb2884edba2ff88a3bf9e01cba2afd613739c344fbd`), [CPU P0 & frozen intake report](Experiment%20Result/Avendi-V8-Y2/avendi-frozen-0acbe87814/RESULTS.md), Planthir `ExperimentModule` adapter (`anra_v5.elendril.avendi.module`), and [AVENDI_V8_Y2_T4X2.ipynb](notebooks/AVENDI_V8_Y2_T4X2.ipynb) are implemented and CPU-qualified. P0 instrument (independent tokenized-input oracle, 10,920-family canonical 1-hop pool, destructive goal/bridge erasure, planted collision control, and 8 reference policies) is `QUALIFIED`; the November 19 audit records the subsequent P1/P2 owner run. P3–P6 remain pending.

**Earlier Y1 results (2026-11-18).** [V8-Y1 audited results](Experiment%20Result/V8-Y1/y1-all-9c80885013/RESULTS.md). Replayed 464,880 recorded episodes from 39 prediction jobs; no outcome mismatches, and the failed T1 gate reproduces. The 12-run S3 screen found no winner; the six-run S20 follow-up did not establish a capacity benefit. New fixture audits expose a perfect timestamp-rank shortcut and terminal-answer placement after queries. Repair those controls before interpreting new gains. Full Y1 checkpoint restoration remains unverified; source, evidence limits and next diagnostic are in the report. Earlier prospective text below is historical where superseded.

**Current prospective protocol (2026-11-16, branch `hitchhiker`):** [AN-RA V8-Y1 registry](research/v8-y1/REGISTRY.json) (`protocol_sha256: 3809785b27aa4feca1c0abfc8c81e9ed6dd69e48ff3c4bc0093481a647322cac`) and [Kaggle T4x2 Y1 notebook](notebooks/ELENDRIL_V8_Y1_KAGGLE_T4X2.ipynb) are implemented and CPU-verified; attaching `ELENDRIL_HITCHHIKER_FULL.zip` in `/kaggle/input` and clicking **Run All** (`PHASE = "auto"` -> `"all"`) automatically executes all three stages (`diagnose` across 15 restored Hitchhiker S3 checkpoints -> `core` T0/T1/T2 -> conditional `followup` T3) in one uninterrupted session with intermediate milestone exports and bounded 2-generation checkpoint rotation. GPU execution remains pending owner GPU runs. Historical `V7.17` S20-low-lr weights remain `NOT_AVAILABLE` and never block S3 work.

**Prior owner results (2026-11-15):** [Hitchhiker Stage 1 S3 results](Experiment%20Result/Hitchhiker/hh-stage1-4bfe3a806e3b/RESULTS.md) completed all 15 jobs; no intervention passed the screen. C seed 7’s apparent four-hop success follows a last-fact positional shortcut on the tested row-order cohort.

~~~text
python -m tools.elendril_y1_campaign --phase prepare --source <sha> --output artifacts/elendril/y1 --dry-run
python -m pytest -q tests/test_elendril_y1_data.py tests/test_elendril_y1_diagnostics.py tests/test_elendril_y1_training.py tests/test_elendril_y1_campaign.py tests/test_elendril_y1_export.py
python -m tools.elendril_y1_campaign --phase canary --source <sha> --output artifacts/elendril/y1-canary --cpu-canary
python -m tools.build_elendril_y1_notebook --source <sha>
~~~

Hitchhiker is an experimental branch for determining which mechanisms improve grounded lookup, variable binding, action selection, and multi-hop generalization. The repository-grounded design and full Hitchhiker experiment + campaign harness (7-factor counterfactual suite & structural holdouts, 4 health probes, Stage 0 260-way decoder audit & evidence sidecars, deterministic Stage 1–3 static/fresh/relabeled streams & trainer, episode-level fail-closed statistical gates, isolated 2-GPU worker & coordinator, dual bundle export/restore + verified selective Stage 2/3 handoff, and source-pinned [Kaggle T4x2 notebook](notebooks/ELENDRIL_HITCHHIKER_KAGGLE_T4X2.ipynb) ready for zero-input `Run All` Stage 1 execution) are implemented and CPU-mechanically verified; Stage 1 S3 has now run; S20 checkpoint mounting/inference and Stage 2–3 training remain pending.

- [User's original architectural proposal](research/hitchhiker/USER_ARCHITECTURAL_PROPOSAL.md), preserved verbatim from the supplied attachment.
- [Conclusion and experiment design](research/hitchhiker/CONCLUSION_AND_EXPERIMENT_DESIGN.md): code audit, refined mechanisms A–H, matched diagnostics, a 15-run initial screen, falsifiers, and implementation handoff.
- [Branch context and next steps](context.md): source identities, evidence limits, CLI/campaign commands, and the next execution sequence.

~~~text
python -m pytest tests/test_elendril_hitchhiker.py -v
python -m tools.elendril_hitchhiker_runner --mode verify-suite --worlds-per-hop 16
python -m tools.elendril_hitchhiker_runner --mode cpu-mechanical-check --worlds-per-hop 2 --output artifacts/elendril/hitchhiker/mechanical_verification.json
python -m tools.elendril_hitchhiker_campaign --phase stage1 --dry-run
python -m tools.elendril_hitchhiker_campaign --phase stage0|stage1|stage2|stage3 --owner-enabled --minutes 480 --output <dir>
python -m tools.build_elendril_hitchhiker_notebook --source <commit-sha>
~~~

**Conclusion:** isolate task interpretation, action legality, evidence binding, and composition before combining architectural changes. V7.17 demonstrates optimization sensitivity and transfer failures; it does not establish memorization as the sole cause or a capacity ceiling. Proposed improvements must survive changed facts, renamed entities, reordered evidence, and unsolvable cases.

**Source boundary:** this branch was created from `Elendril` at `2affffb806b7f73b00bc21aa90538019f37ce904` and now integrates the audited V7.17 code from `0c9cb284d2c8c949e17ac342959c6f2bf351bd30`. The documentation below summarizes the inherited V7.17/V7.1/V6 foundation, alongside the Hitchhiker evidence-producing experiment and mechanism design.

## Inherited ELENDRIL foundation

ELENDRIL studies cumulative, resource-efficient learning through one neural cognitive loop. Completed V7.1 and S8/S20 owner studies show narrow evidence dependence with brittle generalization and substantial seed variation. Historical implementations and ELD-DEV-001–004 evidence remain preserved. The [V7.17 owner study](Experiment%20Result/V7.17/README.md) demonstrates a matched lower-LR improvement while novel chain tasks still fail. Underlying causes of the remaining grounding/composition failure are unresolved; [the registered investigation](research/v7.17/README.md) separates initialization/order effects from capacity claims.

- [Current state](docs/CURRENT_STATE.md), [handoff](docs/HANDOFF.md) and [cognitive kernel](docs/COGNITIVE_KERNEL.md)
- [Foundation decisions](docs/FOUNDATION_AUDIT.md), [qualification](docs/FOUNDATION_QUALIFICATION.md) and [algorithm comparisons](docs/ALGORITHM_COMPARISONS.md)
- [Hypotheses](docs/HYPOTHESIS_REGISTER.md), [next cognitive experiment](docs/NEXT_COGNITIVE_EXPERIMENT.md) and [owner operations](docs/OPERATIONS.md)
- [Evidence custody](docs/EVIDENCE_AND_PROVENANCE.md) and [archived programs](docs/ARCHIVED_PROGRAMS.json)
- Preserved V7.1 core reproduction: [Kaggle two T4s](notebooks/ELENDRIL_V71_KAGGLE_T4X2.ipynb), [launch and recovery instructions](research/v7.1/README.md), [protocol](research/v7.1/PROTOCOL.md), [hypotheses](research/v7.1/HYPOTHESES.md) and [CPU data qualification](research/v7.1/DATA_QUALIFICATION.json).
- [Completed V7.1 core results](Experiment%20Result/V7.1/20c80f66e76e4b589aed2053afbddffc/RESULTS.md) preserve raw evidence, negative findings, measured costs and the [next paired diagnostic iteration](research/v7.1/NEXT_EXPERIMENT.md). The original core notebook remains a reproduction, not the new follow-up launcher.
- Completed V7: [preserved notebook](notebooks/ELENDRIL_V7_KAGGLE_T4X2.ipynb), [handoff](research/v7/HANDOFF.md), [protocol](research/v7/PROTOCOL.md), [historical evidence audit](research/v7/EVIDENCE.md) and [owner-run results](Experiment%20Result/V7/c9981309a94847a4a97d8115c56d75c8/RESULTS.md), documenting all nine runs and measured successes and failures.
- Preserved V6 campaign: [notebook](notebooks/ELENDRIL_V6_KAGGLE_T4X2.ipynb), [launch instructions](docs/V6_CAMPAIGN.md), [protocol package](research/v6/README_FIRST.md) and [implementation amendment](research/v6/IMPLEMENTATION_AMENDMENT_2026-11-07.md).
- Preserved V5 reproductions: [Colab one T4](notebooks/ELENDRIL_ACTION_COLAB_T4.ipynb), [Kaggle two T4s](notebooks/ELENDRIL_ACTION_KAGGLE_T4X2.ipynb).

Use Python 3.11+, the appropriate native PyTorch, and `python -m pip install -e ".[test]"`. Local hardware is for lightweight correctness tests and inference only.

~~~text
python -m pytest -q
python -m tools.verify_elendril_evidence
python -m elendril --help
python -m elendril inspect-checkpoint --root <checkpoint-root> --lineage <lineage>
python -m elendril verify-checkpoint --root <checkpoint-root> --lineage <lineage>
python -m elendril resume-dry-run --root <checkpoint-root> --lineage <action-lineage>
~~~

The owner-authorized V7 notebook launches with Run All; `RUN_STUDY = True` is already set. It automatically obtains source and generates data. Both physical GPUs must pass learning and same-session fresh-process recovery controls before training. Completion additionally requires all registered results and both verified current-run ZIPs. Fresh-VM recovery remains unqualified. No capability promotion follows from passing tests or completing a campaign. [License](LICENSE)
