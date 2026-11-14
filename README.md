# Hitchhiker — Elendril mechanism investigation

Hitchhiker is an experimental branch for determining which mechanisms improve grounded lookup, variable binding, action selection, and multi-hop generalization. The repository-grounded design and full Hitchhiker experiment + campaign harness (7-factor counterfactual suite & structural holdouts, 4 health probes, Stage 0 260-way decoder audit & evidence sidecars, deterministic Stage 1–3 static/fresh/relabeled streams & trainer, episode-level fail-closed statistical gates, isolated 2-GPU worker & coordinator, dual bundle export/restore, and [Kaggle T4x2 notebook](notebooks/ELENDRIL_HITCHHIKER_KAGGLE_T4X2.ipynb)) are implemented and CPU-mechanically verified; real GPU qualification, S20-low-lr checkpoint mounting/inference, and Stage 1–3 GPU training runs remain pending.

- [User's original architectural proposal](research/hitchhiker/USER_ARCHITECTURAL_PROPOSAL.md), preserved verbatim from the supplied attachment.
- [Conclusion and experiment design](research/hitchhiker/CONCLUSION_AND_EXPERIMENT_DESIGN.md): code audit, refined mechanisms A–H, matched diagnostics, a 15-run initial screen, falsifiers, and implementation handoff.
- [Branch context and next steps](context.md): source identities, evidence limits, CLI/campaign commands, and the next execution sequence.

~~~text
python -m pytest tests/test_elendril_hitchhiker.py -v
python -m tools.elendril_hitchhiker_runner --mode verify-suite --worlds-per-hop 16
python -m tools.elendril_hitchhiker_runner --mode cpu-mechanical-check --worlds-per-hop 2 --output artifacts/elendril/hitchhiker/mechanical_verification.json
python -m tools.elendril_hitchhiker_campaign --phase stage1 --dry-run
python -m tools.elendril_hitchhiker_campaign --phase stage0|stage1|stage2|stage3 --owner-enabled --minutes 540 --output <dir>
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
