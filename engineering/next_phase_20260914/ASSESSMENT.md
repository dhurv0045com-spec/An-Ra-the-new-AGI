# Branch assessment: 44e3905

The chief fetched BRAMASTRA and verified local/remote agreement at 44e3905. Reviewed the real-execution handoff, phase operations, E1/E2/E5 code, runner target construction and checkpoint APIs. A bounded Luna review was requested but hit its usage limit before returning findings; this assessment is the chief's own source review. The chief ran:

```powershell
python -m pytest tests/test_research_k8_real.py tests/test_research_k8_readiness.py tests/test_research_k8_launch_gate.py -q -p no:cacheprovider -o addopts=
```

Result: **14 passed in 3.99 seconds**. These selected tests use fixtures and non-learning operations; no optimizer updates or accelerator work were performed. The agent's broader 84-test and verification-script claims were not independently rerun in this review. Passing the selected tests is evidence for their covered contracts, not campaign acceptance.

## Accepted direction

The branch now implements real checkpoint publication APIs, exact parent lookup, namespace separation, stricter fixture rejection and dataset-derived objective channels. Earlier counterexamples that accepted missing parents or metadata-only checkpoint claims are addressed by corresponding validation paths. Retain these changes and their regressions. No wholesale rewrite is requested.

## Integration still missing

- **Reservation propagation:** ProductionOps._build_trainer sets require_allocation=True. initialize_random stores reservation on the handle but does not call trainer.begin_campaign. E1 does not supply a reservation, and the production phase operations have no completed admission bridge. The first actual finalize is therefore not an authorized campaign update. Writing that the GPU must bind a reservation is an implementation assignment, not a hardware check.
- **Calibration:** runner.py assigns fixed update targets 200/80 and E2 case count 32. It does not consume a frozen protocol produced from measured E0 throughput. Minimum admissible values are not substitutes for the prespecified calibration rule and cost-aware selection.
- **Cognitive treatment:** ProductionOps.evaluate_episode performs one free generation and returns one action/call/node; it does not use task_env to run the required interaction loop. E2 assigns mode labels to different cases rather than running the prescribed paired treatments on the same mechanism clusters. Control parent references are resolved but the B handle is used for the common evaluation loop. Prompt construction truncates public text and tokens; the goal-swap metadata changes after the prompt is built. These do not implement information gathering, working-memory use or planning comparisons.
- **RSI:** _measure_trial has a fixture measurement path and an unconditional refusal for production. Production learner handles are dictionaries too, so they encounter the missing method-sensitive-double trainer branch. No real adaptation/evaluation path appears after that refusal. _capture_proposer_choice computes a host-side majority of best methods and packages it as raw_output; it never invokes the proposer decoder. P0 archive learning and P1/P_fixed successor training are still absent. The code can honestly refuse, but cannot yet run E5.
- **Evidence admission:** a learned checkpoint alone does not turn a scripted decision path into learned cognition. Evidence origin must come from the actual operation/session manifest and persisted execution trace, not checkpoint-name prefixes or the absence of a fixture marker. Keep fixture rejection and complete its provenance chain.

These findings justify keeping implementation readiness false. The next phase targets these exact dependencies while retaining the broader cognitive/RSI objective. No new experiment is needed to discover that these consumers are absent; implement them and exercise their production boundaries locally before requesting owner accelerator time.
