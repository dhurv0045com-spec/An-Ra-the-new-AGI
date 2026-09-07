# W07 — Runtime, profiling and durable continuation

**Status:** CPU contract work ready; real backend integration depends on W01/W03 and available device. **Effort:** 3–5 hours. **Role:** systems/training engineer. **Compute:** CPU first; no TPU run without assigned live quota and explicit run allowance.

Read DATA_CONTRACTS.md checkpoint/batch sections and D08. Own `research/runtime/`, runtime tests and `engineering/reports/W07/`. Do not install or alter unrelated global environments.

## Deliverable

One tensor-owning update path with model, objective, backward, finite check, clipping, optimizer, schedule, sampler and checkpoint. Implement a CPU reference adapter and a capability-detected accelerator adapter appropriate to actual available hardware. Unsupported hardware must report unsupported, not produce a synthetic pass.

Include resource accounting, a global deadline, checkpoint reserve, and profiling that separates host preparation, transfer, compile, execution, evaluation and storage.

## Acceptance evidence

- An informative real batch changes expected model parameters and optimizer moments.
- The applied learning rate and token/episode counters match the declared schedule.
- Fresh-process CPU restore produces the same next sampled batch and update under the supported determinism contract.
- Corrupt/partial payloads and mismatched data/runtime identities fail without advancing committed state.
- Checkpoint state covers optimizer, schedule, RNG, sampler and replay, not just weights.
- Interrupted-run recovery uses the last complete checkpoint; total time includes recovery overhead.
- Actual accelerator tests, if run, record device topology/runtime and distinguish numerical tolerance from exact equality.
- A throughput report includes real/eligible targets and unique episodes, not only padded capacity tokens.

Do not spend this packet building a generic distributed platform. Verify one actual deployment topology first. Do not advertise remote durability until an artifact is restored from its external storage location after the original process/session is gone. If hardware is unavailable, deliver CPU correctness plus an explicit unrun accelerator checklist.
