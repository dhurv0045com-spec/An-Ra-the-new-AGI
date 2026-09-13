# CS-TRANSFER-001 Engineering Repair E1 — qualification fixture feasibility

Status: **PRE-EXECUTION ENGINEERING ONLY**. No scientific GPU arm had started when this repair was issued.

Frozen scientific protocol ancestor: `a916d1c8d2637abb86d16b1c78e418c95461f3c7`.

## Trigger

The live Colab CPU qualification failed in `tests/test_cs_transfer_001.py` before Drive mount or any scientific update. Static audit of the frozen test showed that its synthetic `_LowIdTokenizer` fixture selected 30 development rows per family from only 300 candidate worlds per family. The canonical split allocates 20% of candidate worlds to development, so that fixture offered only 60 development candidates. The data contract permits an eligibility rate as low as 15%; therefore only 9 eligible development rows are guaranteed, while the test demanded 30. The fixture could fail from its own sampling geometry even when the production data contract was correct.

The same issue affected the round-trip test, which unnecessarily rebuilt a screened dataset even though serialization round-trip is a local `TokenRow` property.

## Repair

- Increase the deterministic shared-surface qualification fixture to 1,000 candidate worlds per family while retaining 60/30/30 selected rows per family. At the frozen 60/20/20 split and the required 15% minimum eligibility, this guarantees capacity for 60 training and 30 development/sealed rows whenever the acceptance gate itself passes.
- Replace the serialization round-trip dataset build with a direct representative `TokenRow` round-trip fixture.
- Add an explicit arithmetic assertion documenting why the synthetic qualification fixture is feasible under the protocol floor.

## Non-change

No file under `anra_v5/`, `v5_model/`, `v5_training/`, `v5_data/`, `v5_contracts/`, or `experiments/CS_TRANSFER_001/` is changed by E1. Model geometry, matched initialization, tokenizer artifact, data generator, protocol Amendment 1, candidate filtering, train/dev/sealed identities, optimizer, schedule, 480-update endpoint, decision thresholds, checkpoint semantics, and sealed-test policy remain unchanged.

This repair changes qualification code only. It does not waive the data acceptance/shortcut/contamination gates; the real `prepare` step must still pass them before CUDA training.
