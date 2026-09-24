# CS-TRANSFER-001 Operator Amendment 2 — Colab validator invocation

Status: OPERATOR-ONLY. Scientific executable remains frozen at `a916d1c8d2637abb86d16b1c78e418c95461f3c7`.

## Trigger

On a clean Colab runtime, notebook cell 1 verified the exact science commit and all 16 critical Git blobs, then failed while invoking:

`python tools/validate_cs_transfer_001.py`

The validator imports `tools.next_core_compute_model`. Direct script execution sets Python's import root to the `tools/` directory, so the top-level `tools` package/namespace may not resolve. This is an operator invocation defect, not a model, data, protocol, or scientific-validation failure.

## Repair

Invoke the unchanged frozen validator as a module from repository root:

`python -m tools.validate_cs_transfer_001`

The canonical T4 notebook is updated accordingly. CPU tests remain mandatory and fail closed. The notebook also captures per-test stdout/stderr so any subsequent qualification failure names the exact test.

## Scientific non-change

No scientific source blob, preregistration, Amendment 1, model geometry, matched initialization, data-selection rule, optimizer, schedule, endpoint, evaluation metric, sealed firewall, or decision threshold changes. No Drive scientific state had been created by the triggering failure because it occurred before Drive mount/preparation.
