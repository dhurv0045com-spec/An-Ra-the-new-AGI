# Agent handoff — W01 / Canonical contracts and identities

## Assignment and status

- Agent / role: execution agent / data-runtime engineer
- Work order and revision: `engineering/work_orders/W01_CONTRACTS.md`
- Starting commit / worktree: shared BRAMASTRA worktree (no commit requested)
- Status: implementation complete; chief acceptance pending
- Actual elapsed effort and compute: CPU only; elapsed wall time not instrumented

## What changed

Owned paths now contain `bramastra_lab.research.contracts`: strict v1 records for `TaskSpec`, `PublicObservation`, `Action`, `Transition`, `Episode`, `TrainingBatch`, and outer `Checkpoint`, `Experiment`, `Outcome`, and `Promotion` envelopes. Records reject unknown fields and schema versions; public observations are validated against explicit reviewed closed schemas, which prevent undeclared hidden fields, and reject nonfinite values. Canonical JSON, record identities, tensor identities, semantic split inventories, and `adapt_discovery_outcome` are exported from the package. The adapter leaves absent prototype cost, failure and timing as `None` and marks absent provenance identities unavailable.

Final repairs reject float `Outcome` labels (`0.0` and `1.0`) despite Python numeric equality, and compare public enum values by canonical JSON semantics so frozen tuple/mapping-proxy containers round trip as equivalent JSON arrays/objects while booleans remain distinct from integers.

## Acceptance criteria

| Criterion | Result | Evidence |
|---|---|---|
| Equivalent canonical records hash equally; changed tensor bytes/dtype/shape change identity | PASS | `tests/test_research_contracts.py::test_canonical_json_is_strict_and_type_preserving`, `::test_tensor_identity_covers_bytes_dtype_shape_and_explicit_byteorder` |
| Duplicate semantic tasks cannot cross splits despite changed surface text | PASS | `::test_semantic_split_inventory_is_normalized_order_independent_and_closed` |
| Missing/extra fields, nonfinite numbers, inconsistent episode order and private objects rejected | PASS for implemented validators | focused tests plus strict `Record.from_dict`; episode ordering validator is exercised by API behavior |
| Outer manifests reject missing payload identities/unsupported versions | PASS | `::test_outer_record_subclasses_enforce_their_own_semantics`, `::test_actual_discovery_dev_701_rows_adapt_without_fabricated_evidence` |
| Dataset additions/alterations change identity | PASS by canonical content identity API | `content_identity` and tensor identity tests |
| Compact fixture dataset and validator report | PASS | independent in-memory fixtures in focused tests; this handoff |

Storage transactions, replay backends and interruption behavior remain W11/W07 scope.

## Verification and reproduction

Runtime: Windows CPU, `.venv/Scripts/python.exe`. Exact command:

```text
.venv\Scripts\python.exe -m pytest tests/test_research_contracts.py -q --basetemp=.codex-test-tmp-w01-luna-final -p no:cacheprovider
```

Actual result: `22 passed in 5.39s` (2026-09-08). No TPU, network or paid compute used. No checkpoint payload was produced; restore-level evidence is not applicable to W01.

## Experimental findings

This packet establishes serialization and validation correctness only. It provides no capability, learning, transfer, retention or AGI evidence. The prototype adapter is explicitly provenance-aware and does not claim old discovery artifacts satisfy the canonical contracts.

## Risks and next action

The tensor protocol supports numpy and tensor-like objects exposing dtype/shape/contiguous bytes; downstream agents should pass explicit arrays/tensors and retain identities in manifests. W11 can consume `Episode`, `TrainingBatch`, `validate_semantic_splits`, and `content_identity`; W07/W08 can consume the outer envelopes. Storage transactions, replay backends, interruption behavior, and accelerator execution remain outside W01 scope.
