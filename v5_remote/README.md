# v5_remote — off-host execution contracts

Local GPU/CPU training compute is **out of scope** for this package and for the
`cymek` branch. This package only freezes *what* a remote accelerator host must
run and verifies *that* its answer belongs to that request.

## Flow

1. Freeze a `RemoteJob`: accelerator shape, replica count, pinned runtime-image
   hash, exact code commit, command vector, seed, token slice, wall-clock
   limit, and the six evidence-identity slots from `blueprint/LAUNCH_GATES.json`
   (`job_spec.IDENTITY_KEYS`).
2. Transport `submission_envelope(job)` to the remote host out of band. Job
   submission, credentials, and storage live outside this repository.
3. The remote host runs the pinned command (an `anra-v5 ...` operator command
   against the pinned commit) and returns a `RemoteResult` plus the receipt
   files it names.
4. `bind_result(job=..., result=...)` verifies the result's job hash equals the
   frozen request hash, then emits a hash-bound binding receipt. A swapped,
   replayed, or edited answer raises instead of satisfying a gate.

## Rules

- Unknown schemas, unpinned images, empty commands, and identity substitution
  fail closed (`ValueError`, no best effort).
- A `succeeded` result must present at least one receipt hash; a `failed`
  result must name a compact failure code. Both always carry the remote log
  hash so failures stay auditable.
- This package imports only the Python standard library. It never imports a
  research plane, a trainer, an evaluator, or a Connector runtime; enforcement
  lives in `v5_contracts.import_boundaries`.

## Status

Contracts only. No job has been submitted and no result collected from this
branch; nothing here has executed on any accelerator, local or remote.
