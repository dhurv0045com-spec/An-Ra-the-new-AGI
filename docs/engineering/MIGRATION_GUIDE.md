# Migration to Core Executor 0.5.0

V4 neural mathematics are preserved. This is an execution/boundary repair, not
a claim that the trained checkpoint became smarter.

1. Load with `CoreExecutor.from_checkpoint`.
2. Use the exact V4 tokenizer contract bound to the artifact.
3. Create one state per homogeneous request/batch.
4. Prefill once, then use `forward_step` for incremental tokens.
5. Release the state on completion.

Core does not provide portable state serialization until it has a versioned and
validated schema. Use Connector code for sampling and streaming. Treat old
`Brain` helpers as reference Connector code, not permanent Core ABI.
