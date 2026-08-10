# An-Ra agent operating policy

These instructions apply to the entire repository.

## Orchestration

- The primary agent owns architecture, scope, integration decisions, final verification,
  and communication with the repository owner.
- For concrete independent work, delegate bounded subtasks to available GPT-5.6
  subagents. Prefer the strongest available coding/reasoning agent (currently Sol) for
  critical implementation and Terra for contained audits, documentation, or routine
  engineering. Use a specifically requested agent such as Luna only when that agent is
  actually available in the active environment.
- Subagents inspect or implement only their assigned boundary. The primary agent reviews
  their evidence and integrates the result; subagent claims are not completion evidence.
- Parallelize work only when tasks are genuinely independent and concurrent edits will
  not collide. Do not create agents merely to simulate progress.

## Execution priorities

- Prefer completing operational code over expanding speculative planning documents.
- Preserve existing user work and unrelated dirty-tree changes.
- Treat V4 as the only operational tokenizer and the 181M dense model as the canonical
  foundation until a signed growth promotion is approved.
- Keep experimental architecture behind explicit pilot gates. A subsystem is not active
  or healthy merely because its files exist.
- Never confuse model parameters, optimizer steps, tokens processed, or SFT steps.
- Never continue foundation pretraining from an SFT child. Preserve distinct checkpoint
  namespaces and lineage.

## Verification budget

- Use the smallest focused checks that establish the changed contract. Do not repeatedly
  run the full suite, long model inference, or training jobs after ordinary edits.
- Tests around checkpoint integrity, exact resume, token-window uniqueness, sampler
  continuity, signing, safe loading, and destructive operations are mandatory when those
  areas change.
- Before cloud training, prefer static validation plus one short canary. Large training
  runs require explicit owner authorization and must protect a resumable checkpoint first.
- Report exactly what was tested, what was not tested, and any remaining operational step.

## Training and external systems

- Only one canonical trainer may advance a checkpoint lineage at a time. Separate Colab
  or Kaggle machines exchange protected checkpoints sequentially; they do not synchronize
  gradients over the public internet.
- Never start paid compute, publish a public artifact, change sharing permissions, or
  delete remote data without explicit owner authorization.
- Keep credentials out of Git, logs, notebook outputs, reports, and prompts. Prefer the
  provider's secret store; a private key file is a compatibility path only when the owner
  explicitly accepts that trust boundary.
- A cloud session is not complete until the resulting full-resume checkpoint, metadata,
  and digest are durably accessible outside the disposable worker.

## Completion standard

- Lead with the actual outcome and distinguish implemented, verified, proposed, and
  owner-required steps.
- Update concise operational documentation when a user-facing workflow changes.
- Do not call the repository, model, subsystem, or AGI complete without behavioral and
  operational evidence appropriate to that claim.
