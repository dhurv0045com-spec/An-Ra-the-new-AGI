# FAILURE MODES (canary execution)

| Mode | Signature | Detection | Likely cause | Cheapest diagnostic | Rollback |
|---|---|---|---|---|---|
| pack-stream shortfall | `FAIL_CLOSED: requested updates exceed the frozen pack stream` | runner update-index bound | dataset smaller than the token budget (CAUGHT: amendment 1) | tokens/epoch vs requested updates | amend data volume with documented amendment; never lower thresholds |
| NONFINITE_LOSS / NONFINITE_GRAD | backend abort before state advance | `certify_real_update` every update | LR too high; numeric bug | loss curve + grad-norm trace | restore last-known-good; investigate before rerun |
| CLIP_BREACH | post-clip norm > 1.0 + 1e-4 | clip certificate every update | reduction-order drift; clip bug | fused vs per-tensor norm diff (R1C precedent) | single-sourced tolerance in `v5_training/step.py`; kernel-determinism fix |
| SCHEDULE_DRIFT | lr_expected != lr_actual in WSD_TRACE | per-update trace check | rewarm on resume; schedule re-derivation | trace rows around resume boundary | resume must continue token-indexed position (verified bitwise) |
| TOKEN_DRIFT | ledger != consumed real tokens | window ledger cross-check | packing/sampler bug | audit receipt | fail closed; repack |
| resume divergence | model/optimizer bytes differ after fresh-process resume | bitwise artifact comparison | ambient RNG leak (CAUGHT: torch process-start generator non-deterministic → pinned via `torch.manual_seed(seed)`); missing state restore | field-by-field state diff | pin ambient RNG; restore full inventory from durable bytes |
| stale/incompatible checkpoint | store validation error or identity drift at scan | LATEST fence + identity bindings | wrong executable; wrong tokenizer; partial publication | identity diff | FAIL_CLOSED; operator resolves; never silently load |
| partial publication | staging directory present after crash | store invariant | injected/real crash mid-publish | staging existence check | resolve staging (documented recovery); republish; last-known-good preserved |
| sealed-set contamination | sealed examples seen in training/dev | group + normalized screens at prepare | generator regression | re-run screens | regenerate sealed; record invalidation |
| EOS contract failure | generations ending MAX_TOKENS; eos_correct < 1 | stop-reason reporting | objective mis-wiring | stop-reason histogram | objective is LOCKED; violation = caller bug |
| EXPERIMENT_ONLY leak | masked/offset logits in canonical receipts | contract gate (`allow_experimental`) + identity hash | config drift | contract status check | canonical path is the only default; fix config |
| formation failure | all families ≤ baselines after training | dev evaluation vs shortcut table | training mechanics bug (this is a Task-3 FAIL per §31) | loss curve + per-family train exact | diagnose before scale; do not proceed |
| operator/hardware | OOM; CUDA errors | preflight calibration | batch too large; driver | memory model in GPU_FIT_PLAN | reduce batch; resume from checkpoint |
