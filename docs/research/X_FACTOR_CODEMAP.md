# X-factor evidence map

Read from `core-vnext` source and `git show origin/iterate500:<path>`. Leftover `cognition/`, `agents/`, `state/` pycache on disk is **not** this branch.

Status: **D** demonstrated · **P** plausible · **S** speculative · **A** absent · **Stub** types exist, no closed loop

---

## Core (`core-vnext`) — D unless noted

| Item | Evidence |
|---|---|
| 180,093,312 dense params, 18L × 896, vocab 32768, ctx 2048 | `anra_core/config.py` `CANONICAL_CONFIG`; `tests/test_core.py` |
| GQA 14/2, SwiGLU, RoPE, QK-norm, sliding 1024, full attn every 4 | `anra_core/model.py` `GroupedQueryAttention`, `DenseBlock` |
| MoE/MTP forbidden on this Core | `CoreConfig.__post_init__` raises if `use_moe`/`use_mtp` |
| Historical extra ~1.04M dormant tensors not executed | README + architecture spec |
| IDs → logits only | `AnRaCore.forward` / `forward_incremental` |
| Executor owns device/dtype/state | `anra_core/executor.py` |
| Fork / rollback / reset / release | `fork_state`, `rollback_state`; `tests/test_state_isolation.py` |
| Typed failures | `anra_core/errors.py` (`ERR_REPRESENTATION_INCOMPATIBLE`, overflow, …) |
| No state serialization | `serialize_state` → `UnsupportedCapabilityError` |
| Tokenizer is the representation | `anra_core/tokenizer.py`; golden vectors in `test_core.py` |
| Training = next-token windows | `training/train_xla.py` `TokenBlockDataset` |
| Capability floor | `docs/engineering/V4_BEHAVIOR_BASELINE.md` — continuation, not IF/arithmetic |

## Reference Connector on this branch

| Item | Status | Evidence |
|---|---|---|
| Sample + score | **D** | `anra_core/brain.py` `Brain.think` |
| “Self-likelihood” = exp(mean logprob) clipped | **D** (not P(success)) | `Brain.think` |
| Deliberate = ≤4 candidates | **D** | `ThoughtPolicy` |
| Failure classer | **A** | no such code |
| Ablation battery | **A** | fork is tested, never used for K vs π |

## iterate500 overlay (not in this branch)

| Capability asked | Status | What the code actually does |
|---|---|---|
| 1. Failure understanding | **Stub** | `engine/agent_loop.py`: on fail, `"reasoning"` if `causal_type != "unknown"` else `"execution"`. Epistemic tracker is a provenance-weighted confidence formula, not a cause. |
| 2. Failure testing (hold π, change K) | **A** | No such experiment. `innovation/hypothesis.py` maps TODO/stub strings to pytest recipes. |
| 3. Task structure | **Stub** | `intelligence/hgp.py` validates trees; decomposer is injected. `memory/experimental_proof_graph.py` is a JSON graph API. `cognition/cdse.py` is hardcoded seed signatures. phase2 planner is regex/heuristic steps. |
| 4. Self-model P(success \| type, ctx, strategy, tools) | **Stub** | `intelligence/competence.py`: running mean accuracy/calibration/coverage **by domain string**. Policy thresholds → `direct/verify/retrieve_and_decompose/research_or_clarify`. Not a function of strategy/tools. Not called from Core. |
| 5. Internal simulation | **Stub** | `robotics/world_model.py`: GRU on blake2-hashed JSON; gated at 1e5 transitions. **Real** simulation primitive is Core fork (**D**, unused). |
| 6. Learning from experience | **Stub** | Fail → `TrajectoryStore.append` + `CorrectedFailureCurriculum.capture_task_result`. Weights: OGRS `weight_updates_allowed=False`. Type A adds Python tools. Type B mutates files; default benchmark is `SystemExit(0)`. No router for memory vs plan vs weights vs nothing. |

### Named files that are not the Connector

| File | Actual job |
|---|---|
| `inference/full_system_connector.py` | Walk repo, AST class/func names, `importlib` health probes |
| `evaluation/ibs.py` | 50 templated prompts (`"Solve and verify reasoning problem {n}."`), `failure_class=""` |
| `intelligence/ogrs.py` | 20-line drift → retrieval/candidate counts |
| `intelligence/verifier_search.py` | Empty registry unless caller `.register`s |
| `cognition/self_debate.py` | Keyword risk classifier + injected generators |
| `intelligence/curiosity.py` | Rank `(Δloss × novelty × verifiability)` |

## Six questions, short

| # | Answer |
|---|---|
| When it fails, can it determine why? | **No (D).** Two-class heuristic or a string field. |
| Can it test the diagnosis? | **No (D).** |
| Structure formation vs retrieval? | **Types only (Stub).** Trees/graphs are containers. |
| Self-model as P(success\|…)? | **No (D).** Logprob and domain averages. |
| Predict action outcomes before execution? | **Not for language (D).** Fork could; GRU-hash should not. |
| After success/failure, what changes, and who decides? | **Trajectory JSON + optional CDR row (D).** Decision of *what* to change: **A**. |

## Framing

Core → Connector → Outer is **D** as a *boundary* on `core-vnext` (architecture spec table). It is **not** implemented as a working cognition stack. Keep the boundary. Build the experimenter in Connector. Do not grow the overlay.
