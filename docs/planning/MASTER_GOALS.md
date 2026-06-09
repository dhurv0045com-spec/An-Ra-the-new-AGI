# Master Goals — An-Ra

> **Purpose:** Single backlog for everything the project must achieve — research, testing, training, operator/Jarvis features, robotics, governance, and docs.  
> **Status keys:** `DONE` · `ACTIVE` · `NEXT` · `BLOCKED` · `IDEA`  
> **Update this file** when work completes. Log shipped work in [`../engineering/ENGINEERING_LOG.md`](../engineering/ENGINEERING_LOG.md).

Last reviewed: **2026-06-08**

---

## How to use this file

| Column | Meaning |
|--------|---------|
| **ID** | Stable reference (cite in commits, log, issues) |
| **Status** | Current state |
| **Component** | Registry id from `runtime/system_registry.py` |
| **Verifier** | How we know it is done |
| **Means** | Research / build / test / train / operate |

---

## P0 — Critical path (ship real capability)

| ID | Status | Goal | Component | Verifier | Means |
|----|--------|------|-----------|----------|-------|
| P0-01 | NEXT | GPU milestone train + promoted `anra_v2_brain.pt` | `training_loop` | Sovereignty pass + eval delta in `output/v2/eval/` | train |
| P0-02 | NEXT | Golden eval baseline JSON committed or on Drive | `evaluation` | `train_unified --mode eval` artifact | test |
| P0-03 | ACTIVE | Daily `session` loop runs without hidden checkpoint deps | `training_loop` | 7 consecutive session reports | operate |
| P0-04 | DONE | Operator pack: files, open, CAD stub, slash commands | `operator` | `pytest tests/test_operator_tools.py` | build |
| P0-05 | NEXT | Symbolic bridge in default generation path | `symbolic_bridge`, `runtime` | Wrong math blocked before reply; telemetry | build + test |
| P0-06 | NEXT | Telemetry non-empty after smoke session | `engine` | `anra.py --report` latency > 0 | operate |

---

## Research

| ID | Status | Goal | Component | Verifier | Means |
|----|--------|------|-----------|----------|-------|
| R-01 | ACTIVE | DFC / FCC corpus quality audit | `data_mix` | `frontier_dfc.jsonl` template balance report | research |
| R-02 | IDEA | Constraint isomorphism benchmarks across domains | `identity` | Paper-style eval set + scores | research |
| R-03 | IDEA | RLVR scaling study (G, KL, verifier types) | `training_loop` | Ablation in `output/v2/eval/` | research |
| R-04 | IDEA | HAL hormone → behavior correlation study | `identity` | Logged sessions + measured outcomes | research |
| R-05 | NEXT | Engineering CAD: real turbofan dims from public sources | `operator` | REPORT.md cites sources; falsifiers listed | research |
| R-06 | IDEA | Remote node threat model + pairing protocol | `docs` | Design doc + security checklist | research |
| R-07 | DONE | Best-of intelligence and efficiency research pack | `docs`, `training_loop`, `runtime`, `memory` | `docs/research/ANRA_BEST_RESEARCH_FOR_INTELLIGENCE_AND_EFFICIENCY.md` + roadmap | research |

---

## Testing & quality

| ID | Status | Goal | Component | Verifier | Means |
|----|--------|------|-----------|----------|-------|
| T-01 | DONE | Full pytest green on Windows | `tests` | `pytest tests/ -q` - 259 passed, 1 warning on 2026-06-08 | test |
| T-02 | NEXT | CI workflow: report + pytest on push | `engine` | GitHub Actions green | build |
| T-03 | NEXT | Eval harness golden file per component | `evaluation` | `EvalHarness.compare` in CI | test |
| T-04 | NEXT | Phase 3 integration smoke in CI | `symbolic_bridge`, `ghost_memory` | `test_phase3_integration.py` subset | test |
| T-05 | IDEA | Red-team safety suite for agent tools | `agent_loop` | `safety/` automated cases | test |
| T-06 | NEXT | Load test: 1000 telemetry records / report | `engine` | Report completes < 5s | test |

---

## Training & model

| ID | Status | Goal | Component | Verifier | Means |
|----|--------|------|-----------|----------|-------|
| TR-01 | NEXT | Owner data ingest pipeline documented | `data_mix` | `anra_training.txt` growth SOP | operate |
| TR-02 | NEXT | Replay pipeline feeds failures into mix | `training_loop` | Hard examples in next session mix | train |
| TR-03 | NEXT | Identity checkpoint + CIVGuard gate | `identity` | Milestone promote only if drift OK | train |
| TR-04 | NEXT | Ouroboros checkpoint milestone path | `ouroboros` | `anra_v2_ouroboros.pt` + eval | train |
| TR-05 | IDEA | STaR + RLVR combined session mode | `training_loop` | A/B eval vs baseline | research + train |
| TR-06 | IDEA | Distilled 25m smoke model for CPU demos | `brain` | Generation quality rubric | train |
| TR-07 | DONE | SparseLoRA logging-only efficiency estimate | `training_loop` | `v2_sparse_lora_report.json` + focused tests | research + test |
| TR-08 | DONE | Optimizer bake-off scaffold | `training_loop` | `v2_optimizer_bakeoff_report.json` + focused tests | research + test |
| TR-09 | DONE | RLVR DAPO-style telemetry scaffold | `training_loop` | `v2_rlvr_report.json` + focused tests | research + test |
| TR-10 | DONE | GEPA-style self-improvement proposals | `self_improvement` | `v2_gepa_report.json` + focused tests | research + test |
| TR-11 | NEXT | TurboQuant and KVarN audit | `runtime` | `v2_turboquant_audit_report.json` + compression/error tests | research + test |

---

## Operator / Jarvis (desktop work)

| ID | Status | Goal | Component | Verifier | Means |
|----|--------|------|-----------|----------|-------|
| O-01 | DONE | Workspace sandbox + `get_agent_workspace()` | `operator` | Path tests | build |
| O-02 | DONE | `os_action` open/reveal/URL | `agent_loop` | Manual + unit tests | build |
| O-03 | DONE | `cad_generate` raptor_engine template | `operator` | `.scad` + REPORT in workspace | build |
| O-04 | DONE | Chat auto-routes “create file” to tools without `/goal` | `master_system` | `pytest tests/test_operator_tools.py -q` | build |
| O-05 | NEXT | CadQuery adapter (optional dep) | `operator` | STL export without OpenSCAD | build |
| O-06 | NEXT | Web UI: operator panel (write, goal, workspace) | `api_web` | phase4 panel calls API | build |
| O-07 | IDEA | Voice input → goal queue | `master_system` | End-to-end demo | build |

---

## Engineering spine & governance

| ID | Status | Goal | Component | Verifier | Means |
|----|--------|------|-----------|----------|-------|
| E-01 | DONE | ENGINEERING_LOG + LOG_STANDARD + log script | `engine` | `log_engineering_change.py` | build |
| E-02 | DONE | MASTER_GOALS backlog (this file) | `docs` | Owner review | docs |
| E-03 | NEXT | Pre-commit hook suggests log entry if `engine/` touched | `engine` | Hook message | build |
| E-04 | NEXT | Report exports JSON for dashboards | `engine` | `output/v2/reports/*.json` schema | build |
| E-05 | NEXT | Innovation cycle wired to MASTER_GOALS IDs | `self_improvement` | `run_innovation_cycle.py` updates status | operate |
| E-06 | ACTIVE | Every AI PR includes log entry | `docs` | DEVELOPER checklist | operate |

---

## Memory, agency, cognition

| ID | Status | Goal | Component | Verifier | Means |
|----|--------|------|-----------|----------|-------|
| C-01 | NEXT | Goal queue → orchestrator daily autopilot | `goals` | `run_session` summary JSON | operate |
| C-02 | NEXT | Ghost memory injected in `generate.py` by default | `ghost_memory` | A/B context recall test | build |
| C-03 | IDEA | Falsification ledger auto-append on symbolic verify | `identity` | Ledger entries from 45Q | build |
| C-04 | NEXT | Agent goals use sovereignty read before destructive ops | `sovereignty` | Tier-3 block demo | test |

---

## Remote systems & robotics (future)

| ID | Status | Goal | Component | Verifier | Means |
|----|--------|------|-----------|----------|-------|
| X-01 | IDEA | `anra_node` paired agent (health, files, tier 3) | `master_system` | Two-machine demo with consent | research + build |
| X-02 | IDEA | ROS2 bridge: An-Ra goal → action server | `agent_loop` | Gazebo sim task | research + build |
| X-03 | IDEA | Sim benchmark: pick-place success rate | `evaluation` | Scorecard metric | test |
| X-04 | IDEA | Hardware estop + tier 4 gate | `master_system` | Safety doc + test | build |

---

## Documentation

| ID | Status | Goal | Component | Verifier | Means |
|----|--------|------|-----------|----------|-------|
| D-01 | DONE | Documentation map + engineering log + master goals | `docs` | `README.md` | docs |
| D-02 | ACTIVE | Keep ENGINEERING_LOG current | `docs` | No untracked spine merges | operate |
| D-03 | NEXT | Colab notebook section for operator commands | `api_web` | Notebook runs `/cad` equivalent | docs |
| D-04 | IDEA | Video walkthrough script from WALKTHROUGH | `docs` | Published outline | docs |

---

## Milestone definitions (when to call a phase “done”)

### Milestone A — **Operator-ready** (target: near term)

- [x] P0-04 Operator pack  
- [ ] P0-05 Symbolic in loop  
- [ ] P0-06 Telemetry filled  
- [x] O-04 Natural language → tools  

### Milestone B — **Train-ready**

- [ ] P0-01 Promoted brain checkpoint  
- [ ] P0-02 Eval baseline  
- [ ] TR-02 Replay into mix  
- [x] TR-07 SparseLoRA estimate  
- [x] TR-08 Optimizer bake-off scaffold  
- [x] TR-09 RLVR telemetry scaffold  
- [x] TR-10 GEPA proposal scaffold  

### Milestone C — **Field-ready** (long arc)

- [ ] X-01 Paired node  
- [ ] X-02 ROS2 sim  
- [ ] C-01 Autopilot goals  

---

## Adding a new goal

1. Pick next ID prefix (`R-`, `T-`, …) or extend table.  
2. Set `Status` = `IDEA` or `NEXT`.  
3. Name **Verifier** before starting work.  
4. On completion: `DONE` + entry in `ENGINEERING_LOG.md`.

```bash
python scripts/log_engineering_change.py \
  --component evaluation \
  --type ADD \
  --summary "Golden eval baseline on Drive" \
  --verify "train_unified --mode eval"
```

---

*This file is the “by all means” list — research, testing, improving, operating. The engineering log is the “what we actually did.”*
