"""Evidence-backed build verification (F21, FINAL-K8 section 22).

`run_verify_build` executes the registered local contract checks as bounded
child processes, exercises real production interfaces WITHOUT optimizer
commits, validates the data/config/schema closure, and writes a
machine-readable build report. Readiness is DERIVED from that report's
complete passing evidence and matching source identity — never hardcoded,
never inherited from a report file's mere presence.

Fail-closed rules (section 22):
- unrun, failed, timed-out or empty local checks cannot be PASS;
- a requirement with any missing or failed evidence is not PASS;
- changed relevant source/data invalidates a cached report;
- production allocation authority is never created here;
- local optimizer commits remain zero (gradients are discarded).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable

BUILD_VERIFICATION_SCHEMA = "bramastra-k8-build-verification/v1"

REPORT_FILENAME = "build_verification.json"
DEFAULT_REPORT_SEARCH = os.path.join(
    "engineering", "reports", "FINAL_K8", REPORT_FILENAME)

# Registered local contract/integration check groups. Every group is a
# bounded pytest child process; exit code, duration and output tail are
# recorded as receipts. A group that fails, times out or produces no
# collected tests cannot support a PASS.
CHECK_GROUPS: dict[str, dict[str, Any]] = {
    "foundation": {
        "targets": ["tests/test_research_k8_foundation.py",
                    "tests/test_research_k8.py",
                    "tests/test_research_cognition.py",
                    "tests/test_research_cognition_runtime.py",
                    "tests/test_research_scoring_device_contract.py"],
        "timeout_seconds": 900,
    },
    "phase_contracts": {
        "targets": ["tests/test_e0_contracts.py", "tests/test_e0_results.py",
                    "tests/test_e0_scoring.py",
                    "tests/test_research_k8_real.py",
                    "tests/test_research_k8_operational.py",
                    "tests/test_research_k8_executors.py"],
        "timeout_seconds": 900,
    },
    "data_and_splits": {
        "targets": ["tests/test_e3_data_objective.py",
                    "tests/test_research_data.py",
                    "tests/test_research_environments.py"],
        "timeout_seconds": 900,
    },
    "gate_and_readiness": {
        "targets": ["tests/test_research_k8_launch_gate.py",
                    "tests/test_research_k8_readiness.py"],
        "timeout_seconds": 600,
    },
    "checkpoint_and_ledger": {
        "targets": ["tests/test_research_checkpoint.py",
                    "tests/test_research_ledger_replay.py",
                    "tests/test_research_accounting.py"],
        "timeout_seconds": 900,
    },
    "statistics_and_evaluation": {
        "targets": ["tests/test_research_evaluation.py",
                    "tests/test_research_meta_rsi.py"],
        "timeout_seconds": 900,
    },
    "experience_codec": {
        "targets": ["tests/test_research_experience.py",
                    "tests/test_research_contracts.py",
                    "tests/test_research_config.py"],
        "timeout_seconds": 900,
    },
}

# Real-interface exercises. Each runs production code paths in this process
# with gradients discarded and zero optimizer commits. Doubles stay labeled;
# these exercises must use real models/environments on representative paths.
EXERCISES: dict[str, dict[str, Any]] = {
    "campaign_architecture": {
        "description": "instantiate the frozen k8-campaign model and verify "
                       "the registered base parameter count",
    },
    "bundle_contracts": {
        "description": "validate the passed bundle (hashes, splits, "
                       "information-sufficiency witnesses) and compile "
                       "production channels from real rows",
    },
    "no_update_boundary": {
        "description": "real accumulate/no-op finalize on a bundle row with "
                       "finite gradients and zero committed updates",
    },
    "live_episode_no_update": {
        "description": "one real cognition episode with randomly initialized "
                       "model calls and no optimizer involvement",
    },
    "checkpoint_roundtrip": {
        "description": "publish and reload a checkpoint payload with hash "
                       "verification and parent binding",
    },
    "notebook_inputs": {
        "description": "owner notebook parses, carries no hardcoded source "
                       "paths, and drives the registered CLI commands",
    },
    "integrated_rehearsal": {
        "description": "bounded no-update E0-E6 control rehearsal through "
                       "production interfaces",
    },
}

# Requirement registry (F01-F24): the exact evidence each requirement needs.
# Statuses are DERIVED: pass requires every registered check group and
# exercise to have passed; anything missing/failed is not a pass. CUDA-only
# runtime items are listed in hardware_pending and never satisfied locally.
REQUIREMENT_REGISTRY: dict[str, dict[str, Any]] = {
    "F01": {"checks": ["foundation"], "exercises": ["campaign_architecture"],
            "hardware_pending": [], "implemented_symbols": [
                "bramastra_lab.research.models:IntegratedModel",
                "bramastra_lab.research.campaigns.phases.ops:k8_campaign_config"]},
    "F02": {"checks": ["foundation", "data_and_splits"],
            "exercises": ["bundle_contracts"], "hardware_pending": [],
            "implemented_symbols": [
                "bramastra_lab.research.data.k8_bundle:rule_information_witness",
                "bramastra_lab.research.data.k8_bundle:information_sufficiency_witness"]},
    "F03": {"checks": ["data_and_splits"], "exercises": ["bundle_contracts"],
            "hardware_pending": [], "implemented_symbols": [
                "bramastra_lab.research.data.k8_bundle:build_k8_bundle",
                "bramastra_lab.research.data.k8_bundle:validate_bundle"]},
    "F04": {"checks": ["foundation", "experience_codec"],
            "exercises": ["bundle_contracts"], "hardware_pending": [],
            "implemented_symbols": [
                "bramastra_lab.research.campaigns.phases.compiler:"
                "goal_prefix_tokens",
                "bramastra_lab.research.experience.codec:encode_event"]},
    "F05": {"checks": ["phase_contracts"], "exercises": ["no_update_boundary"],
            "hardware_pending": [], "implemented_symbols": [
                "bramastra_lab.research.learning.trainer:Trainer",
                "bramastra_lab.research.experience.supervision:"
                "SupervisionWindow"]},
    "F06": {"checks": ["foundation", "phase_contracts"],
            "exercises": ["no_update_boundary", "live_episode_no_update"],
            "hardware_pending": [], "implemented_symbols": [
                "bramastra_lab.research.learning.k8_scoring:"
                "score_candidates_trainable",
                "bramastra_lab.research.learning.k8_scoring:"
                "value_estimate_trainable",
                "bramastra_lab.research.learning.k8_scoring:"
                "world_transition_token_loss"]},
    "F07": {"checks": ["foundation"], "exercises": ["live_episode_no_update"],
            "hardware_pending": [], "implemented_symbols": [
                "bramastra_lab.research.cognition.episode:mark_conflicts",
                "bramastra_lab.research.cognition.episode:admit_observation_evidence",
                "bramastra_lab.research.cognition.episode:admit_typed_observation",
                "bramastra_lab.research.cognition.episode:sync_typed_conflict_states",
                "bramastra_lab.research.cognition.workspace:CognitiveWorkspace"]},
    "F08": {"checks": ["foundation"], "exercises": ["live_episode_no_update"],
            "hardware_pending": [], "implemented_symbols": [
                "bramastra_lab.research.cognition.episode:BoundedPlannerAdapter",
                "bramastra_lab.research.cognition.episode:run_episode"]},
    "F09": {"checks": ["phase_contracts"], "exercises": [],
            "hardware_pending": ["owner E2 loads the trained E1 A/B "
                                 "checkpoints on T4"],
            "implemented_symbols": [
                "bramastra_lab.research.campaigns.phases.e2:execute"]},
    "F10": {"checks": ["phase_contracts", "checkpoint_and_ledger"],
            "exercises": [], "hardware_pending": ["owner E1 training on T4"],
            "implemented_symbols": [
                "bramastra_lab.research.campaigns.phases.e1:execute"]},
    "F11": {"checks": ["data_and_splits", "phase_contracts"],
            "exercises": [], "hardware_pending": [],
            "implemented_symbols": [
                "bramastra_lab.research.environments.k8_live:ToolEnv",
                "bramastra_lab.research.campaigns.phases.e3:execute"]},
    "F12": {"checks": ["phase_contracts"], "exercises": [],
            "hardware_pending": ["owner E4 gated training on T4"],
            "implemented_symbols": [
                "bramastra_lab.research.models.gated:GatedReuseModel",
                "bramastra_lab.research.campaigns.phases.e4:execute"]},
    "F13": {"checks": ["phase_contracts"], "exercises": [],
            "hardware_pending": [], "implemented_symbols": [
                "bramastra_lab.research.campaigns.trial_service",
                "bramastra_lab.research.campaigns.phases.session:"
                "bind_job_reservation"]},
    "F14": {"checks": ["phase_contracts", "statistics_and_evaluation"],
            "exercises": [], "hardware_pending": ["owner E5 proposer "
                                                  "training on T4"],
            "implemented_symbols": [
                "bramastra_lab.research.campaigns.phases.e5:execute",
                "bramastra_lab.research.metalearning.dispatch:MethodArchive"]},
    "F15": {"checks": ["checkpoint_and_ledger", "phase_contracts"],
            "exercises": [], "hardware_pending": [],
            "implemented_symbols": [
                "bramastra_lab.research.campaigns.supervisor:CampaignLedger",
                "bramastra_lab.research.campaigns.runner:Runner"]},
    "F16": {"checks": ["phase_contracts"], "exercises": ["no_update_boundary"],
            "hardware_pending": ["owner E0: full-profile CUDA qualification, "
                                 "fresh-process resume, complete timing units"],
            "implemented_symbols": [
                "bramastra_lab.research.campaigns.worker:_run_e0"]},
    "F17": {"checks": ["checkpoint_and_ledger"],
            "exercises": ["checkpoint_roundtrip"],
            "hardware_pending": ["owner E0 CUDA/AMP checkpoint resume"],
            "implemented_symbols": [
                "bramastra_lab.research.runtime.checkpoint:publish_checkpoint"]},
    "F18": {"checks": ["statistics_and_evaluation"], "exercises": [],
            "hardware_pending": [], "implemented_symbols": [
                "bramastra_lab.research.evaluation.scoring:decide_promotion",
                "bramastra_lab.research.evaluation.scoring:"
                "clustered_bootstrap_delta"]},
    "F19": {"checks": ["gate_and_readiness"], "exercises": ["notebook_inputs"],
            "hardware_pending": [], "implemented_symbols": [
                "bramastra_lab.research.campaigns.k8:build_parser"]},
    "F20": {"checks": ["phase_contracts", "checkpoint_and_ledger"],
            "exercises": ["integrated_rehearsal"], "hardware_pending": [],
            "implemented_symbols": [
                "bramastra_lab.research.campaigns.export:build_export"]},
    "F21": {"checks": ["gate_and_readiness"], "exercises": [],
            "hardware_pending": [], "implemented_symbols": [
                "bramastra_lab.research.campaigns.verify_build:run_verify_build",
                "bramastra_lab.research.campaigns.readiness:"
                "implementation_readiness"]},
    "F22": {"checks": ["phase_contracts", "gate_and_readiness"],
            "exercises": ["integrated_rehearsal"], "hardware_pending": [],
            "implemented_symbols": [
                "bramastra_lab.research.campaigns.rehearsal:run_rehearsal"]},
    "F23": {"checks": ["gate_and_readiness"], "exercises": ["notebook_inputs"],
            "hardware_pending": [], "implemented_symbols": [
                "bramastra_lab.research.campaigns.verify_build:run_verify_build"]},
    "F24": {"checks": ["gate_and_readiness"],
            "exercises": ["integrated_rehearsal"], "hardware_pending": [],
            "implemented_symbols": [
                "bramastra_lab.research.campaigns.readiness:"
                "require_implementation_ready"]},
}

RUNTIME_GATES = [
    {"id": "G01", "required_before": "E1",
     "criterion": "two distinct actual T4 devices, worker-local mapping, "
                  "compatible precision and live allowance"},
    {"id": "G02", "required_before": "E1",
     "criterion": "full-profile real gradients/updates and fresh-process "
                  "exact continuation within E0 counters/deadline"},
    {"id": "G03", "required_before": "E1",
     "criterion": "complete measured update/evaluation costs meet registered "
                  "minima and fit frozen cases/settings"},
    {"id": "G04", "required_before": "E1",
     "criterion": "current allocation, source/data/schema identities and "
                  "frozen protocol verified by the runner"},
]


# --------------------------------------------------------------------------
# Real-interface exercises (no optimizer commits, gradients discarded)
# --------------------------------------------------------------------------

def exercise_campaign_architecture() -> dict[str, Any]:
    from bramastra_lab.research.campaigns.phases.ops import k8_campaign_config
    from bramastra_lab.research.models import IntegratedModel

    config = k8_campaign_config()
    model = IntegratedModel(config)
    total = sum(p.numel() for p in model.parameters())
    if total != 6_493_952:
        raise ValueError(
            f"campaign model parameter count {total} != 6,493,952")
    layers = int(config.model.layers)
    width = int(config.model.width)
    if (layers, width) != (8, 256):
        raise ValueError(
            f"campaign geometry {layers}L/{width}w != registered 8L/256w")
    return {"base_parameters": total, "layers": layers, "width": width,
            "max_seq": int(config.model.max_seq),
            "config_identity": config.identity()}


def exercise_bundle_contracts(data_dir: str, *, sample_limit: int = 12) -> dict[str, Any]:
    from bramastra_lab.research.campaigns.phases.compiler import (
        build_batch_for_trajectory, compile_channels_for_row)
    from bramastra_lab.research.data.k8_bundle import (
        build_k8_bundle, information_sufficiency_witness, validate_bundle)
    from bramastra_lab.research.experience.sequences import SequenceRow

    result = validate_bundle(data_dir)
    if not result.get("valid"):
        raise ValueError(
            f"bundle validation failed: {result.get('issues', [])[:4]}")
    episodes_dir = os.path.join(data_dir, "episodes")
    if not os.path.isdir(episodes_dir):
        raise ValueError("bundle has no episodes directory")
    witnessed = 0
    compiled = 0
    weights = {"token": 1.0, "world": 0.5, "action": 0.5, "value": 0.1,
               "pair": 0.0, "pg": 0.0}
    enabled = frozenset({"token", "world", "action", "value"})
    for name in sorted(os.listdir(episodes_dir)):
        if not name.endswith(".jsonl"):
            continue
        with open(os.path.join(episodes_dir, name), encoding="utf-8") as fh:
            for index, line in enumerate(fh):
                if not line.strip():
                    continue
                row = json.loads(line)
                information_sufficiency_witness(
                    str(row.get("family")), dict(row.get("public") or {}))
                witnessed += 1
                if compiled < sample_limit and index % 7 == 0:
                    batch = build_batch_for_trajectory(row)
                    compile_channels_for_row(
                        row, batch, arm_weights=dict(weights),
                        arm_enabled=enabled)
                    compiled += 1
    if witnessed == 0:
        raise ValueError("bundle carries no episode rows to witness")
    return {"bundle_identity": result.get("identity"),
            "witnessed_rows": witnessed, "compiled_rows": compiled}


def exercise_no_update_boundary(data_dir: str) -> dict[str, Any]:
    import torch

    from bramastra_lab.research.campaigns.phases.compiler import (
        build_batch_for_trajectory, compile_channels_for_row,
        load_training_trajectories)
    from bramastra_lab.research.campaigns.phases.ops import k8_campaign_config
    from bramastra_lab.research.campaigns.phases.session import (
        drive_noop_boundary)
    from bramastra_lab.research.experience.supervision import SupervisionWindow
    from bramastra_lab.research.learning.k8_scoring import (
        score_candidates_trainable, value_estimate_trainable,
        world_transition_token_loss)
    from bramastra_lab.research.learning.k8_trainer import K8Trainer
    from bramastra_lab.research.models import IntegratedModel

    config = k8_campaign_config()
    model = IntegratedModel(config)
    trainer = K8Trainer(config, model, device="cpu", precision="fp32",
                        require_allocation=False)
    # Local diagnostic admission context (the established E0 no-step
    # pattern): admission is proved at the real boundary, then gradients
    # are discarded. This is not production allocation authority and
    # commits nothing.
    from bramastra_lab.research.learning.k8_trainer import AllocationContext

    trainer.begin_campaign(AllocationContext(
        allocation_id="verify-build-local-diagnostic", device="cpu",
        deadline_unix=time.time() + 600.0, remaining_updates=1,
        job_id="verify-build-noop", phase="VERIFY-BUILD"))
    rows = load_training_trajectories(data_dir, seed=7)
    batch = build_batch_for_trajectory(rows[0])
    compiled = compile_channels_for_row(
        rows[0], batch, arm_weights={"token": 1.0, "world": 0.5,
                                     "action": 0.5, "value": 0.1,
                                     "pair": 0.0, "pg": 0.0},
        arm_enabled=frozenset({"token", "world", "action", "value"}))

    def extra_terms():
        prefix = compiled["world"]["prefix_tokens"]
        scored = score_candidates_trainable(
            model, config, prefix, compiled["action"]["candidates"])
        value = value_estimate_trainable(model, config, prefix)
        world = world_transition_token_loss(
            model, config, prefix, action=compiled["world"]["action"],
            target_feedback=compiled["world"]["target_feedback"])
        return {"action": -scored["log_probs"].sum(),
                "value": value.square(), "world": world}

    def window_builder(target_count: int) -> SupervisionWindow:
        window = SupervisionWindow(
            weights={"token": 1.0, "world": 0.5, "action": 0.5, "value": 0.1,
                     "pair": 0.0, "pg": 0.0},
            enabled_terms=frozenset({"token", "world", "action", "value"}))
        window.add("token", target_count)
        window.add("world", 1)
        window.add("action", 1)
        window.add("value", 1)
        return window

    window = window_builder(batch.target_count)
    extra = extra_terms()
    handle = {"trainer": trainer, "model": model, "config": config,
              "seed": 0, "profile": "k8-campaign"}
    proof = drive_noop_boundary(handle, batch=batch, window=window,
                                extra=extra, pair_rows=None)
    grads = proof.get("named_grads_finite", {})
    if not grads:
        raise ValueError("no-op boundary produced no finite-gradient proof")
    updates = int(trainer.counters.optimizer_updates)
    if updates != 0:
        raise ValueError(
            f"no-update boundary committed {updates} optimizer updates")
    return {"grad_finite_params": len(grads),
            "optimizer_updates": updates,
            "attempted_updates": int(trainer.attempted_updates),
            "boundary": "backward-only-gradients-discarded"}


def exercise_live_episode_no_update() -> dict[str, Any]:
    from bramastra_lab.research.cognition import episode as kernel
    from bramastra_lab.research.campaigns.phases.e2 import EVAL_FAMILIES
    from bramastra_lab.research.campaigns.phases.ops import k8_campaign_config
    from bramastra_lab.research.environments.k8_live import (
        build_live_env, generate_live_mechanism)
    from bramastra_lab.research.models import IntegratedModel

    config = k8_campaign_config()
    model_core = IntegratedModel(config)
    model_core.eval()
    model = kernel.FreeGenerationModel(model_core, config)
    family_traces = []
    for index, family in enumerate(EVAL_FAMILIES, start=3):
        episode_seed = 8609 + index
        mechanism = generate_live_mechanism(family, index, seed=8609)
        env = build_live_env(mechanism, budget=6, seed=episode_seed)
        trace = kernel.run_episode(
            env, kernel.LearnedPolicyAdapter(), model=model, seed=episode_seed,
            mechanism=mechanism, session_job_id="verify-build",
            checkpoint_id=None, action_budget=1, call_budget=4,
            node_budget=4)
        model_calls = int(trace["summary"].get("model_calls", 0))
        if not trace.get("events") or model_calls <= 0:
            raise ValueError(
                f"{family} live episode produced no model-origin call trace")
        if model_calls > 4:
            raise ValueError(
                f"{family} live episode exceeded its four-call diagnostic cap")
        family_traces.append({
            "family": family, "seed": episode_seed,
            "events": len(trace["events"]),
            "model_calls": model_calls,
            "terminated": trace["summary"].get("terminated"),
            "truncated": trace["summary"].get("truncated"),
            "success": trace["summary"].get("success"),
            "model_origins": sorted({
                str(event.get("model_origin"))
                for event in trace["events"]
                if event.get("model_origin") is not None}),
        })
    non_null_gradients = sum(
        parameter.grad is not None for parameter in model_core.parameters())
    if non_null_gradients:
        raise ValueError(
            f"inference diagnostic retained gradients on "
            f"{non_null_gradients} parameters")
    return {"families": list(EVAL_FAMILIES),
            "model_origin": "random-init-real-calls",
            "profile": "k8-campaign",
            "family_traces": family_traces,
            "call_cap_per_family": 4,
            "non_null_gradients": non_null_gradients,
            "model_training_mode": bool(model_core.training),
            "optimizer_updates": 0}


def exercise_checkpoint_roundtrip() -> dict[str, Any]:
    import torch

    from bramastra_lab.research.runtime import checkpoint as ckpt

    with tempfile.TemporaryDirectory(prefix="verify-build-ckpt-") as tmp:
        payload = {"model": {"w": torch.arange(8, dtype=torch.float32)},
                   "counters": {"optimizer_updates": 0}}
        manifest = ckpt.publish_checkpoint(
            run_dir=tmp, run_id="verify-build-roundtrip", update_index=0,
            payload=payload, config_identity="cfg-verify",
            tokenizer_identity="tok-verify", data_identity="data-verify",
            parent_checkpoint_id=None, code_identity="code-verify")
        restored, _loaded_manifest = ckpt.load_checkpoint(
            tmp, checkpoint_id=manifest.checkpoint_id)
        same = bool(torch.equal(restored["model"]["w"], payload["model"]["w"]))
        if not same:
            raise ValueError("checkpoint roundtrip changed the payload")
        return {"checkpoint_id": manifest.checkpoint_id,
                "payload_sha256": manifest.payload_sha256,
                "restored": True, "optimizer_updates": 0}


def exercise_notebook_inputs(notebook_path: str | None) -> dict[str, Any]:
    if not notebook_path or not os.path.exists(notebook_path):
        raise ValueError(
            "owner notebook missing: pass --notebook <path> (the verifier "
            "must validate the actual operator input, not assume it)")
    with open(notebook_path, encoding="utf-8") as handle:
        notebook = json.load(handle)
    cells = notebook.get("cells") or []
    if not cells:
        raise ValueError("owner notebook has no cells")
    sources = []
    for cell in cells:
        source = cell.get("source")
        if isinstance(source, list):
            source = "".join(source)
        sources.append(str(source))
        if cell.get("cell_type") == "code":
            stripped = str(source).lstrip()
            if stripped.startswith("!") or stripped.startswith("%"):
                raise ValueError(
                    "owner notebook uses shell/line magics; registered CLI "
                    "subprocess commands are required")
    joined = "\n".join(sources)
    lowered = joined.lower()
    # Host SOURCE/repository paths are forbidden; operator-supplied output
    # roots (e.g. a Kaggle working directory) are legitimate inputs.
    for forbidden in ("/users/", "c:\\users", "c:/users"):
        if forbidden in lowered:
            raise ValueError(
                f"owner notebook hardcodes a host source path ({forbidden!r})")
    # The notebook drives the registered CLI through checked subprocess
    # argument lists; detect the commands by their argument tokens.
    tokens = lowered.replace("'", " ").replace('"', " ").replace(",", " ")
    required_tokens = {
        "verify-build": "verify-build",
        "e0-gate": "e0",
        "full-campaign": "full",
        "summarize": "summarize",
        "export": "export",
    }
    for label, token in required_tokens.items():
        if token not in tokens.split():
            raise ValueError(
                f"owner notebook does not invoke the registered '{label}' "
                "CLI command")
    if not ("run" in tokens.split() and "--mode" in tokens.split()):
        raise ValueError(
            "owner notebook does not invoke the registered 'run --mode' "
            "CLI command")
    return {"cells": len(cells), "code_cells": sum(
        1 for cell in cells if cell.get("cell_type") == "code"),
        "shell_magics": 0, "hardcoded_paths": 0}


# --------------------------------------------------------------------------
# Check-group runner (bounded child processes)
# --------------------------------------------------------------------------

def _run_check_group(name: str, group: dict[str, Any], *,
                     repo_root: str) -> dict[str, Any]:
    missing = [t for t in group["targets"]
               if not os.path.exists(os.path.join(repo_root, t))]
    if missing:
        return {"group": name, "status": "not_run",
                "reason": f"missing targets: {missing}",
                "exit_code": None, "duration_seconds": None,
                "targets": group["targets"]}
    started = time.monotonic()
    command = [sys.executable, "-m", "pytest", "-q", *group["targets"]]
    try:
        proc = subprocess.run(
            command, cwd=repo_root, capture_output=True, text=True,
            timeout=int(group["timeout_seconds"]))
        exit_code = proc.returncode
        tail = (proc.stdout or "")[-2000:]
    except subprocess.TimeoutExpired:
        return {"group": name, "status": "timeout",
                "reason": f"exceeded {group['timeout_seconds']}s",
                "exit_code": None,
                "duration_seconds": round(time.monotonic() - started, 3),
                "targets": group["targets"]}
    duration = round(time.monotonic() - started, 3)
    passed = exit_code == 0 and "no tests ran" not in tail
    return {"group": name, "status": "passed" if passed else "failed",
            "exit_code": exit_code, "duration_seconds": duration,
            "targets": group["targets"], "output_tail": tail[-800:]}


# --------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------

def run_verify_build(data_dir: str, report_dir: str, *, no_updates: bool = True,
                     notebook_path: str | None = None,
                     repo_root: str | None = None,
                     run_checks: bool = True) -> dict[str, Any]:
    """Execute build verification and write the machine-readable report.

    Returns the report dict. Raises nothing for failing evidence — failures
    are recorded and readiness derives as false; structural errors (missing
    bundle, unreadable notebook) raise.
    """
    from bramastra_lab.research.contracts.core import content_identity
    from bramastra_lab.research.campaigns.phases.ops import k8_campaign_config
    from bramastra_lab.research.experience.codec import codec_identity
    from bramastra_lab.research.runtime.provenance import source_identity

    if not no_updates:
        raise ValueError(
            "verify-build enforces zero optimizer commits; --no-updates is "
            "required and cannot be disabled")
    if repo_root is None:
        repo_root = _repo_root()
    report_dir = os.path.abspath(report_dir)
    os.makedirs(report_dir, exist_ok=True)
    target_report = os.path.join(report_dir, REPORT_FILENAME)
    if os.path.exists(target_report):
        raise ValueError(
            f"refusing to overwrite an existing build report: "
            f"{target_report}; use a new report directory per run")

    identities = source_identity()
    closure = identities.get("source_closure_sha256")
    config = k8_campaign_config()
    started = time.monotonic()

    # Exercise: bundle contracts (also proves the data closure).
    exercise_results: dict[str, dict[str, Any]] = {}
    exercise_results["bundle_contracts"] = _exercise_receipt(
        "bundle_contracts", lambda: exercise_bundle_contracts(data_dir))
    exercise_results["campaign_architecture"] = _exercise_receipt(
        "campaign_architecture", exercise_campaign_architecture)
    exercise_results["no_update_boundary"] = _exercise_receipt(
        "no_update_boundary",
        lambda: exercise_no_update_boundary(data_dir))
    exercise_results["live_episode_no_update"] = _exercise_receipt(
        "live_episode_no_update", exercise_live_episode_no_update)
    exercise_results["checkpoint_roundtrip"] = _exercise_receipt(
        "checkpoint_roundtrip", exercise_checkpoint_roundtrip)
    notebook_default = os.path.join(repo_root, "notebooks", "bramastra_k8.ipynb")
    exercise_results["notebook_inputs"] = _exercise_receipt(
        "notebook_inputs",
        lambda: exercise_notebook_inputs(notebook_path or notebook_default))
    exercise_results["integrated_rehearsal"] = _rehearsal_receipt(repo_root)

    # Registered local check groups (bounded children).
    check_results: dict[str, dict[str, Any]] = {}
    if run_checks:
        for name, group in CHECK_GROUPS.items():
            check_results[name] = _run_check_group(
                name, group, repo_root=repo_root)

    # Requirement statuses DERIVED from the evidence above.
    requirements: dict[str, dict[str, Any]] = {}
    for req_id, spec in REQUIREMENT_REGISTRY.items():
        missing: list[str] = []
        failed: list[str] = []
        receipts: list[dict[str, Any]] = []
        for group_name in spec["checks"]:
            receipt = check_results.get(group_name)
            if receipt is None:
                missing.append(f"check:{group_name}:not_run")
                continue
            receipts.append({"kind": "pytest-group",
                             "selector": receipt["targets"],
                             "exit_code": receipt.get("exit_code"),
                             "duration_seconds":
                                 receipt.get("duration_seconds"),
                             "status": receipt["status"]})
            if receipt["status"] != "passed":
                failed.append(f"check:{group_name}:{receipt['status']}")
        for exercise_name in spec["exercises"]:
            receipt = exercise_results.get(exercise_name)
            if receipt is None:
                missing.append(f"exercise:{exercise_name}:not_run")
                continue
            receipts.append({"kind": "real-interface-exercise",
                             "selector": exercise_name,
                             "status": receipt["status"],
                             "evidence": receipt.get("evidence"),
                             "error": receipt.get("error")})
            if receipt["status"] != "passed":
                failed.append(
                    f"exercise:{exercise_name}:{receipt['status']}")
        if missing:
            status = "not_run"
        elif failed:
            status = "fail"
        else:
            status = "pass"
        limitations = list(spec.get("hardware_pending", []))
        requirements[req_id] = {
            "id": req_id,
            "status": status,
            "implemented_symbols": spec["implemented_symbols"],
            "test_selectors": list(spec["checks"]),
            "command_receipts": receipts,
            "production_path_evidence": [
                f"exercise:{name}" for name in spec["exercises"]],
            "limitations": limitations,
        }

    all_pass = all(r["status"] == "pass" for r in requirements.values())
    bundle_receipt = exercise_results["bundle_contracts"]
    bundle_evidence = bundle_receipt.get("evidence") or {}
    report: dict[str, Any] = {
        "schema": BUILD_VERIFICATION_SCHEMA,
        "created_unix": time.time(),
        "duration_seconds": round(time.monotonic() - started, 3),
        "no_updates": True,
        "source_identity": identities,
        "source_closure_sha256": closure,
        "data_identity": bundle_evidence.get("bundle_identity"),
        "config_identity": config.identity(),
        "codec_identity": codec_identity(),
        "check_groups": check_results,
        "exercises": exercise_results,
        "requirements": requirements,
        "runtime_checks_pending": RUNTIME_GATES,
        "optimizer_updates_local": 0,
        "ready_for_owner_experiment": bool(all_pass),
    }
    report["report_identity"] = content_identity(
        {key: value for key, value in report.items()
         if key != "report_identity"})
    with open(target_report, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return report


def _exercise_receipt(name: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        evidence = fn()
        return {"exercise": name, "status": "passed", "evidence": evidence,
                "error": None}
    except Exception as exc:  # noqa: BLE001 - recorded, never raised silently
        return {"exercise": name, "status": "failed", "evidence": None,
                "error": str(exc)[:400]}


def _rehearsal_receipt(repo_root: str) -> dict[str, Any]:
    """Run the bounded integrated no-update rehearsal (F22) if available."""
    try:
        from bramastra_lab.research.campaigns.rehearsal import run_rehearsal
    except Exception as exc:  # noqa: BLE001
        return {"exercise": "integrated_rehearsal", "status": "not_run",
                "evidence": None,
                "error": f"rehearsal runner unavailable: {exc}"[:200]}
    return _exercise_receipt(
        "integrated_rehearsal", lambda: run_rehearsal(repo_root=repo_root))


def _repo_root() -> str:
    import bramastra_lab

    package_init = os.path.abspath(bramastra_lab.__file__)
    return os.path.dirname(os.path.dirname(package_init))


# --------------------------------------------------------------------------
# Evidence-bound readiness gate (replaces the static false disposition)
# --------------------------------------------------------------------------

def load_build_report(report_path: str | None = None,
                      *, repo_root: str | None = None) -> dict[str, Any]:
    """Locate and structurally validate a build verification report."""
    if report_path is None:
        root = repo_root or _repo_root()
        report_path = os.path.join(root, DEFAULT_REPORT_SEARCH)
    if not os.path.exists(report_path):
        raise FileNotFoundError(
            f"no build verification report at {report_path}; run "
            "`python -m bramastra_lab.research.campaigns.k8 verify-build "
            "--data <bundle> --report-dir <new-dir> --no-updates` first")
    with open(report_path, encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("schema") != BUILD_VERIFICATION_SCHEMA:
        raise ValueError(
            f"report schema {report.get('schema')!r} is not "
            f"{BUILD_VERIFICATION_SCHEMA}")
    if not report.get("no_updates") or \
            int(report.get("optimizer_updates_local", -1)) != 0:
        raise ValueError(
            "build report does not certify zero local optimizer commits")
    if not isinstance(report.get("requirements"), dict) or \
            len(report["requirements"]) != 24:
        raise ValueError(
            "build report does not carry the 24 requirement verdicts")
    from bramastra_lab.research.contracts.core import content_identity

    expected = content_identity(
        {key: value for key, value in report.items()
         if key != "report_identity"})
    if expected != report.get("report_identity"):
        raise ValueError(
            "build report identity mismatch: the report body was modified "
            "after verification")
    return report


def implementation_readiness(*, source_identity: str | None = None,
                             report_path: str | None = None,
                             repo_root: str | None = None,
                             current_source_closure: str | None = None,
                             ) -> dict[str, Any]:
    """Derive implementation readiness from verified build evidence (F21).

    Fail-closed: a missing, stale (source closure changed), tampered or
    incomplete report is never ready. Runtime qualification (G01-G04)
    remains a separate owner-E0 requirement and is reported as pending.
    """
    from bramastra_lab.research.runtime.provenance import source_closure_sha256

    context = {"source_identity": source_identity}
    try:
        report = load_build_report(report_path, repo_root=repo_root)
    except Exception as exc:  # noqa: BLE001
        context.update({"ready": False, "blocked_phases": ["E1", "E2", "E3",
                                                           "E4", "E5", "E6"],
                        "reason": f"build evidence unavailable: {exc}"})
        return context
    closure = current_source_closure or source_closure_sha256()
    if str(report.get("source_closure_sha256")) != str(closure):
        context.update({
            "ready": False,
            "blocked_phases": ["E1", "E2", "E3", "E4", "E5", "E6"],
            "reason": "stale build report: source closure changed after "
                      "verification; rerun verify-build",
            "report_source_closure": report.get("source_closure_sha256"),
            "current_source_closure": closure})
        return context
    failing = sorted(
        req_id for req_id, row in report["requirements"].items()
        if row.get("status") != "pass")
    pending = report.get("runtime_checks_pending", [])
    context.update({
        "ready": not failing,
        "blocked_phases": ["E1", "E2", "E3", "E4", "E5", "E6"] if failing
        else [],
        "report_source_closure": report.get("source_closure_sha256"),
        "data_identity": report.get("data_identity"),
        "config_identity": report.get("config_identity"),
        "failing_requirements": failing,
        "runtime_checks_pending": pending})
    return context


def require_implementation_ready(*, source_identity: str | None = None,
                                 report_path: str | None = None) -> dict[str, Any]:
    """Return readiness or raise a typed fail-closed error."""
    report = implementation_readiness(
        source_identity=source_identity, report_path=report_path)
    if not report["ready"]:
        raise ImplementationReadinessError(report)
    return report


class ImplementationReadinessError(RuntimeError):
    """Raised when K8 implementation readiness is not evidence-backed."""

    def __init__(self, report: dict[str, Any]) -> None:
        self.report = dict(report)
        reason = self.report.get("reason") or ", ".join(
            self.report.get("failing_requirements", []))
        super().__init__(
            f"K8 implementation readiness blocked: {reason or 'no evidence'}")


# --------------------------------------------------------------------------
# Final readiness generation (F24): derived from the verified report
# --------------------------------------------------------------------------

def write_final_readiness(report_path: str | None = None, *,
                          out_dir: str | None = None,
                          repo_root: str | None = None) -> dict[str, Any]:
    """Generate BUILD_READINESS.json + HANDOFF.md from build evidence (F24).

    Every field is derived from the verified build report — no hand-maintained
    counts. Raises when the report is absent, tampered or stale; readiness is
    therefore always evidence-bound.
    """
    from bramastra_lab.research.runtime.provenance import source_closure_sha256

    report = load_build_report(report_path, repo_root=repo_root)
    closure = source_closure_sha256()
    if str(report.get("source_closure_sha256")) != str(closure):
        raise ValueError(
            "stale build report: source closure changed; rerun verify-build "
            "before generating the final readiness package")
    requirements = report["requirements"]
    failing = sorted(req_id for req_id, row in requirements.items()
                     if row.get("status") != "pass")
    passed = [req_id for req_id in sorted(requirements)
              if requirements[req_id]["status"] == "pass"]
    not_run = [req_id for req_id in sorted(requirements)
               if requirements[req_id]["status"] == "not_run"]
    readiness = {
        "schema": "bramastra-k8-build-readiness/v2",
        "derived_from": "build_verification.json",
        "report_identity": report["report_identity"],
        "source_revision": report["source_identity"].get("git_head"),
        "implementation_closure_sha256": report["source_closure_sha256"],
        "data_manifest_sha256": report.get("data_identity"),
        "model_config_sha256": report.get("config_identity"),
        "codec_identity": report.get("codec_identity"),
        "schema_identities": {"codec": report.get("codec_identity"),
                              "config": report.get("config_identity")},
        "requirements": {req_id: {
            "status": requirements[req_id]["status"],
            "implemented_symbols": requirements[req_id]["implemented_symbols"],
            "test_selectors": requirements[req_id]["test_selectors"],
            "command_receipts": requirements[req_id]["command_receipts"],
            "production_path_evidence":
                requirements[req_id]["production_path_evidence"],
            "limitations": requirements[req_id]["limitations"],
        } for req_id in sorted(requirements)},
        "executed_commands": [
            {"command": "python -m bramastra_lab.research.campaigns.k8 "
                        "verify-build --data <offline-bundle> --report-dir "
                        "<new-dir> --no-updates",
             "report_identity": report["report_identity"]},
        ],
        "runtime_checks_pending": report.get("runtime_checks_pending", []),
        "optimizer_updates_local": report.get("optimizer_updates_local", 0),
        "ready_for_owner_experiment": not failing,
        "owner_notebook": "notebooks/bramastra_k8.ipynb",
        "data_artifact": {
            "identity": report.get("data_identity"),
            "manifest": "engineering/final_delivery/data/bundle_manifest.json",
            "audit": "engineering/final_delivery/data/bundle_audit.json",
            "generation_command": (
                "python -m bramastra_lab.research.campaigns.k8 prepare "
                "--out <offline-bundle> --training-mechanisms 4096 "
                "--controller-mechanisms 256 --development-mechanisms 256 "
                "--confirmation-mechanisms 128 --tool-mechanisms 4096 "
                "--tool-heldout 256 --meta-train 24 --meta-validate 6 "
                "--meta-confirm 6")},
        "runbook": "engineering/final_delivery/RUN_EXPERIMENT.md",
        "remaining_blockers": failing,
    }
    if out_dir is None:
        out_dir = os.path.dirname(report_path) if report_path else os.path.join(
            _repo_root(), DEFAULT_REPORT_SEARCH.rsplit(os.sep, 1)[0])
    os.makedirs(out_dir, exist_ok=True)
    lines = ["# FINAL-K8 handoff (generated from build evidence)", "",
             f"- Source revision: `{readiness['source_revision']}`",
             f"- Implementation closure: `{readiness['implementation_closure_sha256'][:16]}...`",
             f"- Data bundle identity: `{readiness['data_manifest_sha256']}`",
             f"- Requirements PASS: {len(passed)}/24"
             + (f" (not_run: {', '.join(not_run)})" if not_run else ""),
             f"- Local optimizer commits: {readiness['optimizer_updates_local']}",
             f"- Ready for owner experiment: **{readiness['ready_for_owner_experiment']}**",
             "", "## Requirement verdicts", "",
             "| Requirement | Status | Evidence |", "| --- | --- | --- |"]
    for req_id in sorted(requirements):
        row = requirements[req_id]
        evidence = ", ".join(
            [f"checks:{s}" for s in row["test_selectors"]]
            + [f"exercise:{name}" for name in row["production_path_evidence"]])
        lines.append(f"| {req_id} | {row['status']} | {evidence} |")
    lines += ["", "## Runtime checks pending (owner E0)", ""]
    for gate in report.get("runtime_checks_pending", []):
        lines.append(f"- {gate['id']} (before {gate['required_before']}): "
                     f"{gate['criterion']}")
    lines += ["", "This file is generated by "
              "`bramastra_lab.research.campaigns.verify_build."
              "write_final_readiness`; regenerate instead of editing.",
              ""]
    with open(os.path.join(out_dir, "BUILD_READINESS.json"), "w",
              encoding="utf-8", newline="\n") as handle:
        json.dump(readiness, handle, indent=2, sort_keys=True)
    with open(os.path.join(out_dir, "HANDOFF.md"), "w",
              encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    return readiness


def build_parser() -> "argparse.ArgumentParser":
    import argparse

    parser = argparse.ArgumentParser(
        prog="bramastra-verify-build",
        description="Evidence-backed build verification (zero optimizer commits).",
    )
    parser.add_argument("--data", required=True, help="prepared bundle directory")
    parser.add_argument("--report-dir", required=True,
                        help="NEW directory for build_verification.json")
    parser.add_argument("--no-updates", action="store_true",
                        help="required flag: enforces zero optimizer commits")
    parser.add_argument("--notebook", default=None,
                        help="owner notebook path (default: notebooks/bramastra_k8.ipynb)")
    parser.add_argument("--skip-check-groups", action="store_true",
                        help="internal: exercises only (used by focused tests)")
    return parser


def main(argv: "list[str] | None" = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.no_updates:
        print("error: verify-build requires --no-updates (zero optimizer "
              "commits are enforced)", file=sys.stderr)
        return 2
    report = run_verify_build(
        args.data, args.report_dir, no_updates=True,
        notebook_path=args.notebook,
        run_checks=not args.skip_check_groups)
    failing = sorted(
        req_id for req_id, row in report["requirements"].items()
        if row["status"] != "pass")
    print(json.dumps({
        "status": "VERIFIED" if report["ready_for_owner_experiment"]
        else "NOT_READY",
        "report": os.path.join(os.path.abspath(args.report_dir),
                               REPORT_FILENAME),
        "source_closure_sha256": report["source_closure_sha256"],
        "failing_requirements": failing,
        "runtime_checks_pending": [gate["id"] for gate in
                                   report["runtime_checks_pending"]],
        "optimizer_updates_local": report["optimizer_updates_local"],
    }, indent=2, sort_keys=True))
    return 0 if report["ready_for_owner_experiment"] else 1


if __name__ == "__main__":
    sys.exit(main())
