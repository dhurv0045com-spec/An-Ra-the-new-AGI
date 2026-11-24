"""Level-7 Front Door (`Palantiri`), read-only campaign dashboard status, and plugin scaffolder."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from anra_v5.elendril.representation import UTF8

from .archives import (
    export_session_archives,
    mirror_to_working_dir,
    prune_completed_job_artifacts,
    verify_session_archive,
    win_long_path,
)
from .backend import get_backend, list_backends, load_backend_modules
from .custody import (
    CampaignLedgerVerifier,
    ConfirmLock,
    canonical_structural_hash,
    verify_campaign_ledger,
    verify_split_disjointness,
)
from .execution import dump_json_safe, sanitize_for_json
from .instrument import (
    run_level7_runtime_shield_suite,
    validate_byte_decoder_resilience,
    validate_end_to_end_lifecycle,
    validate_fp16_autocast_forward_backward,
)
from .notebooks import build_pinned_kaggle_notebook, verify_pinned_notebook_bytes
from .reporting import render_campaign_dashboard, rescore_compact_analysis_archive
from .runtime import build_x1_spec
from .scoring import decode_generated_bytes, infer_public_byte_bound, score_typed_output

PLANTHIR_LEVEL = 7
PLANTHIR_LEVEL_NAME = "Level 7 — Zero-Late-Crash Pre-Flight, Adversarial Byte/AMP Shield & Crash-Safe Custody"
_VALID_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,48}$")


def level_info() -> dict[str, Any]:
    """Return Level-7 architecture metadata, subsystems, registered backends, capabilities, and quickstart CLI."""

    return {
        "schema": "anra-planthir-level7-v1",
        "level": PLANTHIR_LEVEL,
        "name": PLANTHIR_LEVEL_NAME,
        "subsystems": {
            "archives": "Streaming v6 archives, dependency-aware CheckpointStore collection/pruning, zero-copy hardlinks & win_long_path",
            "custody": "ConfirmLock tamper-evident cross-session gate, CampaignLedgerVerifier & structural AST/graph/Horn split disjointness",
            "notebooks": "Preflight-before-launch Kaggle 2xT4 notebook generator, try/finally crash mirror & byte verifier",
            "reporting": "Analyst-first campaign reports, archive rescoring & live Markdown dashboard",
            "runtime": "ExperimentModule, GenericExperimentTrainer (native AMP), BytePromptTrainer, CampaignOrchestrator",
            "scoring": "Lossy 0xB5-safe byte decoding, exact typed grammar/byte-sequence scoring & TypedByteSequenceKernel",
            "causal_science": "McNemar + Holm-Bonferroni statistics, linear+MLP probe manifolds, N-hop decomposition, convergence guard",
            "studio": "Unified Palantiri facade, read-only status dashboard & runnable plugin scaffolder",
        },
        "registered_backends": sorted(list_backends()),
        "capabilities": [
            "streaming_v6_archives",
            "win_long_path_max_path_bypass",
            "confirm_lock_custody",
            "campaign_ledger_verifier",
            "stage_gates_and_blocked_parent_isolation",
            "live_wall_clock_preemption",
            "u0000_every_architecture_smoke",
            "batched_serial_parity_guard",
            "preflight_before_launch_notebooks",
            "exact_mcnemar_and_holm_bonferroni",
            "nonlinear_mlp_probe_comparison",
            "n_hop_subgoal_decomposition",
            "cli_level_status_verify_archive_scaffold",
            "exact_typed_byte_sequence_scoring",
            "canonical_ast_graph_horn_split_disjointness",
            "dependency_aware_leaf_optimizer_compaction",
            "cuda_safe_cpu_representation_handoff",
            "trainer_post_step_and_loop_hooks",
            "u0000_fp16_autocast_gate",
            "lossy_byte_decoding_shield",
            "sanitize_for_json_telemetry",
            "archive_full_evidence_before_summary_cap",
            "notebook_try_finally_crash_mirror",
            "trainer_native_amp_grad_scaler",
            "wave_crash_partial_receipt_persistence",
            "cli_shield_runtime_suite",
        ],
        "quickstart_cli": [
            "python -m anra_v5.palantiri level",
            "python -m anra_v5.palantiri doctor",
            "python -m anra_v5.palantiri calibrate",
            "python -m anra_v5.palantiri shield",
            "python -m anra_v5.palantiri status --work-root <dir>",
            "python -m anra_v5.palantiri verify-archive --archive <zip>",
            "python -m anra_v5.palantiri scaffold --name <backend_id> --out-dir <dir>",
        ],
    }


def _read_opt_json(path: Path) -> Any:
    lp = win_long_path(path)
    if not lp.is_file():
        return None
    try:
        return json.loads(lp.read_text(encoding="utf-8"))
    except Exception:
        return None


def inspect_campaign_status(work_root: Path | str) -> dict[str, Any]:
    """Read-only inspection of a Level-7 CampaignOrchestrator / run work directory plus Markdown dashboard."""

    root = Path(work_root)
    lp_root = win_long_path(root)
    artifact_names = (
        "CAMPAIGN_STATUS.json",
        "STAGE_GATES.json",
        "BLOCKED_JOBS.json",
        "U0000_SMOKE_RECEIPT.json",
        "STAGE5_PREFLIGHT_RECEIPT.json",
        "CONFIRM_LOCK.json",
        "PENDING_QUEUE.json",
        "FINAL_STATUS.json",
        "DELIVERY_RECEIPT.json",
        "DISPATCH_AUDIT.json",
        "REGISTERED_PLAN_STATUS.json",
        "COMPLETED_JOBS.json",
        "JOB_RECEIPTS.json",
    )
    docs: dict[str, Any] = {}
    artifacts_present: list[str] = []
    if lp_root.is_dir():
        for name in artifact_names:
            val = _read_opt_json(root / name)
            if val is not None:
                docs[name] = val
                artifacts_present.append(name)

    c_stat = docs.get("CAMPAIGN_STATUS.json") or {}
    r_stat = docs.get("REGISTERED_PLAN_STATUS.json") or {}
    p_queue = docs.get("PENDING_QUEUE.json") or {}
    b_jobs = docs.get("BLOCKED_JOBS.json") or {}
    s_gates = docs.get("STAGE_GATES.json") or {}
    u0 = docs.get("U0000_SMOKE_RECEIPT.json")
    f_stat = docs.get("FINAL_STATUS.json") or {}
    d_rec = docs.get("DELIVERY_RECEIPT.json") or {}
    comp_doc = docs.get("COMPLETED_JOBS.json") or {}

    completed_jobs = list(
        c_stat.get("completed_jobs")
        or comp_doc.get("completed_jobs")
        or r_stat.get("completed_jobs")
        or []
    )
    pending_jobs = list(
        p_queue.get("pending_job_ids")
        or r_stat.get("deferred_pending_job_ids")
        or []
    )
    blocked_jobs = list(
        b_jobs.get("blocked_job_ids")
        or r_stat.get("blocked_job_ids")
        or []
    )
    failed_jobs = list(r_stat.get("failed_job_ids") or [])
    total_jobs = int(
        c_stat.get("total_jobs")
        or r_stat.get("total_jobs")
        or (len(completed_jobs) + len(pending_jobs) + len(blocked_jobs) + len(failed_jobs))
    )

    lock_report: dict[str, Any] | None = None
    if (root / "CONFIRM_LOCK.json").is_file():
        lock_report = ConfirmLock(root / "CONFIRM_LOCK.json").inspect(completed_jobs)
    elif isinstance(c_stat.get("confirm_lock"), Mapping):
        lock_report = dict(c_stat["confirm_lock"])

    archives: dict[str, str] = {}
    if isinstance(d_rec.get("archives"), Mapping):
        archives = {str(k): str(v) for k, v in d_rec["archives"].items()}
    else:
        for key, field in (("analysis", "analysis_zip"), ("summary", "summary_zip"), ("evidence", "evidence_zip"), ("full", "full_zip")):
            if isinstance(f_stat.get(field), str):
                archives[key] = str(root / f_stat[field])
        exports_dir = root / "exports"
        if not archives and win_long_path(exports_dir).is_dir():
            for zp in sorted(win_long_path(exports_dir).glob("*.zip")):
                archives[zp.stem] = str(exports_dir / zp.name)

    campaign_id = str(
        c_stat.get("campaign_id")
        or r_stat.get("campaign_id")
        or f_stat.get("prefix")
        or root.name
    )
    if not lp_root.exists() or not artifacts_present:
        status = "NOT_STARTED"
    else:
        status = str(c_stat.get("status") or r_stat.get("status") or f_stat.get("status") or "UNKNOWN")

    if not lp_root.exists() or not artifacts_present:
        next_action = "INITIALIZE_CAMPAIGN_OR_CHECK_WORK_ROOT"
    elif failed_jobs:
        next_action = "INSPECT_FAILED_JOBS_AND_LOGS"
    elif pending_jobs or blocked_jobs or (lock_report is not None and lock_report.get("pending_confirm_jobs")):
        next_action = "RESUME_PENDING_JOBS_WITH_PREVIOUS_ARCHIVE"
    elif status == "COMPLETE" and not archives:
        next_action = "EXPORT_SESSION_ARCHIVES"
    elif status == "COMPLETE":
        next_action = "VERIFY_ARCHIVES_AND_RUN_CONFIRM"
    else:
        next_action = "RUN_DOCTOR_AND_INSPECT_ARTIFACTS"

    payload: dict[str, Any] = {
        "schema": "anra-planthir-campaign-dashboard-v1",
        "mutates_files": False,
        "exists": lp_root.exists(),
        "work_root": str(root),
        "campaign_id": campaign_id,
        "status": status,
        "counts": {
            "total": total_jobs,
            "completed": len(completed_jobs),
            "pending": len(pending_jobs),
            "blocked": len(blocked_jobs),
            "failed": len(failed_jobs),
        },
        "completed_jobs": completed_jobs,
        "pending_jobs": pending_jobs,
        "blocked_jobs": blocked_jobs,
        "failed_jobs": failed_jobs,
        "stage_gates": dict(s_gates.get("stage_gate_results") or r_stat.get("stage_gate_results") or {}),
        "u0000_smoke": u0,
        "confirm_lock": lock_report,
        "archives": archives,
        "dispatch_audit": docs.get("DISPATCH_AUDIT.json"),
        "artifacts_present": artifacts_present,
        "recommended_next_action": next_action,
    }
    payload["dashboard_markdown"] = render_campaign_dashboard(payload)
    return payload


def scaffold_experiment_module(
    name: str,
    out_dir: Path | str,
    *,
    overwrite: bool = False,
    include_verify_module: bool = False,
) -> dict[str, Any]:
    """Generate a runnable, Level-6 compliant ExperimentModule package + pytest smoke test."""

    if not isinstance(name, str) or not _VALID_NAME_RE.fullmatch(name):
        raise ValueError(
            f"invalid backend identifier {name!r}: expected lowercase snake_case matching ^[a-z][a-z0-9_]{{1,48}}$"
        )

    base = Path(out_dir)
    pkg_dir = base / name
    init_file = pkg_dir / "__init__.py"
    mod_file = pkg_dir / "module.py"
    verify_file = pkg_dir / "verify.py"
    test_file = base / f"test_{name}.py"

    targets = [init_file, mod_file, test_file]
    if include_verify_module:
        targets.append(verify_file)
    if not overwrite:
        for p in targets:
            if win_long_path(p).exists():
                raise FileExistsError(f"refusing to overwrite existing scaffold file: {p}")

    win_long_path(pkg_dir).mkdir(parents=True, exist_ok=True)
    win_long_path(init_file).write_text(
        f'"""Planthir Level-6 plugin package for `{name}`."""\n\n'
        "from .module import BACKEND_ID, build_module, confirm_lock, register_planthir_backends\n\n"
        '__all__ = ["BACKEND_ID", "build_module", "confirm_lock", "register_planthir_backends"]\n',
        encoding="utf-8",
    )
    if include_verify_module:
        verify_src = "\n".join(
            [
                f'"""Independent receipt and study claim verifier for `{name}`."""',
                "",
                "from __future__ import annotations",
                "",
                "from typing import Any, Mapping, Sequence",
                "",
                "from anra_v5.palantiri import CampaignLedgerVerifier, verify_session_archive",
                "",
                "",
                "def verify_campaign_claims(",
                "    job_receipts: Mapping[str, Mapping[str, Any]],",
                "    expected_job_ids: Sequence[str] = (),",
                ") -> dict[str, Any]:",
                '    """Audit completed campaign records and release gates before unsealing confirm."""',
                "    ledger = CampaignLedgerVerifier.verify(",
                "        expected_job_ids=list(expected_job_ids),",
                "        job_receipts=dict(job_receipts),",
                "    )",
                "    return {",
                '        "passed": bool(ledger.get("verified", False)),',
                '        "ledger": ledger,',
                '        "records_audited": len(job_receipts),',
                "    }",
                "",
                "",
                '__all__ = ["CampaignLedgerVerifier", "verify_campaign_claims", "verify_session_archive"]',
                "",
            ]
        )
        win_long_path(verify_file).write_text(verify_src, encoding="utf-8")

    net_cls = "".join(part.capitalize() for part in name.split("_")) + "Net"
    mod_src = "\n".join(
        [
            f'"""Planthir Level-6 ExperimentModule for `{name}`."""',
            "",
            "from __future__ import annotations",
            "",
            "from pathlib import Path",
            "from typing import Any, Mapping, Sequence",
            "",
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
            "",
            "from anra_v5.planthir import (",
            "    ConfirmLock,",
            "    ExperimentModule,",
            "    GenericDatasetBundle,",
            "    StaticPromptKernel,",
            ")",
            "",
            f'BACKEND_ID = "{name}"',
            "",
            "",
            f"class {net_cls}(nn.Module):",
            "    def __init__(self, width: int = 16, seed: int = 7) -> None:",
            "        super().__init__()",
            "        torch.manual_seed(int(seed))",
            "        self.fc1 = nn.Linear(4, int(width))",
            "        self.fc2 = nn.Linear(int(width), 2)",
            "",
            "    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:",
            "        h = torch.tanh(self.fc1(x))",
            "        return self.fc2(h), h",
            "",
            "",
            "def _build_dataset(cfg: Mapping[str, Any], cpu_canary: bool) -> dict[str, Any]:",
            "    n_tr = 8 if cpu_canary else int(cfg.get('num_samples', 32))",
            "    n_ev = max(4, n_tr // 2)",
            "",
            "    def _mk(prefix: str, count: int) -> list[dict[str, Any]]:",
            "        return [",
            "            {",
            "                'world_id': f'{prefix}_{idx}',",
            "                'sig': f'{prefix}_{idx}',",
            "                'hops': 1,",
            "                'feat': [1.0 if idx % 2 == 0 else -1.0, 0.25, -0.15, 0.1],",
            "                'label': idx % 2,",
            "                'hidden': {'expected': 'A' if idx % 2 == 0 else 'B'},",
            "            }",
            "            for idx in range(count)",
            "        ]",
            "",
            "    tr = _mk('tr', n_tr)",
            "    ev = _mk('ev', n_ev)",
            "    mh = _mk('mh', 2)",
            "    pr = _mk('pr', 4)",
            "    return {",
            "        'train_data': GenericDatasetBundle(",
            f"            dataset_name='{name}_v1',",
            "            items=tuple(tr),",
            f"            data_identity='{name}-dataset-v1',",
            "            manifest={'train': len(tr), 'eval': len(ev)},",
            "        ),",
            "        'train_worlds': tr,",
            "        'eval_worlds': ev,",
            "        'multihop_worlds': mh,",
            "        'probe_train_worlds': pr,",
            "    }",
            "",
            "",
            "def _loss_fn(model: nn.Module, chunk: Sequence[Mapping[str, Any]], recipe: Any) -> torch.Tensor:",
            "    device = next(model.parameters()).device",
            "    x = torch.tensor([row['feat'] for row in chunk], dtype=torch.float32, device=device)",
            "    y = torch.tensor([row['label'] for row in chunk], dtype=torch.int64, device=device)",
            "    logits, _ = model(x)",
            "    return F.cross_entropy(logits, y)",
            "",
            "",
            "def build_module() -> ExperimentModule:",
            "    return ExperimentModule(",
            "        backend_id=BACKEND_ID,",
            "        architectures=('baseline', 'treatment'),",
            "        objectives=('ce',),",
            f"        model_factory=lambda _arch, cfg, seed: {net_cls}(int(cfg.get('width', 16)), seed),",
            "        dataset_factory=_build_dataset,",
            "        loss_fn=_loss_fn,",
            "        kernel_factory=lambda: StaticPromptKernel(action_labels=('A', 'B')),",
            "        action_labels=('A', 'B'),",
            "        default_model_config={'width': 16},",
            "        treatment_model_config={'width': 16},",
            "        extra_model_keys=('width',),",
            "    )",
            "",
            "",
            "def confirm_lock(work_root: Path | str) -> ConfirmLock:",
            '    return ConfirmLock(Path(work_root) / "CONFIRM_LOCK.json")',
            "",
            "",
            "def register_planthir_backends() -> ExperimentModule:",
            "    return build_module().register()",
            "",
            "",
            "module = register_planthir_backends()",
            "",
        ]
    )
    win_long_path(mod_file).write_text(mod_src, encoding="utf-8")

    test_src = "\n".join(
        [
            f'"""Smoke test for scaffolded Level-6 module `{name}`."""',
            "",
            "from __future__ import annotations",
            "",
            "from pathlib import Path",
            "",
            "from anra_v5.palantiri import planthir_doctor",
            f"from {name}.module import BACKEND_ID, confirm_lock, register_planthir_backends",
            "",
            "",
            f"def test_{name}_level6_scaffold(tmp_path: Path) -> None:",
            "    mod = register_planthir_backends()",
            "    assert mod.backend_id == BACKEND_ID",
            "    doc = planthir_doctor()",
            "    assert doc['green_signal'] is True and BACKEND_ID in doc['checks']['backends_and_contracts']['registered']",
            "    lock = confirm_lock(tmp_path)",
            "    lock.seal(['job_a'])",
            "    assert lock.unlock_if_ready(['job_a'])['unlocked'] is True",
            "",
        ]
    )
    win_long_path(test_file).write_text(test_src, encoding="utf-8")

    return {
        "schema": "anra-planthir-scaffold-v1",
        "backend_id": name,
        "out_dir": str(base),
        "module_dotted_path": f"{name}.module",
        "created_files": [
            str(init_file),
            str(mod_file),
            *([str(verify_file)] if include_verify_module else []),
            str(test_file),
        ],
        "next_command": f"python -m anra_v5.palantiri --backend-module {name}.module doctor",
    }


class Palantiri:
    """Unified Level-6 Front Door exposing experiment authoring, custody, status & verification."""

    level = PLANTHIR_LEVEL
    level_name = PLANTHIR_LEVEL_NAME

    @staticmethod
    def level_info() -> dict[str, Any]:
        return level_info()

    @staticmethod
    def doctor() -> dict[str, Any]:
        from .agent_tools import planthir_doctor

        return planthir_doctor()

    @staticmethod
    def calibrate() -> dict[str, Any]:
        from .instrument import run_fault_injection_suite

        return run_fault_injection_suite()

    @staticmethod
    def agent_info() -> dict[str, Any]:
        from .agent_tools import agent_info

        return agent_info()

    @staticmethod
    def backends(modules: Sequence[str] = ()) -> dict[str, Any]:
        if modules:
            load_backend_modules(modules)
        return list_backends()

    @staticmethod
    def get_backend(backend_id: str) -> Any:
        return get_backend(backend_id)

    @staticmethod
    def experiment(*args: Any, **kwargs: Any) -> Any:
        from .runtime import ExperimentModule

        return ExperimentModule(*args, **kwargs)

    @staticmethod
    def campaign(*args: Any, **kwargs: Any) -> Any:
        from .runtime import CampaignOrchestrator

        return CampaignOrchestrator(*args, **kwargs)

    @staticmethod
    def confirm_lock(lock_path: Path | str) -> ConfirmLock:
        return ConfirmLock(lock_path)

    @staticmethod
    def status(work_root: Path | str) -> dict[str, Any]:
        return inspect_campaign_status(work_root)

    @staticmethod
    def dashboard(work_root: Path | str) -> str:
        return str(inspect_campaign_status(work_root)["dashboard_markdown"])

    @staticmethod
    def export_archives(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return export_session_archives(*args, **kwargs)

    @staticmethod
    def verify_archive(archive_path: Path | str, **kwargs: Any) -> dict[str, Any]:
        return verify_session_archive(archive_path, **kwargs)

    @staticmethod
    def verify_ledger(**kwargs: Any) -> dict[str, Any]:
        return verify_campaign_ledger(**kwargs)

    @staticmethod
    def rescore(archive_path: Path | str, output_dir: Path | str) -> dict[str, Any]:
        return rescore_compact_analysis_archive(archive_path, output_dir)

    @staticmethod
    def kaggle_notebook(**kwargs: Any) -> dict[str, Any]:
        return build_pinned_kaggle_notebook(**kwargs)

    @staticmethod
    def verify_notebook(notebook_path: Path | str, expected_notebook: Mapping[str, Any]) -> dict[str, Any]:
        return verify_pinned_notebook_bytes(notebook_path, expected_notebook)

    @staticmethod
    def prune_job_artifacts(job_root: Path | str, **kwargs: Any) -> dict[str, Any]:
        return prune_completed_job_artifacts(job_root, **kwargs)

    @staticmethod
    def mirror_to_working(src_paths: Sequence[Path | str], working_dir: Path | str) -> list[dict[str, str]]:
        return mirror_to_working_dir(src_paths, working_dir)

    @staticmethod
    def build_x1_spec(architecture: str, **kwargs: Any) -> Any:
        return build_x1_spec(architecture, **kwargs)

    @staticmethod
    def scaffold(
        name: str,
        out_dir: Path | str,
        *,
        overwrite: bool = False,
        include_verify_module: bool = False,
    ) -> dict[str, Any]:
        return scaffold_experiment_module(
            name,
            out_dir,
            overwrite=overwrite,
            include_verify_module=include_verify_module,
        )

    @staticmethod
    def score_typed_output(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return score_typed_output(*args, **kwargs)

    @staticmethod
    def infer_public_byte_bound(*args: Any, **kwargs: Any) -> int:
        return infer_public_byte_bound(*args, **kwargs)

    @staticmethod
    def structural_hash(record: Mapping[str, Any], *, custom_hash_fn: Any = None) -> str | None:
        return canonical_structural_hash(record, custom_hash_fn=custom_hash_fn)

    @staticmethod
    def verify_splits(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return verify_split_disjointness(*args, **kwargs)

    @staticmethod
    def validate_fp16_autocast(model: Any, loss_fn: Any, **kwargs: Any) -> dict[str, Any]:
        return validate_fp16_autocast_forward_backward(model, loss_fn, **kwargs)

    @staticmethod
    def decode_lossy(token_ids: Sequence[Any], *, strip_special: bool = True, errors: str = "replace") -> tuple[str, bool]:
        return UTF8.decode_lossy(token_ids, strip_special=strip_special, errors=errors)

    @staticmethod
    def decode_generated_bytes(token_ids: Sequence[Any], **kwargs: Any) -> dict[str, Any]:
        return decode_generated_bytes(token_ids, **kwargs)

    @staticmethod
    def validate_byte_decoder(decode_fn: Any = None) -> dict[str, Any]:
        return validate_byte_decoder_resilience(decode_fn=decode_fn)

    @staticmethod
    def validate_lifecycle(policy: Any, kernel: Any, worlds: Sequence[Mapping[str, Any]], **kwargs: Any) -> dict[str, Any]:
        return validate_end_to_end_lifecycle(policy, kernel, worlds, **kwargs)

    @staticmethod
    def runtime_shield() -> dict[str, Any]:
        return run_level7_runtime_shield_suite()

    @staticmethod
    def sanitize_json(obj: Any) -> Any:
        return sanitize_for_json(obj)

    @staticmethod
    def dump_json_safe(obj: Any, *, indent: int | None = 2) -> str:
        return dump_json_safe(obj, indent=indent)


__all__ = [
    "PLANTHIR_LEVEL",
    "PLANTHIR_LEVEL_NAME",
    "Palantiri",
    "inspect_campaign_status",
    "level_info",
    "scaffold_experiment_module",
]
