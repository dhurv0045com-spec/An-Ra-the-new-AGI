"""Bounded synthetic integration smoke for the Signac production campaign on Kaggle TPU.

This launches the real ``run_campaign`` XLA development lane for one update,
then starts a fresh worker group and resumes for one more update. The input is
synthetic and the result is engineering evidence only; it cannot authorize a
research or production run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any


SCHEMA = "anra-signac-kaggle-xla-development-campaign/v1"
TOKENIZER_SCHEMA = "anra-signac-synthetic-tokenizer/v1"
_TOKENS = (
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi",
    "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
    "amber", "birch", "cobalt", "dawn", "ember", "fern", "granite", "harbor",
    "indigo", "juniper", "kepler", "linen", "marble", "nectar", "onyx", "prairie",
    "quartz", "raven", "silver", "timber", "umber", "velvet", "willow", "xenon",
    "yarrow", "zephyr", "acorn", "brook", "cedar", "drift", "elm", "frost",
    "grove", "hemlock", "island", "jasper", "lagoon", "meadow", "north", "opal",
)
_TOKEN_IDS = {token: index + 4 for index, token in enumerate(_TOKENS)}
_TOKENIZER_SHA256 = hashlib.sha256(
    (TOKENIZER_SCHEMA + ":" + " ".join(_TOKENS)).encode("utf-8")
).hexdigest()


@dataclass(frozen=True, slots=True)
class DevelopmentConfig:
    source_tree_sha256: str
    output_dir: str
    run_id: str
    cymek_sha: str
    identity_kind: str
    seed: int = 73_012
    expected_world_size: int = 8
    campaign_tokens: int = 262_144
    sequence_length: int = 4_096
    session: int = 1

    def validate(self) -> None:
        for label, value in (("source tree", self.source_tree_sha256),
                             ("campaign identity", self.cymek_sha)):
            expected_length = 64 if label == "source tree" else 40
            if (not isinstance(value, str) or len(value) != expected_length
                    or any(character not in "0123456789abcdef" for character in value)):
                raise ValueError(f"{label} must be lowercase hexadecimal")
        if (not isinstance(self.run_id, str) or not self.run_id
                or any(character.isspace() for character in self.run_id)):
            raise ValueError("run_id must be a compact nonempty identity")
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("output_dir must be a nonempty path")
        if (self.expected_world_size <= 0 or self.campaign_tokens <= 0
                or self.campaign_tokens % 2 != 0):
            raise ValueError("world size must be positive and campaign token budget positive and even")
        if self.sequence_length != 4_096:
            raise ValueError("the bounded development profile is frozen at context 4096")
        if self.session not in (1, 2):
            raise ValueError("development campaign session must be one or two")


class _SyntheticTokenizer:
    identity = SimpleNamespace(artifact_sha256=_TOKENIZER_SHA256)

    def encode(self, text: str) -> list[int]:
        try:
            return [_TOKEN_IDS[token] for token in text.split()]
        except KeyError as exc:
            raise ValueError("synthetic development input contains an unknown token") from exc


def _synthetic_documents(
    config: DevelopmentConfig, topology: dict[str, Any],
) -> list[dict[str, str]]:
    """Build enough deterministic synthetic lane supply for the exact campaign plan."""

    from v5_training.production_sampler import campaign_microstep_plan

    demand: dict[int, int] = {}
    for bucket, count in campaign_microstep_plan(
        start_tokens=0, campaign_tokens=config.campaign_tokens, topo=topology,
    ):
        demand[bucket] = demand.get(bucket, 0) + count
    documents: list[dict[str, str]] = []
    sequence = 0
    for bucket, required_tokens in sorted(demand.items()):
        documents_needed = math.ceil(required_tokens / bucket)
        content_length = bucket - 2  # BOS and EOS make one exact bucket row.
        for index in range(documents_needed):
            state = (config.seed + sequence * 97_409 + bucket) & 0xFFFFFFFF
            words: list[str] = []
            for _ in range(content_length):
                state = (1_664_525 * state + 1_013_904_223) & 0xFFFFFFFF
                words.append(_TOKENS[(state >> 16) % len(_TOKENS)])
            doc_id = f"signac-xla-dev-b{bucket}-{index:05d}"
            documents.append({
                "doc_id": doc_id,
                "source_id": "signac-xla-development-synthetic-v1",
                "text": " ".join(words),
                "domain": "synthetic",
                "family": "natural",
                "authorization_category": "first-party-development-only",
            })
            sequence += 1
    return documents


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _rank_receipt_sha256(receipt: dict[str, Any]) -> str:
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def aggregate_group_receipts(
    receipts: list[dict[str, Any]], *, expected_world_size: int,
    session: int,
) -> dict[str, Any]:
    """Require a complete, rank-consistent production-campaign worker group."""

    if session not in (1, 2):
        raise ValueError("development campaign group number must be one or two")
    if any(type(row.get("ordinal")) is not int for row in receipts):
        raise ValueError("rank receipts must carry integer ordinals")
    ordinals = sorted(row["ordinal"] for row in receipts)
    expected_ordinals = list(range(expected_world_size))
    if ordinals != expected_ordinals or len(receipts) != expected_world_size:
        raise ValueError(f"rank receipts must cover {expected_ordinals}; got {ordinals}")
    if any(row.get("schema") != SCHEMA or row.get("status") != "PASS"
           for row in receipts):
        raise ValueError("every worker must publish a passing development receipt")
    for row in receipts:
        if row.get("session") != session:
            raise ValueError("rank receipt session differs from the requested worker group")
        if row.get("receipt_sha256") != _rank_receipt_sha256(row):
            raise ValueError("rank receipt hash does not match its contents")
    if any(row.get("world_size") != expected_world_size for row in receipts):
        raise ValueError("rank receipt world size differs from the expected TPU topology")
    runtime_fields = ("python_version", "torch_version", "platform")
    for field in runtime_fields:
        values = {row.get(field) for row in receipts}
        if len(values) != 1 or not isinstance(next(iter(values)), str):
            raise ValueError(f"rank receipts disagree on runtime field {field}")
    xla_contracts = {
        json.dumps({
            "device_type": row.get("xla_status", {}).get("device_type"),
            "world_size": row.get("xla_status", {}).get("world_size"),
            "versions": row.get("xla_status", {}).get("versions"),
        }, sort_keys=True)
        for row in receipts
    }
    if len(xla_contracts) != 1:
        raise ValueError("rank receipts disagree on XLA runtime identity")
    identity_fields = (
        "source_tree_sha256", "run_id", "model_spec_sha256", "execution_mode",
        "campaign_tokens", "updates_executed", "cumulative_tokens", "state_complete",
        "termination", "resumed", "resume_equal", "resume_verification", "checkpoint_head",
        "data_manifest_sha256", "pack_manifest_sha256",
    )
    for field in identity_fields:
        values = {json.dumps(row.get(field), sort_keys=True) for row in receipts}
        if len(values) != 1:
            raise ValueError(f"rank receipts disagree on {field}")
    reference = receipts[0]
    expected_resumed = session == 2
    expected_updates = session
    if (reference["resumed"] is not expected_resumed
            or reference["updates_executed"] != expected_updates
            or reference["state_complete"] is not (session == 2)
            or reference["termination"] != ("COMPLETE" if session == 2 else "MANUAL_BOUNDARY")
            or reference["resume_equal"] is not None
            or reference["resume_verification"] != "DEFERRED_TO_FRESH_WORKER_GROUP"):
        raise ValueError("development campaign did not stop/resume at the expected update boundary")
    expected_tokens = int(reference["campaign_tokens"]) * session // 2
    if reference["cumulative_tokens"] != expected_tokens:
        raise ValueError("development campaign token count differs from the frozen boundary")
    return {
        "schema": SCHEMA,
        "status": "PASS",
        "session": session,
        "world_size": expected_world_size,
        "source_tree_sha256": reference["source_tree_sha256"],
        "run_id": reference["run_id"],
        "model_spec_sha256": reference["model_spec_sha256"],
        "campaign_tokens": reference["campaign_tokens"],
        "data_manifest_sha256": reference["data_manifest_sha256"],
        "pack_manifest_sha256": reference["pack_manifest_sha256"],
        "execution_mode": "xla-development",
        "updates_executed": reference["updates_executed"],
        "cumulative_tokens": reference["cumulative_tokens"],
        "runtime_identity": {
            "python_version": reference["python_version"],
            "torch_version": reference["torch_version"],
            "platform": reference["platform"],
            "xla_status": {
                "device_type": reference["xla_status"]["device_type"],
                "world_size": reference["xla_status"]["world_size"],
                "versions": reference["xla_status"]["versions"],
            },
        },
        "state_complete": reference["state_complete"],
        "resumed": reference["resumed"],
        "resume_equal": reference["resume_equal"],
        "resume_verification": reference["resume_verification"],
        "fresh_worker_group_resume_passed": session == 2,
        "checkpoint_head": reference["checkpoint_head"],
        "rank_receipts": [
            {"ordinal": row["ordinal"], "sha256": row["receipt_sha256"]}
            for row in sorted(receipts, key=lambda item: item["ordinal"])
        ],
        "synthetic_input": True,
        "research_training_authorized": False,
    }


def validate_fresh_worker_group_pair(
    first: dict[str, Any], second: dict[str, Any],
) -> None:
    """Require a resumed group to use the same code, data, and TPU runtime."""

    for field in (
        "source_tree_sha256", "run_id", "model_spec_sha256", "campaign_tokens",
        "data_manifest_sha256", "pack_manifest_sha256", "execution_mode",
        "world_size", "runtime_identity",
    ):
        if first.get(field) != second.get(field):
            raise ValueError(f"fresh worker groups disagree on {field}")
    if first.get("checkpoint_head") == second.get("checkpoint_head"):
        raise ValueError("fresh worker group did not advance the checkpoint head")


def _worker(local_index: int, raw_config: dict[str, Any]) -> None:
    """Run one rank and atomically preserve its success or failure receipt."""

    config = DevelopmentConfig(**raw_config)
    session = int(raw_config["session"])
    rank_dir = Path(config.output_dir) / f"session-{session:02d}"
    rank = int(local_index)
    receipt_path = rank_dir / f"rank-{rank:02d}.json"
    try:
        import torch
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr

        from signac_100m.spec import MODEL_SPEC
        from v5_training.production_entry import frozen_topology, run_campaign

        rank = int(xr.global_ordinal())
        world_size = int(xr.world_size())
        if world_size != config.expected_world_size:
            raise ValueError(
                f"TPU world size {world_size} differs from expected {config.expected_world_size}"
            )
        topology = frozen_topology()
        if topology["replicas"] != world_size:
            raise ValueError("live TPU world differs from frozen Signac topology")
        device = xm.xla_device()
        result = run_campaign(
            documents=_synthetic_documents(config, topology),
            tokenizer=_SyntheticTokenizer(),
            model_spec=MODEL_SPEC,
            run_id=config.run_id,
            seed=config.seed,
            campaign_tokens=config.campaign_tokens,
            max_updates=1,
            store_root=Path(config.output_dir) / "checkpoint_store",
            device=device,
            torch_module=torch,
            xb=object(),
            development_mode=True,
            cymek_sha=config.cymek_sha,
            source_tree_sha256=config.source_tree_sha256,
            execution="xla-development",
        )
        receipt: dict[str, Any] = {
            "schema": SCHEMA,
            "status": "PASS",
            "session": session,
            "ordinal": rank,
            "world_size": world_size,
            "source_tree_sha256": config.source_tree_sha256,
            "run_id": config.run_id,
            "identity_kind": config.identity_kind,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": str(torch.__version__),
            "xla_status": result["xla_status"],
            "execution_mode": result["execution_mode"],
            "mode": result["mode"],
            "model_spec_sha256": result["model_spec_sha256"],
            "campaign_tokens": result["campaign_tokens"],
            "updates_executed": result["updates_executed"],
            "cumulative_tokens": result["cumulative_tokens"],
            "state_complete": result["state_complete"],
            "termination": result["termination"],
            "resumed": result["resumed"],
            "resume_equal": result["resume_equal"],
            "resume_verification": result["resume_verification"],
            "checkpoint_head": result["checkpoint_head"],
            "data_manifest_sha256": result["data_manifest_sha256"],
            "pack_manifest_sha256": result["pack_manifest_sha256"],
            "precision": result["precision"],
            "last_update_receipt": result["last_update_receipt"],
        }
        if (receipt["execution_mode"] != "xla-development"
                or receipt["mode"] != "DEVELOPMENT"
                or receipt["precision"].get("status") != "UNQUALIFIED_DEVELOPMENT_ONLY"
                or receipt["resume_equal"] is not None
                or receipt["resume_verification"] != "DEFERRED_TO_FRESH_WORKER_GROUP"):
            raise ValueError("campaign receipt did not retain development-only gates")
        receipt["receipt_sha256"] = _rank_receipt_sha256(receipt)
        _atomic_json(receipt_path, receipt)
    except Exception as exc:
        _atomic_json(receipt_path, {
            "schema": SCHEMA,
            "status": "FAIL",
            "session": session,
            "ordinal": rank,
            "world_size": config.expected_world_size,
            "source_tree_sha256": config.source_tree_sha256,
            "run_id": config.run_id,
            "error_type": type(exc).__name__,
            "error": str(exc)[:2_000],
        })
        raise


def run(*, source_tree_sha256: str, output_dir: str | Path,
        expected_world_size: int = 8, seed: int = 73_012) -> dict[str, Any]:
    """Run two fresh Kaggle worker groups against the same synthetic checkpoint."""

    from signac_100m.source_identity import build_source_identity

    root = Path(__file__).resolve().parents[1]
    identity = build_source_identity(root)
    if source_tree_sha256 != identity["source_tree_sha256"]:
        raise ValueError("supplied source tree digest does not match the live source bundle")
    try:
        import torch_xla
    except ImportError as exc:
        raise RuntimeError("Kaggle development campaign requires its matched torch_xla runtime") from exc
    if not callable(getattr(torch_xla, "launch", None)):
        raise RuntimeError("installed torch_xla has no launch API")

    from signac_100m.spec import MODEL_SPEC, resource_estimate
    from v5_training.production_entry import frozen_topology

    topology = frozen_topology()
    if expected_world_size != topology["replicas"]:
        raise ValueError("expected worker count differs from the frozen Signac topology")
    campaign_tokens = int(topology["global_tokens_per_update"]) * 2
    output = Path(output_dir).resolve()
    kaggle_working = Path("/kaggle/working").resolve()
    if output != kaggle_working and kaggle_working not in output.parents:
        raise ValueError("development campaign outputs must stay under /kaggle/working")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"development output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    run_id = f"signac_xla_dev_{time.time_ns()}"
    source_commit = identity.get("source_commit")
    if isinstance(source_commit, str) and len(source_commit) == 40:
        cymek_sha = source_commit.lower()
        identity_kind = "git-commit"
    else:
        cymek_sha = hashlib.sha1(
            f"{SCHEMA}:{source_tree_sha256}".encode("ascii")
        ).hexdigest()
        identity_kind = "source-tree-derived-development-id; not a git commit"
    config = DevelopmentConfig(
        source_tree_sha256=source_tree_sha256,
        output_dir=str(output),
        run_id=run_id,
        cymek_sha=cymek_sha,
        identity_kind=identity_kind,
        seed=seed,
        expected_world_size=expected_world_size,
        campaign_tokens=campaign_tokens,
    )
    config.validate()
    try:
        shared_checkpoint_bytes = int(
            resource_estimate(MODEL_SPEC)["checkpoint_bytes_params_plus_moments"]
        )
        required_free = math.ceil(shared_checkpoint_bytes * 2.5)
        if shutil_disk_free(output) < required_free:
            raise RuntimeError("insufficient working disk for two bounded shared checkpoints")
        groups: list[dict[str, Any]] = []
        for session in (1, 2):
            session_dir = output / f"session-{session:02d}"
            session_dir.mkdir()
            raw_config = {**asdict(config), "session": session}
            try:
                torch_xla.launch(_worker, args=(raw_config,))
            except Exception as exc:
                _atomic_json(session_dir / "launch-failure.json", {
                    "schema": SCHEMA,
                    "status": "FAIL",
                    "session": session,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:2_000],
                })
                raise
            receipts = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in sorted(session_dir.glob("rank-*.json"))
            ]
            group = aggregate_group_receipts(
                receipts, expected_world_size=expected_world_size, session=session,
            )
            _atomic_json(session_dir / "aggregate.json", group)
            groups.append(group)
        validate_fresh_worker_group_pair(groups[0], groups[1])
        result = {
            "schema": SCHEMA,
            "status": "PASS",
            "run_id": run_id,
            "source_tree_sha256": source_tree_sha256,
            "source_commit": identity.get("source_commit"),
            "identity_kind": identity_kind,
            "expected_world_size": expected_world_size,
            "model_spec": "Signac M102 (101,790,080 parameters)",
            "campaign_tokens": campaign_tokens,
            "groups": groups,
            "fresh_worker_group_resume_passed": True,
            "synthetic_input": True,
            "production_training_authorized": False,
            "qualification_scope": (
                "development-only production-backend wiring and same-session fresh-worker-group resume; "
                "not target certification, durable-output round-trip, or research training"
            ),
        }
        _atomic_json(output / "aggregate.json", result)
        return result
    except Exception as exc:
        _atomic_json(output / "failure.json", {
            "schema": SCHEMA,
            "status": "FAIL",
            "run_id": run_id,
            "source_tree_sha256": source_tree_sha256,
            "error_type": type(exc).__name__,
            "error": str(exc)[:2_000],
        })
        raise


def shutil_disk_free(path: Path) -> int:
    return int(shutil.disk_usage(path).free)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--output-dir", default="/kaggle/working/signac_100m_xla_development")
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=73_012)
    args = parser.parse_args()
    result = run(
        source_tree_sha256=args.source_tree_sha256,
        output_dir=args.output_dir,
        expected_world_size=args.expected_world_size,
        seed=args.seed,
    )
    print(json.dumps({
        "status": result["status"],
        "run_id": result["run_id"],
        "aggregate": str(Path(args.output_dir) / "aggregate.json"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
