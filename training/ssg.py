"""Sovereign Scaling Governor for evidence-gated model growth."""

from __future__ import annotations

import hmac
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from anra.anra_paths import (
    CIV_LATEST,
    DEPLOYMENT_BOUNDARY_REPORT,
    IBS_LATEST,
    MEMORY_PROFILE_FRONTIER,
    MODEL_GROWTH_REPORT,
    PROMOTED_RELEASE_MANIFEST,
    SOVEREIGNTY_TEST_REPORT,
    SSG_AUDIT_LOG,
    TOKEN_INVENTORY_MANIFEST,
    TOKENIZER_MANIFEST,
    V2_BRAIN_CHECKPOINT,
)

from training.v2_config import V2_FRONTIER_PARAMETER_COUNT


@dataclass(frozen=True)
class SSGResult:
    allowed: bool
    target_profile: str
    target_params: int
    passed: tuple[str, ...]
    blockers: tuple[str, ...]
    evidence: dict[str, str]
    phase: str
    override_used: bool
    override_reason: str
    generated_at: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class SovereignScalingGovernor:
    REQUIRED_OVERALL = 0.60
    REQUIRED_OWNER_TASK = 0.60
    REQUIRED_IDENTITY = 0.70
    REQUIRED_CIV = 0.88
    SOVEREIGNTY_FLOOR = 0.85
    MAX_PEAK_GB = 13.0
    MIN_CONTINUATION_TOKENS = 21_000_000_000

    @staticmethod
    def _load(path: Path) -> dict | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except Exception:
            return None

    def check(
        self,
        *,
        target_profile: str = "frontier",
        target_params: int = V2_FRONTIER_PARAMETER_COUNT,
        checkpoint_path: str | Path = V2_BRAIN_CHECKPOINT,
        ibs_path: str | Path = IBS_LATEST,
        civ_path: str | Path = CIV_LATEST,
        memory_profile_path: str | Path = MEMORY_PROFILE_FRONTIER,
        growth_report_path: str | Path = MODEL_GROWTH_REPORT,
        token_manifest_path: str | Path = TOKEN_INVENTORY_MANIFEST,
        tokenizer_manifest_path: str | Path = TOKENIZER_MANIFEST,
        phase: str = "training",
        override_authorization: str | None = None,
        override_reason: str = "",
    ) -> SSGResult:
        if phase not in {"growth", "training"}:
            raise ValueError("SSG phase must be 'growth' or 'training'")
        passed: list[str] = []
        blocked: list[str] = []
        paths = {
            "checkpoint": Path(checkpoint_path),
            "release_manifest": PROMOTED_RELEASE_MANIFEST,
            "ibs": Path(ibs_path),
            "civ": Path(civ_path),
            "memory_profile": Path(memory_profile_path),
            "growth_report": Path(growth_report_path),
            "token_manifest": Path(token_manifest_path),
            "tokenizer_manifest": Path(tokenizer_manifest_path),
            "sovereignty_tests": SOVEREIGNTY_TEST_REPORT,
            "deployment_boundary_tests": DEPLOYMENT_BOUNDARY_REPORT,
        }

        for key in ("checkpoint", "release_manifest"):
            if paths[key].exists():
                passed.append(key)
            else:
                blocked.append(f"No promoted {key.replace('_', ' ')} at {paths[key]}")

        ibs = self._load(paths["ibs"])
        if ibs is None:
            blocked.append(f"No IBS results found at {paths['ibs']}")
        else:
            dimensions = ibs.get("dimensions", ibs.get("dimension_scores", {}))
            overall = float(ibs.get("overall", ibs.get("overall_score", 0.0)))
            owner = float(dimensions.get("owner_task", 0.0))
            identity = float(dimensions.get("identity", 0.0))
            seed_count = int(ibs.get("seed_count", len(ibs.get("seed_reports", []))))
            checks = {
                "IBS overall": overall >= self.REQUIRED_OVERALL,
                "owner task": owner >= self.REQUIRED_OWNER_TASK,
                "IBS identity": identity >= self.REQUIRED_IDENTITY,
                "three seeded IBS runs": seed_count >= 3,
            }
            for label, ok in checks.items():
                (passed if ok else blocked).append(label if ok else f"{label} requirement not met")

        civ = self._load(paths["civ"])
        if civ is None:
            blocked.append(f"No CIV measurement found at {paths['civ']}")
        else:
            score = float(civ.get("cosine_similarity", civ.get("score", 0.0)))
            if score < self.SOVEREIGNTY_FLOOR:
                blocked.append(f"SOVEREIGNTY EVENT: CIV {score:.3f} < {self.SOVEREIGNTY_FLOOR:.2f}")
            elif score < self.REQUIRED_CIV:
                blocked.append(f"CIV {score:.3f} < {self.REQUIRED_CIV:.2f}")
            else:
                passed.append("CIV")

        profile = self._load(paths["memory_profile"])
        if profile is None:
            blocked.append(f"No measured frontier memory profile at {paths['memory_profile']}")
        else:
            peak_bytes = float(profile.get("peak_reserved_bytes", 0.0))
            peak_gb = float(profile.get("peak_gb", peak_bytes / 1024**3))
            throughput = float(profile.get("tokens_per_second", 0.0))
            if peak_gb <= 0 or peak_gb > self.MAX_PEAK_GB:
                blocked.append(f"Measured peak memory {peak_gb:.2f} GB is outside frontier budget")
            else:
                passed.append("measured memory")
            if throughput <= 0:
                blocked.append("Measured frontier throughput is missing")
            else:
                passed.append("measured throughput")

        if phase == "training":
            growth = self._load(paths["growth_report"])
            if growth is None:
                blocked.append(f"No model-growth parity report at {paths['growth_report']}")
            elif float(growth.get("parity_cosine", 0.0)) < 0.99:
                blocked.append("Model-growth parity cosine is below 0.99")
            else:
                passed.append("model growth parity")
        else:
            passed.append("growth parity deferred until candidate construction")

        inventory = self._load(paths["token_manifest"])
        if inventory is None:
            blocked.append(f"No licensed token inventory at {paths['token_manifest']}")
        else:
            tokens = int(inventory.get("licensed_tokens", inventory.get("total_tokens", 0)))
            if tokens < self.MIN_CONTINUATION_TOKENS:
                blocked.append(
                    f"Licensed token inventory {tokens:,} < {self.MIN_CONTINUATION_TOKENS:,}"
                )
            else:
                passed.append("licensed token inventory")

        tokenizer_manifest = self._load(paths["tokenizer_manifest"])
        if tokenizer_manifest is None:
            blocked.append(f"No tokenizer manifest at {paths['tokenizer_manifest']}")
        elif (
            int(tokenizer_manifest.get("vocab_size", 0)) != 8209
            or int(tokenizer_manifest.get("schema_version", 0)) < 3
        ):
            blocked.append("Tokenizer manifest is not the canonical 8,209-token V3 contract")
        else:
            passed.append("tokenizer manifest")

        for key, label in (
            ("sovereignty_tests", "sovereignty tests"),
            ("deployment_boundary_tests", "deployment-boundary tests"),
        ):
            report = self._load(paths[key])
            if report is None or not bool(report.get("passed", False)):
                blocked.append(f"No successful {label} evidence at {paths[key]}")
            else:
                passed.append(label)

        release = self._load(paths["release_manifest"])
        if release is not None:
            from evaluation.promotion import verify_release_manifest

            if not verify_release_manifest(release):
                blocked.append("Promoted release manifest signature is invalid")
            else:
                passed.append("signed release manifest")
        rollback = release.get("rollback_checkpoint") if release else None
        if rollback and Path(str(rollback)).exists():
            passed.append("rollback checkpoint")
        else:
            blocked.append("Promoted release has no valid rollback checkpoint")

        expected_owner_token = os.environ.get("ANRA_OWNER_TOKEN", "")
        override_used = bool(
            blocked
            and expected_owner_token
            and override_authorization
            and hmac.compare_digest(expected_owner_token, override_authorization)
            and override_reason.strip()
        )
        result = SSGResult(
            allowed=not blocked or override_used,
            target_profile=target_profile,
            target_params=target_params,
            passed=tuple(passed),
            blockers=tuple(blocked),
            evidence={name: str(path) for name, path in paths.items()},
            phase=phase,
            override_used=override_used,
            override_reason=override_reason if override_used else "",
            generated_at=time.time(),
        )
        SSG_AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with SSG_AUDIT_LOG.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")
        return result
