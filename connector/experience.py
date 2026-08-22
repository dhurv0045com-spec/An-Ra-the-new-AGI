"""Cognitive-credit → training bridge: verified experiences, proposals, contracts.

Minimum executable contracts connecting verified runtime failures to auditable
training proposals. Anti-leak rules preserved: proposal generation sees only
ObservedFailure evidence (never evaluator ground truth, never sealed suites).

  VerifiedExperience   one verified corrective event (baseline fails,
                       one variable changes, verifier passes the repair)
  ExperienceBank       JSONL store with query helpers; evidence is primary,
                       labels are a view
  CapabilityContract   what a parent checkpoint is trusted to do
  TrainingProposal     smallest justified intervention: target, protected
                       capabilities, data, falsification condition, lineage
  propose_from_experiences  deterministic baseline proposer (§16 rules):
                       repeated single-variable flips → capability training;
                       decode-rescued failures → NO_TRAINING (policy change)

Training receipts must reference the proposal id; a child without lineage is
not promotion-grade evidence.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class ObservedFailure:
    """What the diagnostic side may see. No evaluator ground truth."""

    task_id: str
    original_input: str
    success_criterion: str
    failed_output: str


@dataclass(frozen=True, slots=True)
class BehavioralImprovementObservation:
    """Parent fails, trained child succeeds. OBSERVATIONAL evidence only:
    useful for capability discovery, training-history analysis, and hypothesis
    generation — it does NOT establish which cognitive variable caused the
    repair (the intervention was 'weights changed through training', not a
    controlled single-variable change). Must never be labeled with a causal
    changed_variable."""

    observation_id: str
    task: ObservedFailure
    parent_checkpoint_sha256: str
    child_checkpoint_sha256: str
    child_output: str
    observed_capability: str        # descriptive, not causal
    source_commit: str | None
    timestamp: str


@dataclass(frozen=True, slots=True)
class VerifiedExperience:
    """A VerifiedInterventionExperience: same parent, same task, same decode,
    one controlled variable changed, baseline fails, intervention succeeds,
    verifier confirms. ONLY these support cognitive-credit causal claims."""
    experience_id: str
    task: ObservedFailure
    parent_checkpoint_sha256: str
    changed_variable: str          # knowledge | plan | tool | decode | context
    intervention_cost: int         # executions the intervention consumed
    corrected_output: str
    variables_held_constant: tuple[str, ...]
    baseline_success: bool         # always False for experiences
    intervention_success: bool     # always True (only verified repairs enter)
    diagnosis_hypothesis: str
    diagnosis_confidence: float
    source_commit: str | None
    timestamp: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def experience_from_runtime(case_id: str, run_result) -> VerifiedExperience:
    """Build an experience from a connector.runtime RunResult whose repair
    verified. Only repaired runs qualify (status == 'repaired')."""
    if run_result.status != "repaired" or not run_result.learning_candidate:
        raise ValueError("only verified repairs become experiences")
    failed = next((s for s in run_result.steps if s.role == "baseline"), None)
    corrected = next((s for s in run_result.steps if s.role == "repair"), None)
    if failed is None or corrected is None:
        raise ValueError("run result lacks baseline/repair steps")
    payload = json.dumps(
        {"task": run_result.task, "changed": run_result.changed_variable},
        sort_keys=True)
    return VerifiedExperience(
        experience_id="ve-" + _sha(payload + corrected.prompt),
        task=ObservedFailure(
            task_id=case_id,
            original_input=failed.prompt,
            success_criterion="verifier token match",
            failed_output=failed.outputs[0] if failed.outputs else ""),
        parent_checkpoint_sha256="",  # filled by the bank when known
        changed_variable=run_result.changed_variable or "unknown",
        intervention_cost=sum(s.n_executions for s in run_result.steps),
        corrected_output=run_result.learning_candidate["verified_output"],
        variables_held_constant=("question", "decode") if run_result.changed_variable not in ("decode",) else ("question", "knowledge"),
        baseline_success=False,
        intervention_success=True,
        diagnosis_hypothesis=f"changing {run_result.changed_variable} flips the verifier",
        diagnosis_confidence=1.0 if run_result.interventions and sum(
            1 for a in run_result.interventions if a.success) == 1 else 0.6,
        source_commit=None,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


class ExperienceBank:
    """JSONL-backed store of verified experiences; evidence is primary."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def add(self, experience: VerifiedExperience) -> str:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(experience.to_json() + "\n")
        return experience.experience_id

    def all(self) -> list[dict]:
        return [json.loads(l) for l in
                self.path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def fixed_by(self, variable: str) -> list[dict]:
        return [e for e in self.all() if e["changed_variable"] == variable]


@dataclass(frozen=True, slots=True)
class CapabilityContract:
    """What a parent checkpoint is currently trusted to do. Children inherit
    this and may not silently regress PROMOTED capabilities."""

    checkpoint_sha256: str
    capabilities: dict[str, str]     # name -> PROMOTED | EXPERIMENTAL | BELOW_FLOOR
    evidence_refs: dict[str, str] = field(default_factory=dict)

    def protected(self) -> list[str]:
        return [c for c, s in self.capabilities.items() if s == "PROMOTED"]


@dataclass(frozen=True, slots=True)
class TrainingProposal:
    proposal_id: str
    source_experience_ids: tuple[str, ...]
    recommendation: str              # CAPABILITY_TRAINING | NO_TRAINING | ...
    target_capability: str
    protected_capabilities: tuple[str, ...]
    hypothesis: str
    competing_hypotheses: tuple[str, ...]
    replay_mix: dict[str, float]     # family -> fraction (exact)
    min_training_change: dict[str, object]  # lr, updates, objective
    falsification_condition: str
    protected_by_contract: str       # contract checkpoint sha
    timestamp: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def propose_from_experiences(
    bank: ExperienceBank, contract: CapabilityContract,
    *, min_support: int = 3,
) -> TrainingProposal | None:
    """Deterministic baseline proposer. Escalates to training only when at
    least `min_support` verified experiences share one changed variable;
    decode-rescued failures alone recommend NO_TRAINING (policy change)."""
    counts: dict[str, int] = {}
    for e in bank.all():
        counts[e["changed_variable"]] = counts.get(e["changed_variable"], 0) + 1
    if not counts:
        return None
    decode_only = set(counts) == {"decode"}
    if decode_only:
        rec = "NO_TRAINING"
        target = "decode_policy"
    else:
        best = max(counts, key=lambda k: counts[k])
        if counts[best] < min_support:
            return None
        rec = "CAPABILITY_TRAINING"
        target = {"knowledge": "context_binding",
                  "plan": "symbolic_composition",
                  "tool": "tool_result_use"}.get(best, best)
    ids = tuple(e["experience_id"] for e in bank.all())
    payload = json.dumps({"ids": ids, "target": target}, sort_keys=True)
    return TrainingProposal(
        proposal_id="tp-" + _sha(payload),
        source_experience_ids=ids,
        recommendation=rec,
        target_capability=target,
        protected_capabilities=tuple(contract.protected()),
        hypothesis=(f"repeated verified {max(counts, key=counts.get)}-intervention "
                    f"flips imply a missing {target} capability"),
        competing_hypotheses=(
            "H2: capability exists but prompt/decode format hides it",
            "H3: selection works but exact-token fidelity fails"),
        replay_mix={"target": 0.45, "retention": 0.55},
        min_training_change={"lr": 3e-5, "objective": "masked LM + balanced replay",
                             "trainable_scope": "all"},
        falsification_condition=("sealed OOD: target improves but any protected "
                                 "capability drops below its floor"),
        protected_by_contract=contract.checkpoint_sha256,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


def link_child(checkpoint_path: str, proposal: TrainingProposal,
               receipt_path: str = "output/lineage.json") -> dict:
    """Training receipt → proposal → experiences: why does this model exist?"""
    lineage = {
        "checkpoint": checkpoint_path,
        "proposal_id": proposal.proposal_id,
        "recommendation": proposal.recommendation,
        "target_capability": proposal.target_capability,
        "protected_capabilities": list(proposal.protected_capabilities),
        "source_experience_count": len(proposal.source_experience_ids),
        "timestamp": proposal.timestamp,
    }
    Path(receipt_path).write_text(json.dumps(lineage, indent=2), encoding="utf-8")
    return lineage
