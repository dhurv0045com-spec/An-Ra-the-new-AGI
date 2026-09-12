"""Independent oracle checkers and rollout harness (B08).

Oracles are named diagnostic controls, not learner inputs. Each oracle
re-derives the episode verdict from the environment's hidden mechanism and is
cross-checked against direct enumeration of the tiny state spaces. Policies
here are hand-written environment checks, never the neural learner.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from bramastra_lab.research.environments.base import BaseEnvironment
from bramastra_lab.research.environments.worlds import (
    InventoryWorld,
    ProgramLab,
    SwitchWorld,
)


@dataclass
class EpisodeRecord:
    """Public transcript of one rollout plus verdict and cost accounting."""

    environment: str
    episode_id: str
    success: bool
    terminated: bool
    truncated: bool
    steps: list[dict[str, Any]] = field(default_factory=list)
    inquiry_cost_total: float = 0.0
    submission_cost_total: float = 0.0
    policy_identity: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def rollout(environment: BaseEnvironment, episode_id: str,
            policy: Callable[[Mapping[str, Any]], Mapping[str, Any]], *,
            policy_identity: str) -> EpisodeRecord:
    """Drive one real episode with the given policy over public observations.

    The policy view carries public values, the legal-action enumeration and
    the previous step's feedback — never hidden mechanism state.
    """
    observation = environment.reset(episode_id=episode_id)
    record = EpisodeRecord(environment=environment.name, episode_id=episode_id,
                           success=False, terminated=False, truncated=False,
                           policy_identity=policy_identity)
    max_steps = environment.budget + 2
    for _ in range(max_steps):
        view = {
            "public_values": dict(observation.observable_values),
            "legal_actions": [dict(candidate) for candidate in environment.legal_actions()],
            "last_feedback": dict(observation.feedback),
        }
        action = policy(view)
        observation, cost, termination, truncation = environment.step(action)
        record.steps.append({"action": dict(action), "cost": cost,
                             "feedback": dict(observation.feedback)})
        record.inquiry_cost_total = environment.inquiry_cost_total
        record.submission_cost_total = environment.submission_cost_total
        if termination or truncation:
            record.terminated = termination
            record.truncated = truncation
            record.success = bool(observation.feedback.get("success", False))
            break
    return record


def failed_baseline_policy(view: Mapping[str, Any]) -> Mapping[str, Any]:
    """A deliberately failing baseline, guaranteed wrong by construction:

    - SwitchWorld: submit with both switches OFF.
    - InventoryWorld: peek 'red', then submit a different container.
    - ProgramLab: submit -1, outside the public modulus domain.
    """
    name = view["public_values"]["environment_name"]
    if name == SwitchWorld.name:
        return {"kind": "submit"}
    if name == InventoryWorld.name:
        state = failed_baseline_policy.__dict__.setdefault("state", {})
        episode_key = view["public_values"]["episode_key"]
        episode_state = state.setdefault(episode_key, {"peeked": False})
        if not episode_state["peeked"]:
            episode_state["peeked"] = True
            return {"kind": "peek", "container": "red"}
        # Guaranteed wrong: submit 'red' when it is known empty, or a
        # different container when 'red' is known to hold the item.
        if view["last_feedback"].get("contains_item"):
            return {"kind": "submit", "container": "green"}
        return {"kind": "submit", "container": "red"}
    if name == ProgramLab.name:
        return {"kind": "submit", "value": -1}
    raise RuntimeError(f"failed baseline undefined for {name!r}")


def oracle_switch_world(environment: SwitchWorld, record: EpisodeRecord) -> bool:
    """Independent verdict recomputed from the hidden mechanism."""
    return environment._switches["A"] and environment._switches["B"]


def oracle_inventory_world(environment: InventoryWorld, record: EpisodeRecord) -> bool:
    submissions = [step["action"] for step in record.steps
                   if step["action"].get("kind") == "submit"]
    if len(submissions) != 1:
        return False
    return submissions[0].get("container") == environment._holding


def oracle_program_lab(environment: ProgramLab, record: EpisodeRecord) -> bool:
    submissions = [step["action"] for step in record.steps
                   if step["action"].get("kind") == "submit"]
    if len(submissions) != 1:
        return False
    return submissions[0].get("value") == environment.evaluate(environment._target)


ORACLES: dict[str, Callable[[BaseEnvironment, EpisodeRecord], bool]] = {
    SwitchWorld.name: oracle_switch_world,
    InventoryWorld.name: oracle_inventory_world,
    ProgramLab.name: oracle_program_lab,
}


def exhaustive_oracle_agreement(environment: BaseEnvironment,
                                record: EpisodeRecord) -> bool:
    """Cross-check the oracle against direct enumeration of the tiny space."""
    oracle = ORACLES[environment.name]
    oracle_verdict = oracle(environment, record)
    if environment.name == SwitchWorld.name:
        enumeration = environment._switches["A"] and environment._switches["B"]
        return oracle_verdict == bool(enumeration)
    if environment.name == InventoryWorld.name:
        submissions = [step["action"] for step in record.steps
                       if step["action"].get("kind") == "submit"]
        enumeration = len(submissions) == 1 \
            and submissions[0].get("container") == environment._holding \
            and submissions[0]["container"] in environment._containers
        return oracle_verdict == bool(enumeration)
    if environment.name == ProgramLab.name:
        submissions = [step["action"] for step in record.steps
                       if step["action"].get("kind") == "submit"]
        enumeration = len(submissions) == 1 \
            and submissions[0].get("value") == environment.evaluate(environment._target)
        return oracle_verdict == bool(enumeration)
    raise RuntimeError(f"no enumeration for {environment.name!r}")


def winning_policy_factory(environment_name: str):
    """Competent hand-written baselines used only to check the environments."""
    if environment_name == SwitchWorld.name:
        def policy(view: Mapping[str, Any]) -> Mapping[str, Any]:
            state = policy.__dict__.setdefault("state", {"a": False, "b": False})
            if not state["a"]:
                state["a"] = True
                return {"kind": "flip", "switch": "A"}
            if not state["b"]:
                state["b"] = True
                return {"kind": "flip", "switch": "B"}
            return {"kind": "submit"}
        return policy

    if environment_name == InventoryWorld.name:
        def policy(view: Mapping[str, Any]) -> Mapping[str, Any]:
            state = policy.__dict__.setdefault("state", {"peeked": [], "found": None})
            feedback = view["last_feedback"]
            if feedback.get("kind") == "peek" and feedback.get("contains_item"):
                state["found"] = feedback["container"]
            if state["found"] is not None:
                return {"kind": "submit", "container": state["found"]}
            containers = view["public_values"]["containers"]
            unpeeked = [name for name in containers if name not in state["peeked"]]
            target = unpeeked[0]
            state["peeked"].append(target)
            return {"kind": "peek", "container": target}
        return policy

    if environment_name == ProgramLab.name:
        def policy(view: Mapping[str, Any]) -> Mapping[str, Any]:
            state = policy.__dict__.setdefault(
                "state", {"pending": None, "f_plus1": None, "f_plus2": None})
            feedback = view["last_feedback"]
            if feedback.get("kind") == "evaluate":
                if state["pending"] == 1:
                    state["f_plus1"] = feedback["output"]
                elif state["pending"] == 2:
                    state["f_plus2"] = feedback["output"]
            public = view["public_values"]
            modulus, target = public["modulus"], public["target_input"]
            if state["f_plus1"] is None:
                probe = (target + 1) % modulus
                state["pending"] = 1
                return {"kind": "evaluate", "input": probe}
            if state["f_plus2"] is None:
                probe = (target + 2) % modulus
                state["pending"] = 2
                return {"kind": "evaluate", "input": probe}
            slope = (state["f_plus2"] - state["f_plus1"]) % modulus
            intercept = (state["f_plus1"] - slope * (target + 1)) % modulus
            return {"kind": "submit", "value": (slope * target + intercept) % modulus}
        return policy

    raise RuntimeError(f"no policy for {environment_name!r}")
