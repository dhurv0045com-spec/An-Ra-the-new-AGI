"""Production-episode checks for the typed cognition state path."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

from bramastra_lab.research.cognition.episode import (
    Adapter,
    admit_typed_observation,
    mark_conflicts,
    run_episode,
    sync_typed_conflict_states,
)
from bramastra_lab.research.cognition.workspace import CognitiveWorkspace


class TypedCognitionEpisodeTests(unittest.TestCase):
    def test_episode_uses_typed_ledger_and_records_temporal_supersession(self) -> None:
        class Env:
            def __init__(self) -> None:
                self.position = 0

            def reset(self, *, episode_id):
                return SimpleNamespace(
                    observable_values={"question": "door state?"},
                    feedback={"kind": "initial"})

            def legal_actions(self):
                return [{"kind": "inspect"}]

            def step(self, action):
                self.position += 1
                value = "open" if self.position == 1 else "closed"
                feedback = {"kind": "measurement", "variable": "door",
                            "value": value}
                return (SimpleNamespace(feedback=feedback), 0.25,
                        self.position == 2, False)

        class CaptureWorkspacePolicy(Adapter):
            name = "typed-workspace-test"
            uses_workspace = True

            def __init__(self) -> None:
                self.calls = []

            def select(self, *, legal_actions, rendered, model, workspace,
                       state_view):
                self.calls.append({
                    "rendered": list(rendered),
                    "workspace": [dict(row) for row in workspace],
                    "cognition": state_view["cognitive_workspace"],
                })
                return dict(legal_actions[0])

        policy = CaptureWorkspacePolicy()
        trace = run_episode(Env(), policy, model=object(), seed=42,
                            action_budget=2, call_budget=4)

        self.assertEqual(len(policy.calls), 2)
        first_live_input = policy.calls[1]
        self.assertEqual(first_live_input["workspace"][0]["observation_id"], "e1")
        self.assertEqual(first_live_input["cognition"]["evidence"][0]["alias"], "e1")
        self.assertNotIn("source_event_id", str(first_live_input["workspace"]))
        self.assertEqual(trace["cognitive_state"]["step_counter"], 2)
        self.assertEqual(
            [row["conflict_state"] for row in trace["cognitive_state"]["evidence"]],
            ["superseded", "active"])
        self.assertEqual(trace["summary"]["cognition"]["superseded_evidence"], 1)
        self.assertEqual(trace["cognitive_state_identity"],
                         trace["summary"]["cognition"]["state_identity"])

    def test_same_time_conflict_is_mirrored_and_survives_restore(self) -> None:
        typed = CognitiveWorkspace(goal={"task": "resolve"},
                                   success_predicate="verified",
                                   budget=4)
        prompt_projection = []
        for value in ("left", "right"):
            admit_typed_observation(
                typed, prompt_projection,
                observation={"kind": "measurement", "variable": "switch",
                             "value": value, "valid_time": 9},
                observation_id=f"event-{value}")
        pairs = mark_conflicts(prompt_projection)
        sync_typed_conflict_states(typed, prompt_projection)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(
            [row.conflict_state for row in typed.evidence.values()],
            ["conflicting", "conflicting"])
        restored = CognitiveWorkspace.from_dict(typed.to_dict())
        view = restored.rendered_view(budget=4096)
        self.assertEqual([row["status"] for row in view["evidence"]],
                         ["conflicting", "conflicting"])

    def test_conflict_at_one_time_does_not_leave_older_time_active(self) -> None:
        records = [
            {"record_id": "t0", "subject": "switch", "predicate": "state",
             "value": "off", "valid_time": 0, "status": "active"},
            {"record_id": "t1a", "subject": "switch", "predicate": "state",
             "value": "on", "valid_time": 1, "status": "active"},
            {"record_id": "t1b", "subject": "switch", "predicate": "state",
             "value": "off", "valid_time": 1, "status": "active"},
            {"record_id": "t2", "subject": "switch", "predicate": "state",
             "value": "on", "valid_time": 2, "status": "active"},
        ]
        self.assertEqual(len(mark_conflicts(records)), 1)
        by_id = {row["record_id"]: row for row in records}
        self.assertEqual(by_id["t0"]["status"], "superseded")
        self.assertEqual(by_id["t1a"]["supersedes"], "t0")
        self.assertEqual(by_id["t1a"]["status"], "conflicting")
        self.assertEqual(by_id["t1b"]["status"], "conflicting")
        self.assertEqual(by_id["t2"]["status"], "active")
        self.assertEqual(by_id["t2"]["supersedes"], "t1a")


if __name__ == "__main__":
    unittest.main()
