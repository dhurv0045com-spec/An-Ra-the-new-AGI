"""Read-only semantic diagnostics for ecc5953; no training or checkpoint writes.

Run from the repository root. A detected defect is evidence, not a passing
implementation acceptance test. After repair, these flags should become false.
"""
import json
from bramastra_lab.research.cognition import episode as k


class Capture(k.ModelInterface):
    def __init__(self):
        self.prompts = []

    def generate(self, prompt_tokens, *, max_new_tokens):
        self.prompts.append(list(prompt_tokens))
        return {"answer": '{"success_prob":0.5}', "origin": "capture-double"}


def main():
    a = {"feedback": {"kind": "read", "variable": "x", "value": 7}}
    b = {"feedback": {"kind": "read", "variable": "y", "value": 7}}
    records = []
    k.admit_observation_evidence(records, observation=a["feedback"], observation_id="a")
    k.admit_observation_evidence(records, observation={"kind": "read", "variable": "y", "value": 8}, observation_id="b")
    false_conflict = bool(k.mark_conflicts(records))
    temporal = [{"entity": "door", "observed_value": value,
                 "temporal_scope": time, "observation_id": str(time)}
                for time, value in [(0, "closed"), (1, "open")]]
    model = Capture()
    predictor = k.ModelWorldModel(model)
    for value in [0, 1]:
        predictor(state={"goal": {"target": "x"},
                         "history": [{"variable": "x", "value": value}]},
                  action={"kind": "inspect", "variable": "x"}, depth=1)
    states = []
    def world(**kwargs):
        states.append(kwargs)
        return {"feedback": {"changed": True}, "success_prob": 0.5,
                "origin": "capture-double"}
    planner = k.BoundedPlannerAdapter(world_model=world, max_depth=1,
                                      max_nodes=8)
    actions = [{"kind": "inspect", "variable": str(i)} for i in range(8)]
    planner.select(legal_actions=actions, rendered=[], model=model, workspace=[],
                   state_view={"goal": {"target": "x"}, "history": [],
                               "budgets": {"actions_left": 4}})
    roots = [n for n in planner.last_imagined if len(n.action_prefix) == 1]

    # Test an actual depth-2 call. Comparing the first two calls of an eight-
    # root, eight-node search only compares two depth-1 roots (both correctly
    # start from the same real state), so that older check reported a false
    # depth-2 defect without ever expanding a child.
    rollout_calls = []
    def rollout_world(*, state, action, depth):
        rollout_calls.append({"state": state, "action": action, "depth": depth})
        return {"feedback": {"observed": action["variable"]},
                "success_prob": 0.5, "origin": "capture-double"}

    rollout_history = [{"action": {"kind": "inspect", "variable": "prior"},
                        "feedback": {"kind": "observation", "value": "known"}}]
    rollout_actions = [{"kind": "inspect", "variable": "left"},
                       {"kind": "inspect", "variable": "right"}]
    rollout_planner = k.BoundedPlannerAdapter(
        world_model=rollout_world, max_depth=2, max_nodes=3)
    rollout_planner.select(
        legal_actions=rollout_actions, rendered=[], model=model, workspace=[],
        state_view={"goal": {"target": "x"}, "history": rollout_history,
                    "budgets": {"actions_left": 4}})
    depth_two_calls = [call for call in rollout_calls if call["depth"] == 2]

    def lacks_imagined_successor(call):
        history = call["state"].get("history", [])
        return (len(history) <= len(rollout_history)
                or history[:len(rollout_history)] != rollout_history
                or not history[-1].get("imagined"))

    depth_two_reuses_state = (
        not depth_two_calls
        or any(lacks_imagined_successor(call) for call in depth_two_calls))

    result = {
        "schema": "bramastra-cognition-source-diagnostic/v2",
        "baseline_target_revision": "ecc5953",
        "optimizer_updates": 0,
        "defects_observed": {
            "distinct_variables_render_identically": k._compact_history_entry(a) == k._compact_history_entry(b),
            "different_variables_marked_conflicting": false_conflict,
            "ordinary_temporal_change_marked_conflicting": bool(k.mark_conflicts(temporal)),
            "world_prompt_ignores_observation_history": model.prompts[0] == model.prompts[1],
            "depth_two_reuses_untransitioned_state": depth_two_reuses_state,
            "eight_node_search_covers_only_first_root": len(roots) == 1,
        },
        "depth_two_calls_observed": len(depth_two_calls),
        "root_actions_considered": len(roots),
        "legal_actions": len(actions),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
