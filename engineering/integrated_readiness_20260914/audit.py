"""Bounded cross-component review; no optimizer updates or model training.

Run from repository root with PYTHONPATH=. and --output NEW_PATH.
Existing reports are never overwritten. `failed` means a contract failure,
not a failing training experiment. Source hashes identify uncommitted inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


SOURCES = (
    "bramastra_lab/research/cognition/episode.py",
    "bramastra_lab/research/campaigns/phases/e2.py",
    "bramastra_lab/research/campaigns/phases/compiler.py",
    "bramastra_lab/research/learning/k8_scoring.py",
    "bramastra_lab/research/campaigns/phases/e5.py",
    "bramastra_lab/research/campaigns/trial_service.py",
    "tests/test_research_k8_foundation.py",
    "bramastra_lab/research/environments/k8_live.py",
    "bramastra_lab/research/data/k8_bundle.py",
)


def identities():
    return {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in SOURCES if Path(name).is_file()}


def checks():
    from bramastra_lab.research.cognition import episode as k
    from bramastra_lab.research.campaigns.phases import compiler
    from bramastra_lab.research.environments.k8_live import build_live_env, generate_live_mechanism

    results = []

    def record(name, passed, details):
        results.append({"criterion": name, "status": "passed" if passed else "failed",
                        "details": details})

    class Capture(k.ModelInterface):
        def __init__(self):
            self.prompts = []

        def generate(self, prompt_tokens, *, max_new_tokens):
            self.prompts.append(list(prompt_tokens))
            return {"answer": '{"feedback":{},"success_prob":0.5,"value":0}',
                    "origin": "review-capture-double"}

    # Test the actual planner -> predictor -> renderer chain, not predictor alone.
    captured = []
    for observation in (0, 1):
        model = Capture()
        planner = k.BoundedPlannerAdapter(world_model=k.ModelWorldModel(model), max_nodes=2)
        planner.select(legal_actions=[{"kind": "inspect", "variable": "x"}],
                       rendered=[], model=model, workspace=[],
                       state_view={"goal": {"target": "x"},
                                   "history": [{"action": {"kind": "inspect", "variable": "x"},
                                                "feedback": {"value": observation}}],
                                   "workspace": [],
                                   "budgets": {"actions_left": 4, "calls_left": 16, "nodes_left": 2}})
        captured.append(model.prompts)
    record("planner_preserves_received_evidence", captured[0] != captured[1],
           {"prompts_equal": captured[0] == captured[1], "calls_per_case": list(map(len, captured))})

    # Identical candidate prefixes destroy action discrimination before learning.
    row = {"public": {"goal": "inspect x"}, "answer": "1",
           "family": "rule-inquiry", "pool": "training", "mechanism_id": "review-only",
           "canonical_identity": "review-only", "queries": [
               {"kind": "inspect", "variable": "x"}, {"kind": "inspect", "variable": "y"}],
           "history": [{"action": {"kind": "inspect", "variable": "x"},
                        "feedback": {"kind": "observation", "variable": "x", "value": 1}}]}
    batch = compiler.build_batch_for_trajectory(row)
    channels = compiler.compile_channels_for_row(row, batch,
        arm_weights={"action": 0.5}, arm_enabled=frozenset({"action"}))
    candidates = channels["action"]["candidates"]
    record("compiler_preserves_distinct_action_candidates", len({tuple(c) for c in candidates}) == 2,
           {"candidates": candidates, "prefix_tokens": channels["action"]["prefix_tokens"]})

    # A typed submission must survive encode/decode and the actual environment.
    outcomes = {}
    for family in ("rule-inquiry", "inventory", "program"):
        mechanism = generate_live_mechanism(family, 0, seed=99)
        env = build_live_env(mechanism, budget=6, seed=99)
        env.reset(episode_id=f"review-{family}")
        answer = dict(env.oracle_answer())  # Diagnostic ceiling only, never learner input.
        code = k.encode_action_code(answer, env.legal_actions())
        restored, _ = k.decode_action_code(json.loads(code), env.legal_actions())
        outcomes[family] = {"expected": answer, "restored": restored,
                            "equal": answer == restored}
    record("typed_submission_roundtrip_all_families", all(x["equal"] for x in outcomes.values()), outcomes)

    # Unknown is not a calibrated success forecast.
    class Empty(Capture):
        def generate(self, prompt_tokens, *, max_new_tokens):
            return {"answer": "{}", "origin": "review-capture-double"}
    prediction = k.ModelWorldModel(Empty())(state={"goal": {}}, action={"kind": "inspect"}, depth=1)
    record("empty_prediction_is_unknown", prediction.get("prediction_failed") is True or prediction.get("success_prob") is None,
           prediction)

    # The learned adapter needs the public code-to-action map it is asked to use.
    policy_prompts = []
    class Choose(k.ModelInterface):
        def generate(self, prompt_tokens, *, max_new_tokens):
            policy_prompts.append(list(prompt_tokens))
            return {"answer": '{"a":0}', "origin": "review-capture-double"}
    rendered, _ = k.render_public_state(goal={"target": "x"}, history=[])
    for actions in ([{"kind": "inspect", "variable": "x"}], [{"kind": "inspect", "variable": "y"}]):
        k.LearnedPolicyAdapter().select(legal_actions=actions, rendered=rendered,
                                      model=Choose(), workspace=[], state_view={"goal": {"target": "x"}})
    record("policy_input_identifies_legal_action_mapping", policy_prompts[0] != policy_prompts[1],
           {"same_prompt_different_action_meaning": policy_prompts[0] == policy_prompts[1]})

    # Flipping an unobservable target changes the answer while every legal
    # nonterminal observation stays unchanged. This tests task identifiability,
    # not model skill. Both worlds use the same reset RNG and public mechanism.
    import copy
    mechanism = generate_live_mechanism("rule-inquiry", 0, seed=99)
    twin = copy.deepcopy(mechanism)
    twin["rule"]["target_value"] = not mechanism["rule"]["target_value"]
    twin["answer"] = str(twin["rule"]["target_value"]).lower()
    traces, oracle_answers = [], []
    for world in (mechanism, twin):
        env = build_live_env(world, budget=20, seed=99)
        initial = env.reset(episode_id="same-public-episode")
        public_trace = [dict(initial.observable_values), dict(initial.feedback)]
        for action in env.legal_actions():
            if action.get("kind") == "submit":
                continue
            observation, _, _, _ = env.step(action)
            public_trace.append(dict(observation.feedback))
        traces.append(public_trace)
        oracle_answers.append(dict(env.oracle_answer()))
    ambiguous = traces[0] == traces[1] and oracle_answers[0] != oracle_answers[1]
    record("rule_answer_identifiable_from_available_evidence", not ambiguous,
           {"identical_public_traces": traces[0] == traces[1],
            "different_correct_answers": oracle_answers[0] != oracle_answers[1],
            "oracle_answers_for_review_only": oracle_answers})

    # Isolate the production episode in a process so a no-progress loop cannot
    # hang this review. This child performs no model forward or optimizer work.
    child = """
from bramastra_lab.research.cognition import episode as k
from bramastra_lab.research.environments.k8_live import build_live_env, generate_live_mechanism
m=generate_live_mechanism('inventory',0,seed=5)
p=k.BoundedPlannerAdapter(world_model=k.CannedWorldModel([{'feedback':{},'success_prob':0.5,'value':0}]),max_nodes=8)
t=k.run_episode(build_live_env(m,budget=6,seed=5),p,model=k.ScriptedModel([{'kind':'noop'}]),seed=5,mechanism=m)
print(t['summary'])
"""
    try:
        proc = subprocess.run([sys.executable, "-c", child], capture_output=True,
                              text=True, timeout=12, env={**os.environ, "PYTHONPATH": "."})
        record("episode_terminates_on_budget_exhaustion", proc.returncode == 0,
               {"returncode": proc.returncode, "stdout": proc.stdout[-1200:], "stderr": proc.stderr[-1200:]})
    except subprocess.TimeoutExpired:
        record("episode_terminates_on_budget_exhaustion", False,
               {"timeout_seconds": 12, "child_terminated": True})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("output already exists; select a fresh receipt path")
    before = identities()
    started = time.monotonic()
    result = {"schema": "bramastra-integrated-source-audit/v1", "source_hashes_before": before,
              "optimizer_updates": 0, "model_weight_origin": "no learned model used",
              "checks": checks()}
    result["elapsed_seconds"] = time.monotonic() - started
    result["source_hashes_after"] = identities()
    result["source_stable_during_run"] = before == result["source_hashes_after"]
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"output": args.output, "stable": result["source_stable_during_run"],
                      "results": [(r["criterion"], r["status"]) for r in result["checks"]]}, indent=2))


if __name__ == "__main__":
    main()
