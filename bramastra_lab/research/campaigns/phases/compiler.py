"""Real supervision compiler for K8 (contracts S3).

Compiles immutable public decision states with separately stored target
channels from the prepared bundle. No fabricated fixed transitions,
uniformity over arbitrary token lists, or unconditional zero values.
Enforces explicit split membership, prescribed family mixture, canonical
mechanism IDs in sidecars. Empty data fails; never synthesizes rows.
"""
from __future__ import annotations

import glob
import json
import os
import random
from typing import Any, Mapping


K8_FAMILIES = ("rule-inquiry", "inventory", "program")
K8_TRAINING_POOL = "training"
K8_MAX_SEQ = 512


def goal_prefix_tokens(public: Mapping[str, Any], *,
                       max_event_bytes: int = None) -> list[int]:
    """Complete initial public-state prefix used by training and inference.

    This emits the same renderer marker, goal and initial budget events used
    by the live policy. All decision channels share this prefix.
    """
    from bramastra_lab.research.experience.codec import (
        DEFAULT_MAX_EVENT_BYTES, SPECIAL_BOUNDARY, encode_event)
    from bramastra_lab.research.experience.public_state import (
        public_state_prefix_events)
    from bramastra_lab.research.cognition.episode import (
        ACTION_BUDGET_DEFAULT, CALL_BUDGET_DEFAULT, NODE_BUDGET_DEFAULT)

    if max_event_bytes is None:
        max_event_bytes = DEFAULT_MAX_EVENT_BYTES
    initial_budgets = {"actions_left": ACTION_BUDGET_DEFAULT,
                       "calls_left": CALL_BUDGET_DEFAULT,
                       "nodes_left": NODE_BUDGET_DEFAULT}
    tokens = [SPECIAL_BOUNDARY]
    for role, content in public_state_prefix_events(
            public, [], budgets=initial_budgets):
        tokens.extend(encode_event(role, content,
                                   max_event_bytes=max_event_bytes))
    return tokens


def _training_budgets(history: list[dict[str, Any]]) -> dict[str, int]:
    """Reconstruct the declared policy budgets for a direct-policy target.

    Prepared teacher trajectories use one decision call per received action;
    unlike model-based rollouts, they contain no hidden planner calls. The
    convention is explicit and is part of the public-state renderer identity.
    """
    from bramastra_lab.research.cognition.episode import (
        ACTION_BUDGET_DEFAULT, CALL_BUDGET_DEFAULT, NODE_BUDGET_DEFAULT,
        _is_submission_like)

    inquiries = sum(1 for item in history
                    if not _is_submission_like(item.get("action", {})))
    return {"actions_left": max(0, ACTION_BUDGET_DEFAULT - inquiries),
            "calls_left": max(0, CALL_BUDGET_DEFAULT - len(history)),
            "nodes_left": NODE_BUDGET_DEFAULT}


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_training_trajectories(data_dir: str, *, seed: int,
                               families: tuple[str, ...] = K8_FAMILIES,
                               pool: str = K8_TRAINING_POOL) -> list[dict[str, Any]]:
    """Load exact-split trajectories with canonical IDs (no substring match).

    Only rows whose `pool` field equals `pool` exactly are admitted.
    Training-controller, development and sealed-confirmation rows are never
    admitted via substring matching. Empty data fails.
    """
    episode_dir = os.path.join(data_dir, "episodes")
    if not os.path.isdir(episode_dir):
        raise ValueError(f"episode directory missing: {episode_dir}")
    per_family: dict[str, list[dict[str, Any]]] = {fam: [] for fam in families}
    for family in families:
        # Exact filename: {family}-{pool}.jsonl (never substring).
        exact = os.path.join(episode_dir, f"{family}-{pool}.jsonl")
        if not os.path.exists(exact):
            raise ValueError(
                f"required split file missing: {exact}; refusing to "
                "substitute another pool via substring matching")
        for row in _read_jsonl(exact):
            # Explicit split membership check on the row itself.
            if row.get("pool") != pool:
                raise ValueError(
                    f"row {row.get('mechanism_id')} pool {row.get('pool')!r} "
                    f"does not match required split {pool!r}; refusing")
            if row.get("family") != family:
                raise ValueError(
                    f"row family {row.get('family')!r} mismatches file family "
                    f"{family!r}")
            if not row.get("mechanism_id") or not row.get("canonical_identity"):
                raise ValueError("trajectory missing canonical mechanism identity")
            if not row.get("public") or "answer" not in row:
                raise ValueError("trajectory missing public/answer targets")
            per_family[family].append(row)
    # Prescribed family mixture: stratified round-robin (equal representation).
    # Fail on any empty family (never synthesize).
    for family, rows in per_family.items():
        if not rows:
            raise ValueError(
                f"family {family!r} has zero {pool} trajectories; empty data "
                "must fail, never synthesize replacement rows")
    # Deterministic shuffle per family, then round-robin interleave.
    for family in families:
        rng = random.Random(f"K8:{pool}:{family}:{seed}")
        rng.shuffle(per_family[family])
    merged: list[dict[str, Any]] = []
    max_len = max(len(v) for v in per_family.values())
    for index in range(max_len):
        for family in families:
            rows = per_family[family]
            if index < len(rows):
                merged.append(rows[index])
    if not merged:
        raise ValueError("no training trajectories; failing before GPU work")
    return merged


def load_tool_rows(data_dir: str, *, split: str) -> list[dict[str, Any]]:
    """Load tool tasks for an exact split (tool-training vs tool-heldout)."""
    tool_path = os.path.join(data_dir, "tools", "tool_tasks.jsonl")
    if not os.path.exists(tool_path):
        raise ValueError(f"tool tasks missing: {tool_path}")
    rows = [r for r in _read_jsonl(tool_path) if r.get("split") == split]
    if not rows:
        raise ValueError(
            f"no tool rows for split {split!r}; failing (heldout must never "
            "enter training streams)")
    return rows


def prompt_tokens_for_row(row: dict[str, Any], *,
                          max_seq: int = K8_MAX_SEQ) -> list[int]:
    """Complete decision prompt for one trajectory row (single renderer).

    Boundary + goal event + compact received-history events (pre-submission)
    — byte-identical to the context portion of the training row built by
    `build_batch_for_trajectory`. Evaluation MUST use this representation;
    no JSON slicing, no token cuts, explicit overflow failure.
    """
    from bramastra_lab.research.experience.public_state import (
        public_state_prefix_events)
    from bramastra_lab.research.experience.codec import (
        DEFAULT_MAX_EVENT_BYTES, SPECIAL_BOUNDARY, encode_event)

    history_events = []
    for step in row.get("history") or []:
        action = step.get("action")
        if not isinstance(action, dict):
            raise ValueError(
                f"trajectory {row.get('mechanism_id')} history action must "
                "be an object")
        if action.get("kind") == "submit":
            break
        feedback = step.get("feedback")
        if not isinstance(feedback, dict):
            raise ValueError(
                f"trajectory {row.get('mechanism_id')} history feedback "
                "must be an object")
        history_events.append({"action": action, "feedback": feedback})
    tokens = [SPECIAL_BOUNDARY]
    for role, content in public_state_prefix_events(
            dict(row["public"]), history_events,
            budgets=_training_budgets(history_events)):
        tokens += encode_event(role, content,
                               max_event_bytes=DEFAULT_MAX_EVENT_BYTES)
    if len(tokens) + 1 > max_seq:
        raise ValueError(
            f"decision prompt needs {len(tokens)} tokens; max_seq is "
            f"{max_seq}")
    return tokens


def build_batch_for_trajectory(row: dict[str, Any], *,
                               max_seq: int = K8_MAX_SEQ):
    """Real answer/EOS batch with canonical sidecars (no synthetic fallback).

    The prompt compiles the goal PLUS the actual received history so the
    supervised answer is decidable from permitted evidence — never from the
    goal alone, never from future feedback. History entries use the SAME
    compact observation projection the live episode renderer uses
    (`render_public_state` -> `_compact_history_entry`), so training and
    inference share one representation. The final submission and its verdict
    never condition the answer target.
    """
    from bramastra_lab.research.experience.public_state import (
        public_state_identity, public_state_prefix_events)
    from bramastra_lab.research.experience.sequences import (
        build_answer_row, collocate)

    public = dict(row["public"])
    answer = str(row["answer"])
    history = row.get("history") or []
    history_events = []
    for step in history:
        action = step.get("action")
        feedback = step.get("feedback")
        if not isinstance(action, dict):
            raise ValueError(
                f"trajectory {row.get('mechanism_id')} history action must "
                "be an object")
        if action.get("kind") == "submit":
            break
        if not isinstance(feedback, dict):
            raise ValueError(
                f"trajectory {row.get('mechanism_id')} history feedback "
                "must be an object")
        history_events.append({"action": action, "feedback": feedback})
    prompt_events = public_state_prefix_events(
        public, history_events, budgets=_training_budgets(history_events))
    seq = build_answer_row(
        prompt_events, answer,
        provenance={"kind": "trajectory",
                    "episode_id": str(row["mechanism_id"]),
                    "task_semantic_id": str(row.get("family", "k8")),
                    "split": str(row.get("pool", "training")),
                    "source": "k8-bundle",
                    "collection_policy": str(row.get("exploration_mode", "teacher")),
                    "family": str(row.get("family", "k8")),
                    "mechanism_cluster": str(row.get("canonical_identity", ""))[:64],
                    "public_state_identity": public_state_identity()},
        max_tokens=max_seq)
    batch = collocate([seq], max_seq=max_seq)
    return batch


def compile_channels_for_row(row: dict[str, Any], batch,
                             *, arm_weights: dict[str, float],
                             arm_enabled: frozenset[str],
                             max_seq: int = K8_MAX_SEQ) -> dict[str, Any]:
    """Compile real world/action/value/pair channels from episode history.

    - world: actual next observation after the first allowed history action.
    - action: declared teacher policy (history's first action) over legal
      candidates (row queries); never invented uniformity over arbitrary lists.
    - value: specified return convention (1.0 correct / 0.0 otherwise from
      the history verdict); never unconditional zero.
    - pair: actual paired goals are bound by the caller via pair_rows
      (see `build_pair_rows`); here we only declare eligibility.

    Every channel conditions on the COMPLETE prompt prefix (boundary + goal
    event, cross-checked against the compiled batch) and full-length
    candidate/action/target encodings. Oversized content fails explicitly;
    nothing is silently truncated.
    """
    from bramastra_lab.research.experience.codec import (
        encode_event, encode_text)
    import torch

    history = row.get("history") or []
    if not history:
        raise ValueError(
            f"trajectory {row.get('mechanism_id')} has no history; cannot "
            "compile world/action/value targets from fixtures")
    first = history[0]
    action = first.get("action")
    feedback = first.get("feedback")
    if not isinstance(action, dict) or not isinstance(feedback, dict):
        raise ValueError("history action/feedback must be objects")
    # Complete prompt prefix from the same single goal encoding the answer
    # row uses — no fixed prompt, no six-token cut, no answer leakage.
    prefix_tokens = goal_prefix_tokens(row.get("public") or {})
    try:
        batch_prefix = batch.input_ids[0][:len(prefix_tokens)].tolist()
    except Exception as exc:
        raise ValueError(
            f"batch has no input tokens for channel compilation: {exc}") from exc
    if list(prefix_tokens) != batch_prefix:
        raise ValueError(
            "compiled channel prefix disagrees with the batch input prefix; "
            "refusing drifting training/inference representations")
    if len(prefix_tokens) + 1 > max_seq:
        raise ValueError(
            f"prompt prefix needs {len(prefix_tokens)} tokens; max_seq is "
            f"{max_seq}")
    compiled: dict[str, Any] = {"weights": dict(arm_weights),
                                "enabled": frozenset(arm_enabled),
                                "extra": {}}
    # World channel: real action + real target feedback + real goal (the
    # goal is already inside the complete prefix).
    if "world" in arm_enabled and float(arm_weights.get("world", 0.0)) > 0:
        action_event = encode_event("action", action)
        feedback_event = encode_event("feedback", dict(feedback))
        span = len(prefix_tokens) + len(action_event) + len(feedback_event) + 1
        if span > max_seq:
            raise ValueError(
                f"world transition needs {span} tokens; max_seq is {max_seq}")
        compiled["world"] = {"prefix_tokens": list(prefix_tokens),
                             "action": dict(action),
                             "target_feedback": dict(feedback),
                             "denominator": 1}
    # Action channel: legal candidates from row queries + teacher one-hot
    # from the history's actual first action (declared teacher policy).
    # Matching is exact (query object contained in history action). When the
    # bundle declares degenerate identical queries (e.g. program evaluate),
    # the single real history action is the only legal candidate (teacher
    # [1.0]); no invented token lists or uniformity over arbitrary IDs.
    if "action" in arm_enabled and float(arm_weights.get("action", 0.0)) > 0:
        family = str(row.get("family", ""))
        # Program family: bundle queries are placeholder inputs unrelated to
        # the operations teacher. The single real history action (with actual
        # operations) is the only declared legal action; no invented
        # alternatives, no uniformity over arbitrary token lists.
        if family == "program":
            action_text_single = json.dumps(action, sort_keys=True)
            tokens_single = encode_text(action_text_single)
            if not tokens_single:
                raise ValueError("empty history-action encoding; refusing")
            if len(prefix_tokens) + len(tokens_single) > max_seq:
                raise ValueError(
                    f"action candidate needs {len(prefix_tokens) + len(tokens_single)} "
                    f"tokens; max_seq is {max_seq}")
            candidates = [list(tokens_single)]
            teacher = [1.0]
        else:
            queries = row.get("queries") or []
            if not queries:
                raise ValueError(
                    "action enabled but trajectory carries no legal queries")
            # All declared queries are candidates, at FULL length (no silent
            # truncation: shuffled exploration orders must stay covered).
            candidates = []
            for query in queries:
                text = json.dumps(query, sort_keys=True)
                tokens = encode_text(text)
                if not tokens:
                    raise ValueError("empty candidate encoding; refusing")
                if len(prefix_tokens) + len(tokens) > max_seq:
                    raise ValueError(
                        f"action candidate needs {len(prefix_tokens) + len(tokens)} "
                        f"tokens; max_seq is {max_seq}")
                candidates.append(list(tokens))
            if not candidates:
                raise ValueError("no action candidates; refusing")
            query_texts = [json.dumps(q, sort_keys=True) for q in queries[:len(candidates)]]
            if len(set(query_texts)) == 1:
                # Degenerate identical queries: single real history action.
                action_text_single = json.dumps(action, sort_keys=True)
                tokens_single = encode_text(action_text_single)
                if not tokens_single:
                    raise ValueError("empty history-action encoding; refusing")
                if len(prefix_tokens) + len(tokens_single) > max_seq:
                    raise ValueError(
                        f"action candidate needs {len(prefix_tokens) + len(tokens_single)} "
                        f"tokens; max_seq is {max_seq}")
                candidates = [list(tokens_single)]
                teacher = [1.0]
            else:
                # Exact dict-subset match only (query object contained in the
                # history action). No substring/JSON-containment fallback.
                teacher_index = None
                for idx, query in enumerate(queries[:len(candidates)]):
                    if all(action.get(k) == v for k, v in query.items()):
                        teacher_index = idx
                        break
                if teacher_index is None:
                    raise ValueError(
                        f"trajectory {row.get('mechanism_id')} history action does not "
                        "contain any legal query exactly; refusing invented teacher")
                teacher = [0.0] * len(candidates)
                teacher[teacher_index] = 1.0
        compiled["action"] = {"prefix_tokens": list(prefix_tokens),
                              "candidates": candidates,
                              "teacher_distribution": teacher,
                              "legal_mask": [True] * len(candidates),
                              "denominator": 1}
    # Value channel: real return from the verdict (not unconditional zero).
    if "value" in arm_enabled and float(arm_weights.get("value", 0.0)) > 0:
        verdict = None
        for step in history:
            feedback_step = step.get("feedback") or {}
            if feedback_step.get("kind") == "verdict":
                verdict = feedback_step
                break
        if verdict is None:
            raise ValueError("value enabled but history has no verdict")
        target_return = 1.0 if verdict.get("correct") else 0.0
        compiled["value"] = {"prefix_tokens": list(prefix_tokens),
                             "target_return": float(target_return),
                             "denominator": 1}
    # Pair eligibility: caller must supply pair_rows from real distinct answers.
    if "pair" in arm_enabled and float(arm_weights.get("pair", 0.0)) > 0:
        compiled["pair"] = {"denominator": 1,
                            "requires_pair_rows": True}
    return compiled


def build_pair_rows(row_a: dict[str, Any], row_b: dict[str, Any], *,
                    max_seq: int = K8_MAX_SEQ):
    """Real pair rows binding actual paired goals and valid answers.

    Uses two trajectories with different answers (verified, not invented).
    Own rows carry each goal with its own valid answer; swapped rows carry
    each goal with the other's answer (valid contrast). Identical answers
    are refused (no negative).
    """
    from bramastra_lab.research.experience.sequences import (
        build_answer_row)

    answer_a = str(row_a["answer"])
    answer_b = str(row_b["answer"])
    if answer_a == answer_b:
        raise ValueError("pair requires distinct answers; refusing identical")
    own_a = build_answer_row(
        [("goal", dict(row_a["public"]))], answer_a,
        provenance={"kind": "trajectory",
                    "episode_id": str(row_a["mechanism_id"]),
                    "task_semantic_id": str(row_a.get("family", "k8")),
                    "split": str(row_a.get("pool", "training")),
                    "source": "k8-bundle",
                    "collection_policy": "paired",
                    "family": str(row_a.get("family", "k8"))},
        max_tokens=max_seq)
    own_b = build_answer_row(
        [("goal", dict(row_b["public"]))], answer_b,
        provenance={"kind": "trajectory",
                    "episode_id": str(row_b["mechanism_id"]),
                    "task_semantic_id": str(row_b.get("family", "k8")),
                    "split": str(row_b.get("pool", "training")),
                    "source": "k8-bundle",
                    "collection_policy": "paired",
                    "family": str(row_b.get("family", "k8"))},
        max_tokens=max_seq)
    swapped_a = build_answer_row(
        [("goal", dict(row_a["public"]))], answer_b,
        provenance={"kind": "trajectory",
                    "episode_id": str(row_a["mechanism_id"]),
                    "task_semantic_id": str(row_a.get("family", "k8")),
                    "split": str(row_a.get("pool", "training")),
                    "source": "k8-bundle",
                    "collection_policy": "paired-swapped",
                    "family": str(row_a.get("family", "k8"))},
        max_tokens=max_seq)
    swapped_b = build_answer_row(
        [("goal", dict(row_b["public"]))], answer_a,
        provenance={"kind": "trajectory",
                    "episode_id": str(row_b["mechanism_id"]),
                    "task_semantic_id": str(row_b.get("family", "k8")),
                    "split": str(row_b.get("pool", "training")),
                    "source": "k8-bundle",
                    "collection_policy": "paired-swapped",
                    "family": str(row_b.get("family", "k8"))},
        max_tokens=max_seq)
    return [own_a, own_b], [swapped_a, swapped_b]
