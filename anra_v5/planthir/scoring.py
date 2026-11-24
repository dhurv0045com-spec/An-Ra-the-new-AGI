"""Level-6 exact typed grammar / byte-sequence scoring, public shape bounds & TypedByteSequenceKernel."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from fractions import Fraction
import json
import operator
import re
from typing import Any, Callable, Mapping, Sequence

from anra_v5.elendril.representation import SpecialToken, UTF8

from .contracts import UniversalAction, UniversalObservation

_ALLOWED_DEFAULT_OPS: tuple[str, ...] = ("ANSWER", "ABSTAIN", "QUERY", "RETRIEVE")
_RATIONAL_UNIT_RE = re.compile(
    r"^\s*([+-]?(?:\d+/\d+|\d*\.\d+|\d+\.\d*|\d+))(?:\s+(.*)|\s*([A-Za-z%°µμ$€£].*))?\s*$"
)
_MAX_BYTES_RE = re.compile(r"\bMAX_BYTES\s*[:=]\s*(\d+)\b", re.IGNORECASE)
_GRID_DIM_RE = re.compile(r"\bGRID\s*[:=]\s*(\d+)\s*[xX×]\s*(\d+)\b")
_ARC_QUERY_RE = re.compile(r"Q:([0-9/]+)->\?")


def _strip_eos_markers(text: str) -> str:
    s = str(text).strip()
    changed = True
    while changed:
        changed = False
        for marker in ("<EOS>", "\x02", "\x00"):
            if s.endswith(marker):
                s = s[: -len(marker)].rstrip()
                changed = True
    return s


def parse_typed_output_envelope(
    raw_output: str,
    *,
    allowed_ops: Sequence[str] = _ALLOWED_DEFAULT_OPS,
    require_explicit_op: bool = True,
) -> dict[str, Any]:
    """Parse strict `OP:PAYLOAD` typed envelopes (`ANSWER:...`, `ABSTAIN[:reason]`, `QUERY`, `RETRIEVE`)."""

    allowed_set = {str(op).strip().upper() for op in allowed_ops if str(op).strip()}
    clean = _strip_eos_markers(raw_output)
    if not clean:
        return {
            "valid": False,
            "op": None,
            "payload": "",
            "has_explicit_op": False,
            "clean_text": "",
            "error_code": "EMPTY_OUTPUT",
        }

    if ":" in clean:
        raw_op, raw_payload = clean.split(":", 1)
        op = raw_op.strip().upper()
        payload = raw_payload.strip()
        if op not in allowed_set:
            return {
                "valid": False,
                "op": op or None,
                "payload": payload,
                "has_explicit_op": True,
                "clean_text": clean,
                "error_code": f"UNKNOWN_OP:{op or 'EMPTY'}",
            }
        if op == "ANSWER" and not payload:
            return {
                "valid": False,
                "op": op,
                "payload": "",
                "has_explicit_op": True,
                "clean_text": clean,
                "error_code": "EMPTY_ANSWER_PAYLOAD",
            }
        return {
            "valid": True,
            "op": op,
            "payload": payload,
            "has_explicit_op": True,
            "clean_text": clean,
            "error_code": None,
        }

    cand_op = clean.upper()
    if cand_op in allowed_set and cand_op in ("ABSTAIN", "QUERY", "RETRIEVE"):
        return {
            "valid": True,
            "op": cand_op,
            "payload": "",
            "has_explicit_op": True,
            "clean_text": clean,
            "error_code": None,
        }

    if require_explicit_op:
        return {
            "valid": False,
            "op": None,
            "payload": clean,
            "has_explicit_op": False,
            "clean_text": clean,
            "error_code": "MISSING_TYPED_OP_ENVELOPE",
        }

    if "ANSWER" not in allowed_set:
        return {
            "valid": False,
            "op": None,
            "payload": clean,
            "has_explicit_op": False,
            "clean_text": clean,
            "error_code": "ANSWER_OP_NOT_ALLOWED",
        }

    return {
        "valid": True,
        "op": "ANSWER",
        "payload": clean,
        "has_explicit_op": False,
        "clean_text": clean,
        "error_code": None,
    }


def infer_public_byte_bound(
    prompt: str,
    *,
    custom_bound_fn: Callable[[str], int] | None = None,
    default_bound: int = 64,
    min_bound: int = 16,
    max_bound: int = 512,
) -> int:
    """Compute a deterministic UTF-8 byte generation ceiling strictly from public prompt structure."""

    lo = int(min_bound)
    hi = max(lo, int(max_bound))
    p = str(prompt)
    if custom_bound_fn is not None:
        val = int(custom_bound_fn(p))
        return max(lo, min(hi, val))

    m_max = _MAX_BYTES_RE.search(p)
    if m_max:
        return max(lo, min(hi, int(m_max.group(1))))

    m_grid = _GRID_DIM_RE.search(p)
    if m_grid:
        rows = max(1, int(m_grid.group(1)))
        cols = max(1, int(m_grid.group(2)))
        return max(lo, min(hi, rows * (cols + 1) + 16))

    m_arc = _ARC_QUERY_RE.search(p)
    if m_arc:
        rows = m_arc.group(1).split("/")
        r_cnt = len(rows)
        c_cnt = max((len(r) for r in rows), default=3)
        dim = max(r_cnt, c_cnt, 3)
        return max(lo, min(hi, dim * (dim + 1) + 20))

    return max(lo, min(hi, int(default_bound)))


def normalize_typed_value(
    text: str,
    *,
    value_kind: str = "exact",
    require_unit_match: bool = True,
) -> dict[str, Any]:
    """Normalize exact string, grid, rational+unit, or token-sequence outputs without lossy mod-16 hashing."""

    vk = str(value_kind).strip().lower()
    raw = str(text).strip()
    if not raw:
        return {
            "valid": False,
            "value_kind": vk,
            "canonical": "",
            "fraction": None,
            "unit": None,
            "shape": None,
            "error_code": "EMPTY_VALUE",
        }

    if vk == "exact":
        return {
            "valid": True,
            "value_kind": "exact",
            "canonical": raw,
            "fraction": None,
            "unit": None,
            "shape": None,
            "error_code": None,
        }

    if vk == "tokens":
        canon = " ".join(raw.split())
        return {
            "valid": bool(canon),
            "value_kind": "tokens",
            "canonical": canon,
            "fraction": None,
            "unit": None,
            "shape": (len(canon.split()),),
            "error_code": None,
        }

    if vk == "grid":
        raw_rows = [part.strip() for part in raw.split("/")]
        if not raw_rows or any(not r for r in raw_rows):
            return {
                "valid": False,
                "value_kind": "grid",
                "canonical": raw,
                "fraction": None,
                "unit": None,
                "shape": None,
                "error_code": "EMPTY_GRID_ROW",
            }
        norm_rows: list[str] = []
        col_counts: set[int] = set()
        for r in raw_rows:
            if any(ch.isspace() for ch in r):
                cells = r.split()
                norm_rows.append(" ".join(cells))
                col_counts.add(len(cells))
            else:
                norm_rows.append(r)
                col_counts.add(len(r))
        if len(col_counts) != 1:
            return {
                "valid": False,
                "value_kind": "grid",
                "canonical": "/".join(norm_rows),
                "fraction": None,
                "unit": None,
                "shape": None,
                "error_code": "JAGGED_GRID",
            }
        cols = next(iter(col_counts))
        return {
            "valid": True,
            "value_kind": "grid",
            "canonical": "/".join(norm_rows),
            "fraction": None,
            "unit": None,
            "shape": (len(norm_rows), cols),
            "error_code": None,
        }

    if vk == "rational_unit":
        m = _RATIONAL_UNIT_RE.fullmatch(raw)
        if not m:
            return {
                "valid": False,
                "value_kind": "rational_unit",
                "canonical": raw,
                "fraction": None,
                "unit": None,
                "shape": None,
                "error_code": "INVALID_RATIONAL_UNIT",
            }
        num_str = m.group(1)
        unit_str = " ".join((m.group(2) or m.group(3) or "").strip().lower().split())
        try:
            frac = Fraction(num_str)
        except (ValueError, ZeroDivisionError):
            return {
                "valid": False,
                "value_kind": "rational_unit",
                "canonical": raw,
                "fraction": None,
                "unit": None,
                "shape": None,
                "error_code": "INVALID_RATIONAL_UNIT",
            }
        if require_unit_match is False and not unit_str:
            unit_str = ""
        val_str = f"{frac.numerator}" if frac.denominator == 1 else f"{frac.numerator}/{frac.denominator}"
        canon = f"{val_str} {unit_str}".strip()
        return {
            "valid": True,
            "value_kind": "rational_unit",
            "canonical": canon,
            "fraction": frac,
            "unit": unit_str,
            "shape": None,
            "error_code": None,
        }

    raise ValueError(f"unsupported value_kind: {value_kind!r}")


def score_typed_output(
    prompt: str,
    raw_output: str,
    *,
    expected_op: str = "ANSWER",
    expected_payload: str | None = None,
    value_kind: str = "exact",
    max_new_bytes: int | None = None,
    bound_fn: Callable[[str], int] | None = None,
    terminated_by_eos: bool = True,
    allowed_ops: Sequence[str] = _ALLOWED_DEFAULT_OPS,
    syntax_validator: Callable[[str, str, str], tuple[bool, str]] | None = None,
    require_explicit_op: bool = True,
    require_unit_match: bool = True,
) -> dict[str, Any]:
    """Score exact UTF-8 typed outputs against public shape bounds and typed normalization without mod-16 collapse."""

    bound = (
        int(max_new_bytes)
        if max_new_bytes is not None
        else infer_public_byte_bound(prompt, custom_bound_fn=bound_fn)
    )
    raw_str = str(raw_output)
    byte_len = len(raw_str.encode("utf-8"))
    overflow = byte_len > bound
    unterminated = not bool(terminated_by_eos)
    truncated = bool(overflow or unterminated)

    env = parse_typed_output_envelope(
        raw_str,
        allowed_ops=allowed_ops,
        require_explicit_op=require_explicit_op,
    )

    syntax_ok = True
    syntax_reason = ""
    if env["valid"] and syntax_validator is not None:
        syntax_ok, syntax_reason = syntax_validator(str(prompt), str(env["op"]), str(env["payload"]))

    exp_op = str(expected_op).strip().upper()
    op_match = bool(env["valid"] and syntax_ok and env["op"] == exp_op)

    pred_norm: dict[str, Any] | None = None
    exp_norm: dict[str, Any] | None = None
    payload_match = False
    value_err: str | None = None

    if env["valid"] and syntax_ok and op_match:
        if exp_op in ("ABSTAIN", "QUERY", "RETRIEVE"):
            if expected_payload is None or str(expected_payload).strip() == "":
                payload_match = True
            else:
                payload_match = env["payload"].strip() == str(expected_payload).strip()
                if not payload_match:
                    value_err = "NON_ANSWER_PAYLOAD_MISMATCH"
        else:
            if expected_payload is None:
                payload_match = False
                value_err = "MISSING_EXPECTED_PAYLOAD"
            else:
                pred_norm = normalize_typed_value(
                    env["payload"],
                    value_kind=value_kind,
                    require_unit_match=require_unit_match,
                )
                exp_norm = normalize_typed_value(
                    str(expected_payload),
                    value_kind=value_kind,
                    require_unit_match=require_unit_match,
                )
                if not pred_norm["valid"]:
                    value_err = str(pred_norm["error_code"] or "INVALID_PREDICTED_VALUE")
                elif not exp_norm["valid"]:
                    value_err = str(exp_norm["error_code"] or "INVALID_EXPECTED_VALUE")
                elif value_kind.strip().lower() == "rational_unit":
                    frac_ok = pred_norm["fraction"] == exp_norm["fraction"]
                    unit_ok = (not require_unit_match) or (pred_norm["unit"] == exp_norm["unit"])
                    payload_match = bool(frac_ok and unit_ok)
                    if frac_ok and not unit_ok:
                        value_err = "UNIT_MISMATCH"
                    elif not frac_ok:
                        value_err = "VALUE_MISMATCH"
                else:
                    payload_match = pred_norm["canonical"] == exp_norm["canonical"]
                    if not payload_match:
                        if (
                            value_kind.strip().lower() == "grid"
                            and pred_norm.get("shape") != exp_norm.get("shape")
                        ):
                            value_err = "GRID_SHAPE_MISMATCH"
                        else:
                            value_err = "VALUE_MISMATCH"

    exact_match = bool((not overflow) and (not unterminated) and env["valid"] and syntax_ok and op_match and payload_match)

    failure_reason: str | None = None
    if not exact_match:
        if overflow:
            failure_reason = "BYTE_BOUND_OVERFLOW"
        elif unterminated:
            failure_reason = "UNTERMINATED_NO_EOS"
        elif not env["valid"]:
            failure_reason = str(env["error_code"] or "INVALID_ENVELOPE")
        elif not syntax_ok:
            failure_reason = str(syntax_reason or "SYNTAX_VALIDATION_FAILED")
        elif not op_match:
            failure_reason = "OP_MISMATCH"
        else:
            failure_reason = str(value_err or "VALUE_MISMATCH")

    return {
        "schema": "anra-planthir-typed-score-v1",
        "exact_match": exact_match,
        "valid_syntax": bool(env["valid"] and syntax_ok and not overflow and not unterminated),
        "op_match": op_match,
        "payload_match": payload_match,
        "predicted_op": env["op"],
        "predicted_payload": env["payload"],
        "predicted_canonical": pred_norm["canonical"] if pred_norm is not None else env["payload"],
        "expected_op": exp_op,
        "expected_payload": expected_payload,
        "expected_canonical": exp_norm["canonical"] if exp_norm is not None else expected_payload,
        "value_kind": str(value_kind).strip().lower(),
        "has_explicit_op": env["has_explicit_op"],
        "byte_length": byte_len,
        "public_byte_bound": bound,
        "overflow": overflow,
        "unterminated": unterminated,
        "truncated": truncated,
        "failure_reason": failure_reason,
    }


@dataclass(frozen=True, slots=True)
class _TypedByteState:
    world_id: str
    public_prompt: str
    rendered_text: str
    available_tools: dict[str, bool]
    consumed_tools: frozenset[str]
    step: int = 0
    terminal: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class TypedByteSequenceKernel:
    """Truth-firewalled EnvironmentKernel for exact typed multi-byte generation without mod-16 collapse."""

    kernel_id = "planthir-typed-byte-sequence-kernel-v1"

    def __init__(
        self,
        *,
        bound_fn: Callable[[str], int] | None = None,
        default_value_kind: str = "exact",
        syntax_validator: Callable[[str, str, str], tuple[bool, str]] | None = None,
        require_explicit_op: bool = True,
        require_unit_match: bool = True,
        structural_hash_fn: Callable[[Mapping[str, Any]], str | None] | None = None,
    ) -> None:
        self.bound_fn = bound_fn
        self.default_value_kind = str(default_value_kind)
        self.syntax_validator = syntax_validator
        self.require_explicit_op = bool(require_explicit_op)
        self.require_unit_match = bool(require_unit_match)
        self.structural_hash_fn = structural_hash_fn
        self._expected_by_wid: dict[str, tuple[str, str | None]] = {}

    @staticmethod
    def _world_id(world: Mapping[str, Any]) -> str:
        return str(world.get("world_id") or world.get("case_id") or world.get("task_id") or world.get("id") or "w0")

    @staticmethod
    def _public_prompt(world: Mapping[str, Any]) -> str:
        for key in ("public_prompt", "rendered_text", "prompt"):
            if world.get(key) is not None:
                return str(world[key])
        wid = TypedByteSequenceKernel._world_id(world)
        sig = world.get("sig", wid)
        feat = world.get("feat")
        return f"PROMPT sig={sig}" + (f" feat={json.dumps(list(feat))}" if feat is not None else "")

    def _extract_expected(self, world: Mapping[str, Any]) -> tuple[str, str | None]:
        hidden = world.get("hidden") if isinstance(world.get("hidden"), Mapping) else {}
        raw_op = hidden.get("expected_operation", world.get("expected_operation"))
        raw_arg = None
        for key in ("expected", "expected_answer", "gold_text", "target", "label"):
            if key in hidden and hidden[key] is not None:
                raw_arg = hidden[key]
                break
            if key in world and world[key] is not None:
                raw_arg = world[key]
                break
        if raw_op is not None:
            op_s = str(raw_op).strip().upper()
            return (op_s, None if raw_arg is None else str(raw_arg))
        if raw_arg is None:
            return ("ABSTAIN", None)
        arg_s = str(raw_arg).strip()
        if arg_s.upper().startswith(("ABSTAIN", "QUERY", "RETRIEVE", "ANSWER:")):
            env = parse_typed_output_envelope(arg_s, require_explicit_op=True)
            if env["valid"] and env["op"] is not None:
                return (str(env["op"]), env["payload"] or None)
        return ("ANSWER", arg_s)

    def structural_signature(self, world: Mapping[str, Any]) -> str:
        from .design import canonical_structural_hash

        sh = canonical_structural_hash(world, custom_hash_fn=self.structural_hash_fn)
        if sh is not None:
            return sh
        return str(
            world.get("canonical_family_key")
            or world.get("sig")
            or world.get("family_id")
            or self._world_id(world)
        )

    def initial_state(self, world: Mapping[str, Any]) -> _TypedByteState:
        wid = self._world_id(world)
        self._expected_by_wid[wid] = self._extract_expected(world)
        prompt = self._public_prompt(world)
        if isinstance(world.get("available_tools"), Mapping):
            avail = {str(k): bool(v) for k, v in world["available_tools"].items()}
        else:
            avail = {
                "query": bool(world.get("query_available", False)),
                "memory": bool(world.get("memory_available", False)),
            }
        pub_bound = infer_public_byte_bound(prompt, custom_bound_fn=self.bound_fn)
        vk = str(world.get("value_kind") or self.default_value_kind)
        meta: dict[str, Any] = {
            "world_id": wid,
            "public_byte_bound": pub_bound,
            "value_kind": vk,
        }
        if world.get("feat") is not None:
            meta["feat"] = list(world["feat"])
        if world.get("prompt_tokens") is not None:
            meta["prompt_tokens"] = [int(t) for t in world["prompt_tokens"]]
        return _TypedByteState(
            world_id=wid,
            public_prompt=prompt,
            rendered_text=prompt,
            available_tools=avail,
            consumed_tools=frozenset(),
            step=0,
            terminal=False,
            metadata=meta,
        )

    def observe(self, state: _TypedByteState) -> UniversalObservation:
        obs = UniversalObservation(
            obs_id=f"{state.world_id}:s{state.step}",
            rendered_text=state.rendered_text,
            available_tools=dict(state.available_tools),
            consumed_tools=state.consumed_tools,
            decision_index=int(state.step),
            budget_remaining=max(1, 1 - int(state.step)),
            metadata=dict(state.metadata),
        )
        violations = obs.validate_invariants()
        if violations:
            raise ValueError(f"UniversalObservation invariant violation: {violations}")
        return obs

    def ideal_action(self, state: _TypedByteState | Mapping[str, Any]) -> tuple[str, str | None]:
        if isinstance(state, _TypedByteState):
            if state.world_id in self._expected_by_wid:
                return self._expected_by_wid[state.world_id]
            return ("ABSTAIN", None)
        return self._extract_expected(state)

    def step(self, state: _TypedByteState, operation: str, argument: str | None) -> _TypedByteState:
        consumed = set(state.consumed_tools)
        avail = dict(state.available_tools)
        if operation == "QUERY":
            consumed.add("query")
            avail["query"] = False
        elif operation == "RETRIEVE":
            consumed.add("retrieve")
            avail["retrieve"] = False
        return _TypedByteState(
            world_id=state.world_id,
            public_prompt=state.public_prompt,
            rendered_text=state.rendered_text,
            available_tools=avail,
            consumed_tools=frozenset(consumed),
            step=state.step + 1,
            terminal=True,
            metadata=dict(state.metadata),
        )

    def score_actions(
        self,
        world: Mapping[str, Any],
        actions: Sequence[UniversalAction | tuple[str, str | None]],
    ) -> dict[str, Any]:
        prompt = self._public_prompt(world)
        exp_op, exp_payload = self._extract_expected(world)
        vk = str(world.get("value_kind") or self.default_value_kind)
        if not actions:
            return score_typed_output(
                prompt,
                "",
                expected_op=exp_op,
                expected_payload=exp_payload,
                value_kind=vk,
                bound_fn=self.bound_fn,
                terminated_by_eos=False,
                syntax_validator=self.syntax_validator,
                require_explicit_op=self.require_explicit_op,
                require_unit_match=self.require_unit_match,
            )

        act = actions[0]
        if isinstance(act, UniversalAction):
            meta = act.metadata or {}
            raw_out = str(
                meta["raw_output"]
                if "raw_output" in meta
                else (f"{act.operation}:{act.argument}" if act.argument is not None else act.operation)
            )
            term_eos = bool(meta.get("terminated_by_eos", True))
            max_b = int(meta["max_new_bytes"]) if meta.get("max_new_bytes") is not None else None
        else:
            op_s, arg_s = act
            raw_out = f"{op_s}:{arg_s}" if arg_s is not None else str(op_s)
            term_eos = True
            max_b = None

        return score_typed_output(
            prompt,
            raw_out,
            expected_op=exp_op,
            expected_payload=exp_payload,
            value_kind=vk,
            max_new_bytes=max_b,
            bound_fn=self.bound_fn,
            terminated_by_eos=term_eos,
            syntax_validator=self.syntax_validator,
            require_explicit_op=self.require_explicit_op,
            require_unit_match=self.require_unit_match,
        )

    def verify(
        self,
        world: Mapping[str, Any],
        actions: Sequence[UniversalAction | tuple[str, str | None]],
    ) -> dict[str, Any]:
        score = self.score_actions(world, actions)
        exp_pair = (score["expected_op"], score["expected_payload"])
        obs_pair = (score["predicted_op"], score["predicted_payload"] or None)
        return {
            "correct": bool(score["exact_match"]),
            "terminal_status": "abstained" if score["expected_op"] == "ABSTAIN" and score["exact_match"] else "answered",
            "expected": exp_pair,
            "observed": obs_pair,
            "typed_score": score,
        }

    def decompose_subgoals(self, world: Mapping[str, Any]) -> tuple[Any, ...]:
        return ()


def decode_generated_bytes(
    token_ids: Iterable[Any],
    *,
    stop_chars: Sequence[str] = ("\n",),
    strip_whitespace: bool = True,
    stop_on_special_answer: bool = True,
) -> dict[str, Any]:
    """Decode unconstrained autoregressive UTF-8 byte token emissions without crashing on non-UTF-8 bytes."""

    ans_id = UTF8.special(SpecialToken.ANSWER)
    collected: list[Any] = []
    valid_byte_ids: list[int] = []
    terminated_by_eos = False

    for raw_tok in token_ids:
        if isinstance(raw_tok, bool):
            collected.append(raw_tok)
            continue
        try:
            tok = operator.index(raw_tok)
        except TypeError:
            collected.append(raw_tok)
            continue
        if tok == UTF8.eos_id or (stop_on_special_answer and tok == ans_id):
            terminated_by_eos = True
            break
        if tok in (UTF8.pad_id, UTF8.bos_id):
            continue
        if UTF8.is_ascii_stop_token(tok, stop_chars):
            terminated_by_eos = True
            break
        collected.append(tok)
        if 4 <= tok < UTF8.vocabulary_size:
            valid_byte_ids.append(tok)

    raw_text, valid_utf8 = UTF8.decode_lossy(collected, strip_special=True)
    return {
        "text": raw_text.strip() if strip_whitespace else raw_text,
        "raw_text": raw_text,
        "valid_utf8": bool(valid_utf8),
        "terminated_by_eos": bool(terminated_by_eos),
        "generated_bytes": len(valid_byte_ids),
    }


__all__ = [
    "TypedByteSequenceKernel",
    "decode_generated_bytes",
    "infer_public_byte_bound",
    "normalize_typed_value",
    "parse_typed_output_envelope",
    "score_typed_output",
]
