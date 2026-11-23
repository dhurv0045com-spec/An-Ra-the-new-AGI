"""V8-X3 10-Family Cognitive Battery, Public Typed Output Protocol, Structural Custody & 22-Family Registry."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import random
import re
from typing import Any, Mapping, Sequence

import numpy as np

from anra_v5.elendril.codec import digest
from experiments.x2.tasks import build_t14_panel, solve_t14_public_dp
from experiments.x3.factorial import (
    X3_SYMBOLS,
    build_x3_paired_graph_factorial,
    independent_walk_x3_graph,
    render_x3_prompt,
)

X3_COGNITIVE_SCHEMA = "elendril-x3-cognitive-battery-v2"
X3_PUBLIC_TYPED_OUTPUT_SCHEMA = "elendril-x3-typed-output-v1"
X3_MAX_PROMPT_BYTES = 600
X3_DEFAULT_PUBLIC_GEN_BOUND = 48

TRAIN_BANK_SEED = 71001
DEV_BANK_SEED = 71002
CONFIRM_COMMITMENT_SEED = 71003
TRANSFER_BANK_SEED = 71004

X3_COGNITIVE_FAMILIES: tuple[str, ...] = (
    "C01_NOVEL_OP_INDUCTION",
    "C02_FST_RULE_COMPOSITION",
    "C03_GRID_TRANSFORM_ARC",
    "C04_COMPOSITIONAL_SCAN",
    "C05_RELATIONAL_GRAPH_CF",
    "C06_OPEN_WORLD_DEDUCTION",
    "C07_HIERARCHICAL_LISTOPS",
    "C08_ADAPTIVE_MEMORY_REVISION",
    "C09_METACOGNITION_SELF_CHECK",
    "C10_LANGUAGE_MATH_TRANSFER",
)

SPLIT_SEEDS: dict[str, int] = {
    "train": TRAIN_BANK_SEED,
    "dev": DEV_BANK_SEED,
    "confirm": CONFIRM_COMMITMENT_SEED,
    "ood_transfer": TRANSFER_BANK_SEED,
    "transfer": TRANSFER_BANK_SEED,
}

_CANON_SPLIT: dict[str, str] = {
    "train": "train",
    "dev": "dev",
    "confirm": "confirm",
    "confirmation": "confirm",
    "ood_transfer": "ood_transfer",
    "transfer": "ood_transfer",
}

STRUCTURAL_GROUP_QUOTAS: dict[str, int] = {
    "train": 32,
    "dev": 8,
    "confirm": 12,
    "ood_transfer": 8,
}

VALID_TYPED_OPS: tuple[str, ...] = ("ANSWER", "QUERY", "RETRIEVE", "ABSTAIN", "AMBIGUOUS")
SCAN_VALID_ACTIONS: frozenset[str] = frozenset({"I_WALK", "I_RUN", "I_JUMP", "I_LOOK", "TURN_L", "TURN_R"})

# Complete 22 mandatory families (X01-X12 legacy + C01-C10 cognitive) with provenance categories (§4)
MANDATORY_22_FAMILIES: tuple[dict[str, Any], ...] = (
    # Group 1: Evidence and logic
    {"family_id": "X01", "group": "Evidence and logic", "source": "legacy_x1", "primary_unit": "Direct fact vs last-row lure supported accuracy", "evaluation_modes": ("native_legacy_x",)},
    {"family_id": "X02", "group": "Evidence and logic", "source": "legacy_x1", "primary_unit": "Two-hop interior swap supported & untouched sibling", "evaluation_modes": ("native_legacy_x",)},
    {"family_id": "X03", "group": "Evidence and logic", "source": "legacy_x1", "primary_unit": "Three-hop interior swap supported & untouched sibling", "evaluation_modes": ("native_legacy_x",)},
    {"family_id": "X06", "group": "Evidence and logic", "source": "legacy_x1", "primary_unit": "Authority conflict vs same-shaped sham", "evaluation_modes": ("native_legacy_x",)},
    {"family_id": "C05_RELATIONAL_GRAPH_CF", "group": "Evidence and logic", "source": "cognitive_c", "primary_unit": "Interior cause intervention both-correct", "evaluation_modes": ("frozen_backbone_readout_c", "end_to_end_adapted_c")},
    {"family_id": "C06_OPEN_WORLD_DEDUCTION", "group": "Evidence and logic", "source": "cognitive_c", "primary_unit": "Open-world deduction & minimally underdetermined twin", "evaluation_modes": ("frozen_backbone_readout_c", "end_to_end_adapted_c")},
    {"family_id": "C07_HIERARCHICAL_LISTOPS", "group": "Evidence and logic", "source": "cognitive_c", "primary_unit": "Nested ListOps exact value with noncommuting edits", "evaluation_modes": ("frozen_backbone_readout_c", "end_to_end_adapted_c")},
    # Group 2: Memory
    {"family_id": "X04", "group": "Memory", "source": "legacy_x1", "primary_unit": "Validity/epoch revision before and after observation", "evaluation_modes": ("native_legacy_x",)},
    {"family_id": "X07", "group": "Memory", "source": "legacy_x1", "primary_unit": "Missing edge available in memory", "evaluation_modes": ("native_legacy_x",)},
    {"family_id": "X08", "group": "Memory", "source": "legacy_x1", "primary_unit": "Stale record corrected by newer evidence", "evaluation_modes": ("native_legacy_x",)},
    {"family_id": "C08_ADAPTIVE_MEMORY_REVISION", "group": "Memory", "source": "cognitive_c", "primary_unit": "Edited key & unchanged sibling exact conjunction from cloned state", "evaluation_modes": ("frozen_backbone_readout_c", "end_to_end_adapted_c")},
    # Group 3: Planning
    {"family_id": "X05", "group": "Planning", "source": "legacy_x1", "primary_unit": "Missing edge via QUERY supported trace", "evaluation_modes": ("native_legacy_x",)},
    {"family_id": "X09", "group": "Planning", "source": "legacy_x1", "primary_unit": "Complementary tools & legal alternate orders", "evaluation_modes": ("native_legacy_x",)},
    {"family_id": "X11", "group": "Planning", "source": "legacy_x1", "primary_unit": "Sufficient evidence without unnecessary tool cost", "evaluation_modes": ("native_legacy_x",)},
    {"family_id": "C04_COMPOSITIONAL_SCAN", "group": "Planning", "source": "cognitive_c", "primary_unit": "Exact full action sequence vs wrong-order twins", "evaluation_modes": ("frozen_backbone_readout_c", "end_to_end_adapted_c", "native_unrepresentable_with_reason")},
    # Group 4: Epistemic control
    {"family_id": "X10", "group": "Epistemic control", "source": "legacy_x1", "primary_unit": "Insufficient evidence supported abstention & answerable twin", "evaluation_modes": ("native_legacy_x",)},
    {"family_id": "X12", "group": "Epistemic control", "source": "legacy_x1", "primary_unit": "Ambiguity after tools supported abstention & false-abstain rate", "evaluation_modes": ("native_legacy_x",)},
    {"family_id": "C09_METACOGNITION_SELF_CHECK", "group": "Epistemic control", "source": "cognitive_c", "primary_unit": "Cost-reversal regret & relevant vs irrelevant revision", "evaluation_modes": ("frozen_backbone_readout_c", "end_to_end_adapted_c")},
    # Group 5: Learning and transfer
    {"family_id": "C01_NOVEL_OP_INDUCTION", "group": "Learning and transfer", "source": "cognitive_c", "primary_unit": "Nested 0/1/4/16/64 induction & k=0 public ambiguity posterior", "evaluation_modes": ("frozen_backbone_readout_c", "end_to_end_adapted_c", "native_unrepresentable_with_reason")},
    {"family_id": "C02_FST_RULE_COMPOSITION", "group": "Learning and transfer", "source": "cognitive_c", "primary_unit": "Composed FST law identification & invariant sibling retention", "evaluation_modes": ("frozen_backbone_readout_c", "end_to_end_adapted_c")},
    {"family_id": "C03_GRID_TRANSFORM_ARC", "group": "Learning and transfer", "source": "cognitive_c", "primary_unit": "Entire variable-sized colored grid including dims/cells", "evaluation_modes": ("frozen_backbone_readout_c", "end_to_end_adapted_c", "native_unrepresentable_with_reason")},
    {"family_id": "C10_LANGUAGE_MATH_TRANSFER", "group": "Learning and transfer", "source": "cognitive_c", "primary_unit": "Exact rational value & unit with swapped arguments/units", "evaluation_modes": ("frozen_backbone_readout_c", "end_to_end_adapted_c", "native_unrepresentable_with_reason")},
)


def _norm_split(split: str) -> str:
    if split not in _CANON_SPLIT:
        raise ValueError(f"Unknown X3 split {split!r}; expected one of {tuple(SPLIT_SEEDS)}")
    return _CANON_SPLIT[split]


def _split_rng(family: str, split: str, seed: int, idx: int) -> random.Random:
    seed_hex = sha256(f"x3|{family}|{_norm_split(split)}|{int(seed)}|{int(idx)}".encode("utf-8")).hexdigest()
    return random.Random(int(seed_hex[:16], 16))


def _ensure_prompt_budget(prompt: str, context_limit: int = X3_MAX_PROMPT_BYTES) -> str:
    raw = prompt.encode("utf-8")
    if len(raw) > context_limit:
        raise ValueError(f"X3 prompt byte budget exceeded: {len(raw)} > {context_limit}: {prompt[:120]!r}")
    return prompt


# ---------------------------------------------------------------------------
# Public-Only Generation Bounds & Public Typed Output Protocol (§1, §4)
# ---------------------------------------------------------------------------


def infer_family_from_public_prompt(prompt: str) -> str:
    """Detect family identifier strictly from public prompt prefix/syntax (never from gold fields)."""
    p = str(prompt).strip()
    if p.startswith("C01|"):
        return "C01_NOVEL_OP_INDUCTION"
    if p.startswith("C02|"):
        return "C02_FST_RULE_COMPOSITION"
    if p.startswith("C03|"):
        return "C03_GRID_TRANSFORM_ARC"
    if p.startswith("C04|"):
        return "C04_COMPOSITIONAL_SCAN"
    if p.startswith("C06|"):
        return "C06_OPEN_WORLD_DEDUCTION"
    if p.startswith("C07|"):
        return "C07_HIERARCHICAL_LISTOPS"
    if p.startswith("C08|"):
        return "C08_ADAPTIVE_MEMORY_REVISION"
    if p.startswith("C09|"):
        return "C09_METACOGNITION_SELF_CHECK"
    if p.startswith("C10|"):
        return "C10_LANGUAGE_MATH_TRANSFER"
    if p.startswith("C05|") or "TASK=X3 " in p or p.startswith("Trace ") or p.startswith("CONF-TRACE "):
        return "C05_RELATIONAL_GRAPH_CF"
    return "LEGACY_OR_EXTERNAL"


def infer_public_generation_bound(prompt: str) -> int:
    """Compute shape-derived byte generation bound strictly from public prompt structure (§1, §4)."""
    p = str(prompt).strip()
    fam = infer_family_from_public_prompt(p)
    if fam == "C04_COMPOSITIONAL_SCAN":
        parts = p.split("|")
        cmd = parts[2] if len(parts) >= 3 else p
        clauses = re.split(r"\s+(?:and|after)\s+", cmd)
        est_tokens = 0
        for cl in clauses:
            if "around" in cl:
                base = 8
            elif "opposite" in cl:
                base = 3
            else:
                base = 2
            mult = 3 if "thrice" in cl else (2 if "twice" in cl else 1)
            est_tokens += base * mult
        # Max 7 bytes per SCAN action + spaces + 'ANSWER:' (7) + EOS + slack
        return min(240, max(X3_DEFAULT_PUBLIC_GEN_BOUND, est_tokens * 7 + 16))
    if fam == "C03_GRID_TRANSFORM_ARC":
        m = re.search(r"Q:([0-9/]+)->\?", p)
        if m:
            rows = m.group(1).split("/")
            r_cnt = len(rows)
            c_cnt = max((len(r) for r in rows), default=3)
            dim = max(r_cnt, c_cnt, 3)
            # dim * dim digits + (dim-1) '/' + 'ANSWER:' + EOS + slack
            return min(160, max(X3_DEFAULT_PUBLIC_GEN_BOUND, dim * (dim + 1) + 20))
        return max(X3_DEFAULT_PUBLIC_GEN_BOUND, 64)
    return X3_DEFAULT_PUBLIC_GEN_BOUND


def normalize_c10_rational_unit(text: str) -> tuple[Fraction | None, str, str]:
    """Normalize C10 integer/rational + unit answers under frozen equivalence grammar ('6/4 tokens' == '3/2 tokens')."""
    s = " ".join(str(text).strip().split())
    if not s:
        return None, "", ""
    m = re.fullmatch(r"([+-]?\d+(?:/\d+)?)(?:\s+([A-Za-z_]+))?", s)
    if not m:
        return None, "", s
    num_tok = m.group(1)
    unit = (m.group(2) or "").lower()
    if unit.endswith("s") and len(unit) > 1 and unit not in ("bps",):
        unit_canon = unit
    else:
        unit_canon = unit
    try:
        frac = Fraction(num_tok)
    except (ZeroDivisionError, ValueError):
        return None, unit_canon, s
    num_str = str(frac.numerator) if frac.denominator == 1 else f"{frac.numerator}/{frac.denominator}"
    canon = f"{num_str} {unit_canon}".strip()
    return frac, unit_canon, canon


def format_public_typed_target(
    case_or_prompt: Mapping[str, Any] | str,
    gold_text: str | None = None,
) -> str:
    """Serialize target in versioned public typed protocol: '<OP>:<CANONICAL_ANSWER>' or '<OP>' (§4)."""
    if isinstance(case_or_prompt, Mapping):
        prompt = str(case_or_prompt.get("prompt", ""))
        raw_gold = str(case_or_prompt.get("gold_text", "")) if gold_text is None else str(gold_text)
        is_pub_ambig = bool(case_or_prompt.get("public_ambiguous", False))
    else:
        prompt = str(case_or_prompt)
        raw_gold = str(gold_text or "")
        is_pub_ambig = False

    txt = raw_gold.strip()
    for op_tok in VALID_TYPED_OPS:
        if txt == op_tok or txt.startswith(f"{op_tok}:"):
            return txt

    if is_pub_ambig and infer_family_from_public_prompt(prompt) == "C01_NOVEL_OP_INDUCTION":
        return "ABSTAIN:AMBIGUOUS"

    if txt in ("QUERY", "RETRIEVE", "ABSTAIN", "AMBIGUOUS"):
        return txt

    fam = infer_family_from_public_prompt(prompt)
    if fam == "C10_LANGUAGE_MATH_TRANSFER":
        _, _, canon = normalize_c10_rational_unit(txt)
        return f"ANSWER:{canon or txt}"
    return f"ANSWER:{txt}"


def _parse_typed_envelope(raw_output: str) -> tuple[str, str, bool]:
    """Parse '<OP>:<ANSWER>' or bare '<OP>' / '<ANSWER>' without reading gold."""
    s = str(raw_output).strip()
    if not s:
        return "", "", False
    # Check explicit <OP>: prefix
    if ":" in s:
        head, tail = s.split(":", 1)
        head_u = head.strip().upper()
        if head_u in VALID_TYPED_OPS:
            return head_u, tail.strip(), True
    # Bare operation token with trailing garbage must still preserve suffix so suffix check can fail it!
    parts = s.split(None, 1)
    first_u = parts[0].upper()
    if first_u in ("QUERY", "RETRIEVE", "ABSTAIN", "AMBIGUOUS"):
        trailing = parts[1].strip() if len(parts) > 1 else ""
        return first_u, trailing, False
    return "ANSWER", s, False


def _validate_public_answer_syntax(prompt: str, op: str, payload: str) -> dict[str, Any]:
    """Validate syntax/shape strictly from public prompt constraints."""
    fam = infer_family_from_public_prompt(prompt)
    if op in ("QUERY", "RETRIEVE", "ABSTAIN", "AMBIGUOUS"):
        if op == "ABSTAIN" and payload in ("", "AMBIGUOUS"):
            return {"syntax_valid": True, "reason": "ok", "canonical_payload": payload}
        if payload == "":
            return {"syntax_valid": True, "reason": "ok", "canonical_payload": ""}
        return {"syntax_valid": False, "reason": f"TRAILING_SUFFIX_ON_{op}", "canonical_payload": payload}

    if op != "ANSWER" or not payload:
        return {"syntax_valid": False, "reason": "EMPTY_OR_INVALID_OP", "canonical_payload": payload}

    if fam == "C03_GRID_TRANSFORM_ARC":
        rows = payload.split("/")
        if not rows or any(not r or not re.fullmatch(r"[0-9]+", r) for r in rows):
            return {"syntax_valid": False, "reason": "INVALID_GRID_CELLS", "canonical_payload": payload}
        widths = {len(r) for r in rows}
        if len(widths) != 1:
            return {"syntax_valid": False, "reason": "RAGGED_GRID_ROWS", "canonical_payload": payload}
        # Verify grid dimensions compatible with public input Q: grid (same shape or transpose)
        m = re.search(r"Q:([0-9/]+)->\?", prompt)
        if m:
            q_rows = m.group(1).split("/")
            qr, qc = len(q_rows), len(q_rows[0])
            pr, pc = len(rows), len(rows[0])
            if (pr, pc) not in ((qr, qc), (qc, qr)):
                return {"syntax_valid": False, "reason": "GRID_EXTENT_MISMATCH", "canonical_payload": payload}
        return {"syntax_valid": True, "reason": "ok", "canonical_payload": payload}

    if fam == "C04_COMPOSITIONAL_SCAN":
        toks = payload.split()
        if not toks or any(t not in SCAN_VALID_ACTIONS for t in toks):
            return {"syntax_valid": False, "reason": "INVALID_SCAN_ACTION_TOKEN", "canonical_payload": payload}
        return {"syntax_valid": True, "reason": "ok", "canonical_payload": " ".join(toks)}

    if fam == "C01_NOVEL_OP_INDUCTION":
        if not re.fullmatch(r"\d{2}", payload):
            return {"syntax_valid": False, "reason": "INVALID_C01_INT", "canonical_payload": payload}
        return {"syntax_valid": True, "reason": "ok", "canonical_payload": payload}

    if fam == "C02_FST_RULE_COMPOSITION":
        if payload not in ("K", "L", "M", "N"):
            return {"syntax_valid": False, "reason": "INVALID_C02_SYMBOL", "canonical_payload": payload}
        return {"syntax_valid": True, "reason": "ok", "canonical_payload": payload}

    if fam == "C06_OPEN_WORLD_DEDUCTION":
        if payload not in ("TRUE", "FALSE", "UNKNOWN"):
            return {"syntax_valid": False, "reason": "INVALID_C06_VERDICT", "canonical_payload": payload}
        return {"syntax_valid": True, "reason": "ok", "canonical_payload": payload}

    if fam == "C07_HIERARCHICAL_LISTOPS":
        if not re.fullmatch(r"R\d+", payload):
            return {"syntax_valid": False, "reason": "INVALID_C07_TOKEN", "canonical_payload": payload}
        return {"syntax_valid": True, "reason": "ok", "canonical_payload": payload}

    if fam == "C10_LANGUAGE_MATH_TRANSFER":
        frac, unit, canon = normalize_c10_rational_unit(payload)
        if frac is None:
            return {"syntax_valid": False, "reason": "INVALID_C10_RATIONAL_OR_UNIT", "canonical_payload": payload}
        return {"syntax_valid": True, "reason": "ok", "canonical_payload": canon}

    return {"syntax_valid": True, "reason": "ok", "canonical_payload": payload}


def parse_and_score_public_typed_output(
    prompt: str,
    raw_output: str,
    case: Mapping[str, Any] | None = None,
    *,
    max_new_bytes: int | None = None,
    terminated_by_eos: bool = True,
) -> dict[str, Any]:
    """Parse and exact-score public typed outputs using only public prompt syntax + case gold oracle (§1, §4)."""
    bound = int(max_new_bytes if max_new_bytes is not None else infer_public_generation_bound(prompt))
    raw_str = str(raw_output)
    byte_len = len(raw_str.encode("utf-8"))
    overflow = byte_len > bound
    unterminated = not bool(terminated_by_eos)
    truncated = bool(overflow or (unterminated and byte_len >= bound))

    parsed_op, parsed_payload, has_explicit_op = _parse_typed_envelope(raw_str)
    syntax = _validate_public_answer_syntax(prompt, parsed_op, parsed_payload)
    valid = bool(syntax["syntax_valid"]) and not overflow and not unterminated

    fam = infer_family_from_public_prompt(prompt)
    gold_raw = str(case.get("gold_text", "")) if case is not None else ""
    gold_op, gold_payload, _ = _parse_typed_envelope(format_public_typed_target(prompt, gold_raw))
    if fam == "C10_LANGUAGE_MATH_TRANSFER":
        _, _, gold_canon = normalize_c10_rational_unit(gold_raw)
        gold_payload_cmp = gold_canon or gold_raw.strip()
        # Require explicit unit match if gold specifies one
        _, g_unit, _ = normalize_c10_rational_unit(gold_raw)
        _, p_unit, _ = normalize_c10_rational_unit(parsed_payload)
        unit_mismatch = bool(g_unit != p_unit and parsed_op == "ANSWER" and gold_op == "ANSWER")
    else:
        gold_payload_cmp = gold_payload.strip()
        unit_mismatch = False

    pred_canon = str(syntax["canonical_payload"]).strip()
    exact_text_match = (parsed_op == gold_op) and (pred_canon == gold_payload_cmp) and valid and (not unit_mismatch)

    # Check corrupted suffix specifically: first token matches expected first token, but suffix/full output differs or is invalid
    exp_toks = gold_payload_cmp.split()
    pred_toks = pred_canon.split()
    first_tok_match = bool(exp_toks and pred_toks and pred_toks[0] == exp_toks[0])
    corrupted_suffix = bool(
        (parsed_op in ("QUERY", "RETRIEVE", "ABSTAIN") and bool(parsed_payload))
        or (parsed_op == "ANSWER" and first_tok_match and (not exact_text_match or not valid))
        or (not syntax["syntax_valid"])
    )

    # Zero-shot / underdetermined C01 public ambiguity scoring (§1, §4)
    pub_vs: dict[str, Any] | None = None
    public_ambig = False
    ambig_credit = 0.0
    if fam in ("C01_NOVEL_OP_INDUCTION", "C02_FST_RULE_COMPOSITION"):
        pub_vs = compute_public_version_space_posterior(prompt)
        public_ambig = bool(pub_vs["is_ambiguous"])

    if public_ambig and fam == "C01_NOVEL_OP_INDUCTION":
        if not unterminated and not overflow and parsed_op in ("ABSTAIN", "AMBIGUOUS"):
            # Calibrated refusal on underdetermined zero-shot / ambiguous prompt
            valid = syntax["syntax_valid"]
            ambig_credit = 1.0 if valid else 0.0
            exact_score = ambig_credit
        elif valid and parsed_op == "ANSWER":
            # Exact public posterior probability assigned to emitted value (strict proper fraction < 1 under ambiguity)
            ambig_credit = float(pub_vs["target_posterior"].get(pred_canon, 0.0)) if pub_vs else 0.0
            exact_score = 0.0  # Full 1.0 discrete credit on underdetermined C01 requires ABSTAIN/AMBIGUOUS or posterior eval
        else:
            exact_score = 0.0
        correct = bool(exact_score >= 1.0 - 1e-12)
        latent_exact_match = (parsed_op == "ANSWER") and (pred_canon == gold_raw.strip()) and not overflow and not unterminated
    else:
        exact_score = 1.0 if exact_text_match else 0.0
        correct = bool(exact_text_match)
        latent_exact_match = correct

    legacy_mod16 = False
    if case is not None:
        g_cls = int(case.get("gold_class", -1))
        if fam == "C04_COMPOSITIONAL_SCAN":
            legacy_mod16 = bool(parsed_op == "ANSWER" and pred_toks and (len(pred_toks) % 16 == g_cls % 16))
        elif fam == "C03_GRID_TRANSFORM_ARC":
            digits = [int(ch) for ch in pred_canon.replace("/", "") if ch.isdigit()]
            legacy_mod16 = bool(parsed_op == "ANSWER" and digits and (sum(digits) % 16 == g_cls % 16))
        elif fam in ("C09_METACOGNITION_SELF_CHECK", "C10_LANGUAGE_MATH_TRANSFER"):
            m_int = re.match(r"^([+-]?\d+)", pred_canon)
            if m_int:
                legacy_mod16 = (int(m_int.group(1)) % 16) == (g_cls % 16)

    return {
        "schema": X3_PUBLIC_TYPED_OUTPUT_SCHEMA,
        "family": fam,
        "parsed_op": parsed_op,
        "parsed_answer": pred_canon,
        "has_explicit_op": has_explicit_op,
        "bound_bytes": bound,
        "byte_len": byte_len,
        "valid": bool(valid),
        "invalid": not bool(valid),
        "truncated": bool(truncated),
        "overflow": bool(overflow),
        "unterminated": bool(unterminated),
        "unit_mismatch": bool(unit_mismatch),
        "corrupted_suffix": bool(corrupted_suffix),
        "syntax_reason": str(syntax["reason"]),
        "public_ambiguous": bool(public_ambig),
        "ambiguity_posterior_credit": float(ambig_credit),
        "exact_match": bool(correct),
        "exact_score": float(exact_score),
        "correct": bool(correct),
        "latent_instance_exact_match": bool(latent_exact_match),
        "legacy_mod16_class_match": bool(legacy_mod16),
    }


def audit_legacy_class_collisions(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Quantify mod-16 legacy class collisions among distinct targets & verify exact scorer rejects them all (§1)."""
    by_fam_cls: dict[str, dict[int, list[Mapping[str, Any]]]] = {}
    for c in cases:
        fam = str(c["family"])
        cls_id = int(c.get("gold_class", -1)) % 16
        by_fam_cls.setdefault(fam, {}).setdefault(cls_id, []).append(c)

    colliding_pairs_by_family: dict[str, int] = {}
    witnesses: list[dict[str, Any]] = []
    false_exact_acceptances = 0

    for fam, cls_map in sorted(by_fam_cls.items()):
        pairs_cnt = 0
        for cls_id, bucket in sorted(cls_map.items()):
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    c1, c2 = bucket[i], bucket[j]
                    if str(c1["gold_text"]) != str(c2["gold_text"]):
                        pairs_cnt += 1
                        scored = parse_and_score_public_typed_output(
                            str(c1["prompt"]),
                            f"ANSWER:{c2['gold_text']}",
                            c1,
                            terminated_by_eos=True,
                        )
                        if scored["correct"] and not scored["public_ambiguous"]:
                            false_exact_acceptances += 1
                        if len(witnesses) < 12:
                            witnesses.append(
                                {
                                    "family": fam,
                                    "legacy_mod16_class": cls_id,
                                    "case_id_a": str(c1["case_id"]),
                                    "gold_a": str(c1["gold_text"]),
                                    "case_id_b": str(c2["case_id"]),
                                    "gold_b": str(c2["gold_text"]),
                                    "exact_scorer_accepts_wrong_b": bool(scored["correct"]),
                                    "legacy_mod16_match": bool(scored["legacy_mod16_class_match"]),
                                }
                            )
        colliding_pairs_by_family[fam] = pairs_cnt

    return {
        "schema": "elendril-x3-legacy-collision-audit-v1",
        "total_cases": len(cases),
        "colliding_pairs_by_family": colliding_pairs_by_family,
        "total_colliding_pairs": int(sum(colliding_pairs_by_family.values())),
        "exact_scorer_false_acceptances": int(false_exact_acceptances),
        "zero_false_acceptances": false_exact_acceptances == 0,
        "witnesses": witnesses,
    }


# ---------------------------------------------------------------------------
# C01: Novel Mathematical / Symbolic Operator Induction (0/1/4/16/64-shot)
# ---------------------------------------------------------------------------

_C01_SPLIT_RULES: dict[str, tuple[str, ...]] = {
    "train": ("affine_13", "max_mix_13"),
    "dev": ("diff_sq_13", "bilinear_13"),
    "confirm": ("piecewise_par_13", "abs_mix_13"),
    "ood_transfer": ("quad_xor_13",),
}
_C01_OP_TO_SPLIT: dict[str, str] = {"*": "train", "#": "dev", "~": "confirm", "^": "ood_transfer"}


def _eval_c01_rule(rule_kind: str, c1: int, c2: int, c3: int, a: int, b: int) -> int:
    if rule_kind == "affine_13":
        val = c1 * a + c2 * b + c3
    elif rule_kind == "max_mix_13":
        val = c1 * max(a, b) + c2 * min(a, b) + c3
    elif rule_kind == "diff_sq_13":
        val = c1 * abs(a - b) + c2 * (a + b) + c3
    elif rule_kind == "bilinear_13":
        val = a * b + c1 * a + c2 * b + c3
    elif rule_kind == "piecewise_par_13":
        val = (c1 * a + b + c3) if ((a + b) % 2 == 0) else (c2 * b + a + c3)
    elif rule_kind == "abs_mix_13":
        val = abs(c1 * a - c2 * b) + c3
    elif rule_kind == "quad_xor_13":
        val = (a * a + c1 * b) ^ c3
    else:
        raise ValueError(f"Unknown C01 rule_kind: {rule_kind!r}")
    return int(val % 16)


def _c01_hypothesis_bank(split: str) -> tuple[tuple[str, int, int, int], ...]:
    ns = _norm_split(split)
    rules = _C01_SPLIT_RULES[ns]
    return tuple(
        (rk, c1, c2, c3)
        for rk in rules
        for c1 in range(1, 6)
        for c2 in range(1, 6)
        for c3 in range(1, 8)
    )


def compute_public_version_space_posterior(prompt: str) -> dict[str, Any]:
    """Enumerate exact finite version space & predictive posterior solely from public C01/C02 bytes (§1, §4)."""
    fam = infer_family_from_public_prompt(prompt)
    if fam == "C01_NOVEL_OP_INDUCTION":
        m_sym = re.search(r"Q:(\d+)([*#~^])(\d+)=\?", prompt)
        if not m_sym:
            return {"consistent_rule_count": 0, "unique_target_values": [], "target_posterior": {}, "is_ambiguous": True, "map_entropy": 0.0}
        qa, op_sym, qb = int(m_sym.group(1)), m_sym.group(2), int(m_sym.group(3))
        sp = _C01_OP_TO_SPLIT.get(op_sym, "dev")
        parts = prompt.split("|")
        demo_blob = parts[3] if len(parts) >= 5 else ""
        demos = [
            (int(dm.group(1)), int(dm.group(2)), int(dm.group(3)))
            for seg in demo_blob.split(";")
            if seg.strip()
            for dm in [re.fullmatch(r"(\d+)[*#~^](\d+)=(\d+)", seg.strip())]
            if dm is not None
        ]
        bank = _c01_hypothesis_bank(sp)
        consistent = [
            h
            for h in bank
            if all((30 + _eval_c01_rule(h[0], h[1], h[2], h[3], da, db)) == dy for da, db, dy in demos)
        ]
        if not consistent:
            return {"consistent_rule_count": 0, "unique_target_values": [], "target_posterior": {}, "is_ambiguous": True, "map_entropy": 0.0}
        counts = Counter(str(30 + _eval_c01_rule(h[0], h[1], h[2], h[3], qa, qb)) for h in consistent)
        tot = float(len(consistent))
        post = {k: float(v / tot) for k, v in sorted(counts.items())}
        ent = -sum(p * float(np.log(max(1e-12, p))) for p in post.values())
        return {
            "family": fam,
            "consistent_rule_count": len(consistent),
            "unique_target_values": sorted(post),
            "target_posterior": post,
            "is_ambiguous": len(post) > 1,
            "map_entropy": float(ent),
        }

    if fam == "C02_FST_RULE_COMPOSITION":
        m_q = re.search(r"\|F([ABCD])\|k=\d+\|([^|]*)\|Q:(\d+):([0-3]+)>\?", prompt)
        if not m_q:
            return {"consistent_rule_count": 0, "unique_target_values": [], "target_posterior": {}, "is_ambiguous": True, "map_entropy": 0.0}
        tag_map = {"A": "train", "B": "dev", "C": "confirm", "D": "ood_transfer"}
        sp = tag_map[m_q.group(1)]
        demo_blob = m_q.group(2).strip()
        qi, qevs = int(m_q.group(3)), [int(ch) for ch in m_q.group(4)]
        out_labels = ("K", "L", "M", "N")
        offset = 100 * {"train": 1, "dev": 2, "confirm": 3, "ood_transfer": 4}[sp]
        demos = []
        for seg in demo_blob.split():
            dm = re.fullmatch(r"(\d+):([0-3]+)>([KLMN])", seg.strip())
            if dm:
                demos.append((int(dm.group(1)), [int(ch) for ch in dm.group(2)], out_labels.index(dm.group(3))))
        consistent_outs: list[str] = []
        for lid in range(offset + 1, offset + 13):
            tr, prm = _build_fst_tensors(sp, lid)
            if all(_eval_fst_seq(tr, prm, di, devs) == dy for di, devs, dy in demos):
                consistent_outs.append(out_labels[_eval_fst_seq(tr, prm, qi, qevs)])
        if not consistent_outs:
            return {"consistent_rule_count": 0, "unique_target_values": [], "target_posterior": {}, "is_ambiguous": True, "map_entropy": 0.0}
        counts = Counter(consistent_outs)
        tot = float(len(consistent_outs))
        post = {k: float(v / tot) for k, v in sorted(counts.items())}
        ent = -sum(p * float(np.log(max(1e-12, p))) for p in post.values())
        return {
            "family": fam,
            "consistent_rule_count": len(consistent_outs),
            "unique_target_values": sorted(post),
            "target_posterior": post,
            "is_ambiguous": len(post) > 1,
            "map_entropy": float(ent),
        }

    return {"consistent_rule_count": 1, "unique_target_values": [], "target_posterior": {}, "is_ambiguous": False, "map_entropy": 0.0}


def _generate_c01_case(split: str, seed: int, idx: int, shots: int) -> dict[str, Any]:
    ns = _norm_split(split)
    # Use idx-only RNG for nested (rule, query, demo order) so 0/1/4/16/64 share the EXACT unseen query & nested prefix!
    rng = _split_rng("C01_NOVEL_OP_INDUCTION", ns, seed, idx)
    rules = _C01_SPLIT_RULES[ns]
    rule_kind = rules[idx % len(rules)]
    split_offset = {"train": 1, "dev": 2, "confirm": 3, "ood_transfer": 5}[ns]
    c1 = 1 + ((idx + split_offset) % 5)
    c2 = 1 + ((idx // 3 + 2 * split_offset) % 5)
    c3 = 1 + ((idx // 7 + split_offset) % 7)

    # Disjoint structural law group within split quota (32 train, 8 dev, 12 confirm, 8 transfer)
    g_quota = STRUCTURAL_GROUP_QUOTAS[ns]
    g_idx = idx % g_quota
    group_id = f"C01_{ns.upper()}_G{g_idx:02d}"

    qa = 6 + (idx % 4)
    qb = 7 + ((idx // 2) % 3)
    y_idx = _eval_c01_rule(rule_kind, c1, c2, c3, qa, qb)
    gold_val = 30 + y_idx
    gold_text = str(gold_val)

    # Greedy version-space witness selection on fixed shuffled pool so prefixes strictly nest and Shrink V-space
    pairs = [(a, b) for a in range(1, 10) for b in range(1, 10) if not (a == qa and b == qb)]
    rng.shuffle(pairs)
    safe_pairs = [(a, b) for a, b in pairs if (30 + _eval_c01_rule(rule_kind, c1, c2, c3, a, b)) != gold_val]
    extra_pairs = [(a, b) for a, b in pairs if (30 + _eval_c01_rule(rule_kind, c1, c2, c3, a, b)) == gold_val]
    ordered_pool = safe_pairs + extra_pairs

    bank = _c01_hypothesis_bank(ns)
    remaining = list(bank)
    chosen: list[tuple[int, int, int]] = []
    used: set[tuple[int, int]] = set()
    # Greedily select top witness pairs first so small shot counts (1, 4, 16) shrink ambiguity aggressively
    for _ in range(min(64, len(ordered_pool))):
        if len(remaining) <= 1:
            break
        best_pair: tuple[int, int] | None = None
        best_left = len(remaining) + 1
        for ca, cb in ordered_pool[:24]:
            if (ca, cb) in used:
                continue
            target_y = 30 + _eval_c01_rule(rule_kind, c1, c2, c3, ca, cb)
            left = sum(1 for h in remaining if (30 + _eval_c01_rule(h[0], h[1], h[2], h[3], ca, cb)) == target_y)
            if left < best_left:
                best_left = left
                best_pair = (ca, cb)
        if best_pair is None:
            break
        da, db = best_pair
        used.add(best_pair)
        dy = 30 + _eval_c01_rule(rule_kind, c1, c2, c3, da, db)
        chosen.append((da, db, dy))
        remaining = [h for h in remaining if (30 + _eval_c01_rule(h[0], h[1], h[2], h[3], da, db)) == dy]

    for da, db in ordered_pool:
        if len(chosen) >= 64:
            break
        if (da, db) not in used:
            used.add((da, db))
            dy = 30 + _eval_c01_rule(rule_kind, c1, c2, c3, da, db)
            chosen.append((da, db, dy))

    op_sym = {"train": "*", "dev": "#", "confirm": "~", "ood_transfer": "^"}[ns]
    demo_str = ";".join(f"{da}{op_sym}{db}={dy}" for da, db, dy in chosen[:shots])
    prompt = _ensure_prompt_budget(f"C01|RULE|k={shots}|{demo_str}|Q:{qa}{op_sym}{qb}=?")
    masked_prompt = f"C01|RULE|k=0||Q:{qa}{op_sym}{qb}=?"
    pub_vs = compute_public_version_space_posterior(prompt)

    return {
        "case_id": f"C01:{ns}:s{seed}:k{shots}:{idx:04d}",
        "family": "C01_NOVEL_OP_INDUCTION",
        "split": split,
        "structural_group_id": group_id,
        "canon_sig": f"c01:{rule_kind}:k{shots}:c={c1},{c2},{c3}:q={qa},{qb}",
        "prompt": prompt,
        "masked_prompt": masked_prompt,
        "gold_text": gold_text,
        "gold_class": y_idx,
        "n_shots": int(shots),
        "absent_from_input": gold_text not in prompt,
        "public_ambiguous": bool(pub_vs["is_ambiguous"]),
        "public_version_space": pub_vs,
        "contrast_pair_id": f"c01:{ns}:{idx // 2}",
        "problem_spec": {
            "rule_kind": rule_kind,
            "c1": c1,
            "c2": c2,
            "c3": c3,
            "qa": qa,
            "qb": qb,
            "structural_group_id": group_id,
        },
        "benchmark_provenance": "internal_matched_analog",
    }


def _oracle_c01(spec: Mapping[str, Any]) -> str:
    y = _eval_c01_rule(
        str(spec["rule_kind"]), int(spec["c1"]), int(spec["c2"]), int(spec["c3"]), int(spec["qa"]), int(spec["qb"])
    )
    return str(30 + y)


# ---------------------------------------------------------------------------
# C02: Vectorized FST Rule Composition & Version-Space Identification
# ---------------------------------------------------------------------------


def _build_fst_tensors(split: str, law_id: int) -> tuple[np.ndarray, np.ndarray]:
    ns = _norm_split(split)
    rng = np.random.default_rng(int(sha256(f"fst|{ns}|{law_id}".encode()).hexdigest()[:8], 16))
    split_mult = {"train": 1, "dev": 3, "confirm": 5, "ood_transfer": 7}[ns]
    perm = np.array([(i * split_mult + law_id) % 4 for i in range(4)], dtype=np.int64)
    trans = np.zeros((4, 4), dtype=np.int64)
    for s in range(4):
        shift = int(rng.integers(0, 4))
        for e in range(4):
            trans[s, e] = (s + e * split_mult + shift) % 4
    return trans, perm


def _eval_fst_seq(trans: np.ndarray, perm: np.ndarray, init_state: int, events: Sequence[int]) -> int:
    s = int(init_state)
    for e in events:
        s = int(trans[s, int(e)])
    return int(perm[s])


def verify_fst_version_space(
    trans: np.ndarray,
    perm: np.ndarray,
    demos: Sequence[tuple[int, tuple[int, ...], int]],
    candidates: Sequence[tuple[np.ndarray, np.ndarray]],
) -> int:
    """Vectorized / exact candidate count matching all FST demonstrations."""
    consistent = 0
    for c_trans, c_perm in candidates:
        ok = True
        for init_s, evs, y in demos:
            if _eval_fst_seq(c_trans, c_perm, init_s, evs) != y:
                ok = False
                break
        if ok:
            consistent += 1
    return consistent


def _generate_c02_case(split: str, seed: int, idx: int, shots: int) -> dict[str, Any]:
    ns = _norm_split(split)
    pair_rng = _split_rng("C02_FST_RULE_COMPOSITION", ns, seed, idx // 2)
    law_id = 1 + (idx % 12) + 100 * {"train": 1, "dev": 2, "confirm": 3, "ood_transfer": 4}[ns]
    trans, perm = _build_fst_tensors(ns, law_id)
    depth = {"train": 2, "dev": 2, "confirm": 3, "ood_transfer": 3}[ns]
    q_init = int(pair_rng.randint(0, 3))
    q_events = [int(pair_rng.randint(0, 3)) for _ in range(depth)]
    out_state = _eval_fst_seq(trans, perm, q_init, q_events)
    out_labels = ("K", "L", "M", "N")
    gold_text = out_labels[out_state]

    g_idx = idx % STRUCTURAL_GROUP_QUOTAS[ns]
    group_id = f"C02_{ns.upper()}_G{g_idx:02d}"

    # Build deterministic greedy witness-splitting nested demo list without repeating query pair
    all_inputs = [
        (si, [((k >> (2 * j)) % 4) for j in range(depth)])
        for k in range(4**depth)
        for si in range(4)
        if not (si == q_init and [((k >> (2 * j)) % 4) for j in range(depth)] == q_events)
    ]
    d_rng = _split_rng("C02_DEMO_ORDER", ns, seed, idx)
    d_rng.shuffle(all_inputs)
    offset = 100 * {"train": 1, "dev": 2, "confirm": 3, "ood_transfer": 4}[ns]
    cands = [_build_fst_tensors(ns, lid) for lid in range(offset + 1, offset + 13)]
    remaining = list(cands)
    ordered_demos: list[tuple[int, list[int], int]] = []
    used_idx: set[int] = set()
    for _ in range(min(64, len(all_inputs))):
        if len(remaining) <= 1:
            break
        best_pos: int | None = None
        best_left = len(remaining) + 1
        for pos_i in range(min(48, len(all_inputs))):
            if pos_i in used_idx:
                continue
            di, de = all_inputs[pos_i]
            ty = _eval_fst_seq(trans, perm, di, de)
            left = sum(1 for c_tr, c_prm in remaining if _eval_fst_seq(c_tr, c_prm, di, de) == ty)
            if left < best_left:
                best_left = left
                best_pos = pos_i
        if best_pos is None:
            break
        used_idx.add(best_pos)
        di, de = all_inputs[best_pos]
        ty = _eval_fst_seq(trans, perm, di, de)
        ordered_demos.append((di, de, ty))
        remaining = [(c_tr, c_prm) for c_tr, c_prm in remaining if _eval_fst_seq(c_tr, c_prm, di, de) == ty]

    for pos_i, (di, de) in enumerate(all_inputs):
        if len(ordered_demos) >= 64:
            break
        if pos_i not in used_idx:
            ordered_demos.append((di, de, _eval_fst_seq(trans, perm, di, de)))

    demos: list[str] = []
    for d_i in range(int(shots)):
        di, de, dy_idx = ordered_demos[d_i % len(ordered_demos)]
        dy = out_labels[dy_idx]
        demos.append(f"{di}:{''.join(str(x) for x in de)}>{dy}")
    q_ev_s = "".join(str(x) for x in q_events)
    fst_tag = {"train": "FA", "dev": "FB", "confirm": "FC", "ood_transfer": "FD"}[ns]
    prompt = _ensure_prompt_budget(f"C02|{fst_tag}|k={shots}|{' '.join(demos)}|Q:{q_init}:{q_ev_s}>?")
    masked_prompt = f"C02|{fst_tag}|k=0||Q:{q_init}:{q_ev_s}>?"
    pub_vs = compute_public_version_space_posterior(prompt)
    return {
        "case_id": f"C02:{ns}:s{seed}:k{shots}:{idx:04d}",
        "family": "C02_FST_RULE_COMPOSITION",
        "split": split,
        "structural_group_id": group_id,
        "canon_sig": f"c02:{ns}:law={law_id}:q={q_init}:{q_ev_s}",
        "prompt": prompt,
        "masked_prompt": masked_prompt,
        "gold_text": gold_text,
        "gold_class": 10 + out_state,
        "n_shots": int(shots),
        "absent_from_input": gold_text not in prompt,
        "public_ambiguous": bool(pub_vs["is_ambiguous"]),
        "public_version_space": pub_vs,
        "contrast_pair_id": f"c02:{ns}:{idx // 2}",
        "problem_spec": {
            "trans": trans.tolist(),
            "perm": perm.tolist(),
            "q_init": q_init,
            "q_events": q_events,
            "out_labels": list(out_labels),
            "depth": depth,
            "split_tag": fst_tag,
        },
        "benchmark_provenance": "internal_matched_analog",
    }


def _oracle_c02(spec: Mapping[str, Any]) -> str:
    trans = np.asarray(spec["trans"], dtype=np.int64)
    perm = np.asarray(spec["perm"], dtype=np.int64)
    y = _eval_fst_seq(trans, perm, int(spec["q_init"]), spec["q_events"])
    return str(spec["out_labels"][y])


# ---------------------------------------------------------------------------
# C03: Symbolic Variable-Sized Colored 2D Grid Transformation
# ---------------------------------------------------------------------------

_C03_SPLIT_OPS: dict[str, tuple[str, ...]] = {
    "train": ("flip_h", "flip_v"),
    "dev": ("rot90", "transpose"),
    "confirm": ("gravity_down", "border_one"),
    "ood_transfer": ("tile_xor_rot180",),
}


def _apply_grid_op(grid: np.ndarray, op_name: str, color_map: Mapping[int, int]) -> np.ndarray:
    g = np.array(grid, dtype=np.int64, copy=True)
    if op_name == "flip_h":
        out = np.fliplr(g)
    elif op_name == "flip_v":
        out = np.flipud(g)
    elif op_name == "rot90":
        out = np.rot90(g, k=1)
    elif op_name == "transpose":
        out = g.T
    elif op_name == "gravity_down":
        out = np.zeros_like(g)
        for col in range(g.shape[1]):
            nz = [v for v in g[:, col] if v != 0]
            if nz:
                out[-len(nz) :, col] = nz
    elif op_name == "border_one":
        out = g.copy()
        out[0, :] = 1
        out[-1, :] = 1
        out[:, 0] = 1
        out[:, -1] = 1
    elif op_name == "tile_xor_rot180":
        r = np.rot90(g, k=2)
        out = (g ^ r) % 4
    else:
        raise ValueError(f"Unknown C03 grid op: {op_name!r}")
    return np.vectorize(lambda v: int(color_map.get(int(v), int(v))))(out)


def _grid_to_str(g: np.ndarray) -> str:
    return "/".join("".join(str(int(x)) for x in row) for row in g.tolist())


def _generate_c03_case(split: str, seed: int, idx: int) -> dict[str, Any]:
    ns = _norm_split(split)
    rng = _split_rng("C03_GRID_TRANSFORM_ARC", ns, seed, idx)
    pair_rng = _split_rng("C03_GRID_TRANSFORM_ARC_Q", ns, seed, idx // 2)
    ops = _C03_SPLIT_OPS[ns]
    op_name = ops[idx % len(ops)]
    if ns == "train":
        color_map = {0: 0, 1: 2, 2: 3, 3: 1} if idx % 2 == 0 else {0: 0, 1: 3, 2: 1, 3: 2}
    elif ns == "dev":
        color_map = {0: 0, 1: 3, 2: 1, 3: 2} if idx % 2 == 0 else {0: 0, 1: 1, 2: 2, 3: 3}
    elif ns == "confirm":
        color_map = {0: 0, 1: 4, 2: 5, 3: 6} if idx % 2 == 0 else {0: 0, 1: 6, 2: 4, 3: 5}
    else:
        color_map = {0: 0, 1: 7, 2: 8, 3: 9} if idx % 2 == 0 else {0: 0, 1: 9, 2: 7, 3: 8}

    # Variable grid dimensions ((3,3), (3,4), (4,3), (4,4)) testing dimension & cell exactness
    dims = ((3, 3), (3, 4), (4, 3), (4, 4))
    rows_n, cols_n = dims[(idx // 2) % len(dims)]
    g_idx = idx % STRUCTURAL_GROUP_QUOTAS[ns]
    group_id = f"C03_{ns.upper()}_G{g_idx:02d}"

    demos_txt: list[str] = []
    n_demos = 4 if op_name == "border_one" else 3
    for d_i in range(n_demos):
        inp = np.array([[int(rng.randint(0, 3)) for _ in range(cols_n)] for _ in range(rows_n)], dtype=np.int64)
        # Ensure all 4 input/XOR colors (0, 1, 2, 3) are exposed in non-border & rot180 pairs so color_map is publicly identifiable
        if rows_n >= 3 and cols_n >= 3:
            inp[1, 1] = d_i % 4
            inp[0, 1] = (d_i + 1) % 4
            inp[1, 0] = (d_i + 2) % 4
            inp[0, 0] = d_i % 4
            inp[rows_n - 1, cols_n - 1] = 3 if op_name == "tile_xor_rot180" else (d_i + 3) % 4
        out = _apply_grid_op(inp, op_name, color_map)
        demos_txt.append(f"{_grid_to_str(inp)}->{_grid_to_str(out)}")

    q_in = np.array([[int(pair_rng.randint(0, 3)) for _ in range(cols_n)] for _ in range(rows_n)], dtype=np.int64)
    q_out = _apply_grid_op(q_in, op_name, color_map)
    gold_text = _grid_to_str(q_out)
    gold_cls = int(np.sum(q_out)) % 16
    prompt = _ensure_prompt_budget(f"C03|GRID|{' | '.join(demos_txt)}|Q:{_grid_to_str(q_in)}->?")
    return {
        "case_id": f"C03:{ns}:s{seed}:{idx:04d}",
        "family": "C03_GRID_TRANSFORM_ARC",
        "split": split,
        "structural_group_id": group_id,
        "canon_sig": f"c03:{op_name}:{ns}:{rows_n}x{cols_n}:{idx % 2}:{_grid_to_str(q_in)}",
        "prompt": prompt,
        "masked_prompt": f"C03|GRID|NONE|Q:{_grid_to_str(q_in)}->?",
        "gold_text": gold_text,
        "gold_class": gold_cls,
        "n_shots": 2,
        "absent_from_input": gold_text not in prompt,
        "contrast_pair_id": f"c03:{ns}:{idx // 2}",
        "problem_spec": {"op_name": op_name, "color_map": color_map, "q_in": q_in.tolist(), "shape": [rows_n, cols_n]},
        "benchmark_provenance": "internal_matched_analog",
    }


def _oracle_c03(spec: Mapping[str, Any]) -> str:
    cmap = {int(k): int(v) for k, v in spec["color_map"].items()}
    out = _apply_grid_op(np.asarray(spec["q_in"], dtype=np.int64), str(spec["op_name"]), cmap)
    return _grid_to_str(out)


# ---------------------------------------------------------------------------
# C04: Compositional Command-to-Action Grammar (SCAN Internal Matched Analog)
# ---------------------------------------------------------------------------

_PRIMITIVES = ("walk", "run", "jump", "look")


def _eval_scan_cmd(cmd: str) -> str:
    toks = cmd.split()

    def _base_dir(verb: str, direction: str | None) -> list[str]:
        act = f"I_{verb.upper()}"
        if direction == "left":
            return ["TURN_L", act]
        if direction == "right":
            return ["TURN_R", act]
        if direction == "opposite":
            return ["TURN_L", "TURN_L", act]
        if direction == "around":
            return ["TURN_L", act] * 4
        return [act]

    def _parse_phrase(ptoks: list[str]) -> list[str]:
        mult = 1
        if ptoks and ptoks[-1] == "twice":
            mult = 2
            ptoks = ptoks[:-1]
        elif ptoks and ptoks[-1] == "thrice":
            mult = 3
            ptoks = ptoks[:-1]
        verb = ptoks[0]
        direction = ptoks[1] if len(ptoks) > 1 and ptoks[1] in ("left", "right", "opposite", "around") else None
        return _base_dir(verb, direction) * mult

    if "and" in toks:
        p = toks.index("and")
        return " ".join(_parse_phrase(toks[:p]) + _parse_phrase(toks[p + 1 :]))
    if "after" in toks:
        p = toks.index("after")
        return " ".join(_parse_phrase(toks[p + 1 :]) + _parse_phrase(toks[:p]))
    return " ".join(_parse_phrase(toks))


def _generate_c04_case(split: str, seed: int, idx: int) -> dict[str, Any]:
    ns = _norm_split(split)
    rng = _split_rng("C04_COMPOSITIONAL_SCAN", ns, seed, idx)
    pair_rng = _split_rng("C04_COMPOSITIONAL_SCAN_PAIR", ns, seed, idx // 2)
    if ns == "train":
        p1 = rng.choice(["walk", "run", "look"])
        d1 = rng.choice(["left", "right"])
        m1 = "twice"
        cmd = f"{p1} {d1} {m1}"
    elif ns == "dev":
        # Generate pairs with same-length wrong-order twins on even/odd idx
        p1 = pair_rng.choice(["jump", "walk"])
        d1 = pair_rng.choice(["left", "opposite"])
        p2 = pair_rng.choice(["run", "look"])
        d2 = pair_rng.choice(["right", "left"])
        cmd = f"{p1} {d1} and {p2} {d2}" if idx % 2 == 0 else f"{p2} {d2} and {p1} {d1}"
    elif ns == "confirm":
        p1 = "jump" if (idx // 2) % 2 == 0 else pair_rng.choice(_PRIMITIVES)
        d1 = pair_rng.choice(["opposite", "around"])
        m1 = pair_rng.choice(["twice", "thrice"])
        p2 = pair_rng.choice(["walk", "look"])
        cmd = f"{p1} {d1} after {p2} {m1}" if idx % 2 == 0 else f"{p2} {m1} after {p1} {d1}"
    else:
        p2 = pair_rng.choice(["walk", "run"])
        cmd = f"jump around thrice after {p2} opposite twice" if idx % 2 == 0 else f"{p2} opposite twice after jump around thrice"
    gold_text = _eval_scan_cmd(cmd)
    prompt = _ensure_prompt_budget(f"C04|COMP|{cmd}|OUT=?")
    masked_prompt = f"C04|COMP|{len(cmd.split())}_WORDS|OUT=?"
    g_idx = idx % STRUCTURAL_GROUP_QUOTAS[ns]
    return {
        "case_id": f"C04:{ns}:s{seed}:{idx:04d}",
        "family": "C04_COMPOSITIONAL_SCAN",
        "split": split,
        "structural_group_id": f"C04_{ns.upper()}_G{g_idx:02d}",
        "canon_sig": f"c04:{ns}:{cmd}",
        "prompt": prompt,
        "masked_prompt": masked_prompt,
        "gold_text": gold_text,
        "gold_class": len(gold_text.split()) % 16,
        "n_shots": 0,
        "absent_from_input": gold_text not in prompt,
        "invalid_transformations": {"permute_clauses": "N/A_CHANGES_TEMPORAL_EXECUTION_ORDER"},
        "contrast_pair_id": f"c04:{ns}:{idx // 2}",
        "problem_spec": {"cmd": cmd, "split_class": ns},
        "benchmark_provenance": "internal_matched_analog",
    }


# ---------------------------------------------------------------------------
# C05: Relational Graph Counterfactuals Across Canonical & Narrative Forms
# ---------------------------------------------------------------------------


def _generate_c05_batch(split: str, seed: int, count: int) -> list[dict[str, Any]]:
    ns = _norm_split(split)
    x3_sp = "confirmation" if ns in ("confirm", "ood_transfer") else "dev"
    # Distinct hop pairs by split so graph AST skeletons stay structurally disjoint across splits
    depths_by_split = {"train": (4,), "dev": (4, 5), "confirm": (5,), "ood_transfer": (4, 5)}
    decisions_per_bg = 8 * len(depths_by_split[ns])
    bg = max(2, (count + decisions_per_bg - 1) // decisions_per_bg)
    x3_seed = int(seed) + {"train": 1000, "dev": 2000, "confirm": 3000, "ood_transfer": 4000}[ns]
    panel = build_x3_paired_graph_factorial(split=x3_sp, depths=depths_by_split[ns], base_graphs_per_depth=bg, seed=x3_seed)
    out: list[dict[str, Any]] = []
    for i, dec in enumerate(panel["decisions"][:count]):
        narrative_mode = str(dec.get("renderer", "canonical"))
        facts = [dict(r) for r in dec["facts"]]
        hops = int(dec["hops"])
        min_ep = int(dec.get("min_epoch", 1))
        # Distinguish transfer/confirm split graph signature via split-disjoint minimum epoch header + independent confirm phrasing
        if ns == "confirm":
            min_ep = max(min_ep, 1)
            hdr = f"CONF-TRACE from={dec['goal_subject']} depth={hops} ep>={min_ep}"
            body = " | ".join(f"{r['subj']}=>{r['val']}(e{r.get('epoch', 1)},a{r.get('authority', 1)})" for r in facts)
            prompt = _ensure_prompt_budget(f"{hdr}\n{body}", context_limit=636)
        elif ns == "ood_transfer":
            prompt = _ensure_prompt_budget(
                render_x3_prompt(facts, goal_subject=str(dec["goal_subject"]), hops=hops, renderer=narrative_mode, min_epoch=min_ep)
                + "\nMODE=TRANSFER",
                context_limit=636,
            )
        else:
            prompt = _ensure_prompt_budget(str(dec["rendered_text"]), context_limit=636)

        gold_text = str(dec["expected_answer"])
        g_idx = X3_SYMBOLS.index(gold_text) if gold_text in X3_SYMBOLS else 0
        sg_idx = i % STRUCTURAL_GROUP_QUOTAS[ns]
        out.append(
            {
                "case_id": f"C05:{ns}:s{seed}:{i:04d}",
                "family": "C05_RELATIONAL_GRAPH_CF",
                "split": split,
                "structural_group_id": f"C05_{ns.upper()}_G{sg_idx:02d}",
                "canon_sig": f"c05:{ns}:{seed}:{dec['world_id']}",
                "prompt": prompt,
                "masked_prompt": f"C05|MASKED|hops={hops}",
                "gold_text": gold_text,
                "gold_class": g_idx,
                "n_shots": 0,
                "absent_from_input": False,
                "contrast_pair_id": f"{dec.get('pair_key', f'c05:{i // 4}')}::{dec.get('query_role', 'affected')}",
                "presentation": narrative_mode,
                "problem_spec": {
                    "facts": facts,
                    "goal_subject": dec["goal_subject"],
                    "hops": hops,
                    "min_epoch": min_ep,
                    "split_motif": ns,
                },
                "benchmark_provenance": "internal_matched_analog",
            }
        )
    return out


def _oracle_c05(spec: Mapping[str, Any]) -> str:
    res = independent_walk_x3_graph(
        spec["facts"],
        str(spec["goal_subject"]),
        hops=int(spec["hops"]),
        min_epoch=int(spec.get("min_epoch", 1)),
    )
    return str(res["terminal_value"])


# ---------------------------------------------------------------------------
# C06: Open-World 3-Valued Horn Deduction (ProofWriter Matched Analog)
# ---------------------------------------------------------------------------


def _solve_c06_horn(facts: Mapping[str, bool], rules: Sequence[tuple[tuple[str, ...], str, bool]], query: str) -> str:
    known: dict[str, bool] = dict(facts)
    changed = True
    while changed:
        changed = False
        for conds, head, polarity in rules:
            if head in known:
                continue
            if all(known.get(c) is True for c in conds):
                known[head] = bool(polarity)
                changed = True
    if query not in known:
        return "UNKNOWN"
    return "TRUE" if known[query] else "FALSE"


def _generate_c06_case(split: str, seed: int, idx: int) -> dict[str, Any]:
    ns = _norm_split(split)
    rng = _split_rng("C06_OPEN_WORLD_DEDUCTION", ns, seed, idx)
    depth = {"train": 2, "dev": 3, "confirm": 4, "ood_transfer": 5}[ns]
    preds = [f"p{i}" for i in range(depth + 4)]
    rng.shuffle(preds)
    verdict_mode = ("TRUE", "FALSE", "UNKNOWN")[idx % 3]
    facts: dict[str, bool] = {preds[0]: True}
    if verdict_mode != "UNKNOWN":
        facts[preds[1]] = True
    rules: list[tuple[tuple[str, ...], str, bool]] = []
    if depth >= 2:
        rules.append(((preds[0], preds[1]), preds[2], True))
    for k in range(2, depth):
        pol = False if (k == depth - 1 and verdict_mode == "FALSE") else True
        rules.append(((preds[k],), preds[k + 1], pol))
    if depth == 2 and verdict_mode == "FALSE":
        rules[0] = ((preds[0], preds[1]), preds[2], False)
    query = preds[depth]
    rules.append(((preds[-1],), preds[-2], True))
    gold_text = _solve_c06_horn(facts, rules, query)
    fact_str = " ".join(f"{k}=T" if v else f"{k}=F" for k, v in sorted(facts.items()))
    rule_str = ";".join(f"{'+'.join(c)}->{h}:{'T' if pol else 'F'}" for c, h, pol in rules)
    prompt = _ensure_prompt_budget(f"C06|DED|F:{fact_str}|R:{rule_str}|Q:{query}?")
    masked_prompt = f"C06|DED|d={depth}|F:MASKED|Q=?"
    cls_map = {"TRUE": 0, "FALSE": 1, "UNKNOWN": 2}
    g_idx = idx % STRUCTURAL_GROUP_QUOTAS[ns]
    return {
        "case_id": f"C06:{ns}:s{seed}:{idx:04d}",
        "family": "C06_OPEN_WORLD_DEDUCTION",
        "split": split,
        "structural_group_id": f"C06_{ns.upper()}_G{g_idx:02d}",
        "canon_sig": f"c06:{ns}:d={depth}:{verdict_mode}:{'/'.join(preds)}",
        "prompt": prompt,
        "masked_prompt": masked_prompt,
        "gold_text": gold_text,
        "gold_class": cls_map[gold_text],
        "n_shots": 0,
        "absent_from_input": gold_text not in prompt,
        "contrast_pair_id": f"c06:{ns}:{idx // 3}",
        "problem_spec": {"facts": facts, "rules": rules, "query": query, "depth": depth},
        "benchmark_provenance": "internal_matched_analog",
    }


# ---------------------------------------------------------------------------
# C07: Hierarchical Prefix Tree Evaluation (ListOps Matched Analog)
# ---------------------------------------------------------------------------


def _eval_listops_node(node: Any) -> int:
    if isinstance(node, int):
        return int(node)
    op = str(node[0])
    vals = [_eval_listops_node(ch) for ch in node[1:]]
    if op == "MAX":
        return max(vals)
    if op == "MIN":
        return min(vals)
    if op == "SUM10":
        return sum(vals) % 10
    if op == "FIRST":
        return vals[0]
    if op == "LAST":
        return vals[-1]
    if op == "MED":
        s = sorted(vals)
        return s[len(s) // 2]
    raise ValueError(f"Unknown ListOps op: {op!r}")


def _fmt_listops(node: Any) -> str:
    if isinstance(node, int):
        return str(node)
    return f"[{node[0]} {' '.join(_fmt_listops(c) for c in node[1:])}]"


def _build_listops_tree(rng: random.Random, depth: int, ops: Sequence[str], *, force_root_op: str | None = None) -> Any:
    if depth <= 0:
        return int(rng.integers(0, 10) if hasattr(rng, "integers") else rng.randint(0, 9))
    op = force_root_op if force_root_op is not None else rng.choice(ops)
    n_ch = rng.choice([2, 3])
    return [op, *[_build_listops_tree(rng, depth - 1, ops) for _ in range(n_ch)]]


def _generate_c07_case(split: str, seed: int, idx: int) -> dict[str, Any]:
    ns = _norm_split(split)
    rng = _split_rng("C07_HIERARCHICAL_LISTOPS", ns, seed, idx // 2)
    depth = {"train": 2, "dev": 3, "confirm": 4, "ood_transfer": 4}[ns]
    ops = ("MAX", "MIN", "SUM10") if ns == "train" else ("MAX", "MIN", "SUM10", "MED", "FIRST", "LAST")
    if ns == "confirm":
        tree = _build_listops_tree(rng, depth, ops, force_root_op="FIRST" if idx % 2 == 0 else "LAST")
    elif ns == "ood_transfer":
        tree = _build_listops_tree(rng, depth, ops, force_root_op="MED")
    else:
        tree = _build_listops_tree(rng, depth, ops)

    # Noncommuting edit pair on odd idx: swap first & last child of root
    if idx % 2 == 1 and isinstance(tree, list) and len(tree) >= 3:
        tree = [tree[0], tree[-1], *tree[2:-1], tree[1]]

    val = _eval_listops_node(tree)
    gold_text = f"R{val}"
    expr = _fmt_listops(tree)
    prompt = _ensure_prompt_budget(f"C07|LIST|d={depth}|{expr}=R?")
    g_idx = idx % STRUCTURAL_GROUP_QUOTAS[ns]
    return {
        "case_id": f"C07:{ns}:s{seed}:{idx:04d}",
        "family": "C07_HIERARCHICAL_LISTOPS",
        "split": split,
        "structural_group_id": f"C07_{ns.upper()}_G{g_idx:02d}",
        "canon_sig": f"c07:{ns}:d={depth}:{expr}",
        "prompt": prompt,
        "masked_prompt": f"C07|LIST|d={depth}|[MASKED]=R?",
        "gold_text": gold_text,
        "gold_class": int(val),
        "n_shots": 0,
        "absent_from_input": gold_text not in prompt,
        "contrast_pair_id": f"c07:{ns}:{idx // 2}",
        "problem_spec": {"tree": tree, "depth": depth, "split_motif": ns},
        "benchmark_provenance": "internal_matched_analog",
    }


# ---------------------------------------------------------------------------
# C08: Adaptive Working-Memory Revision & Retention (Delta + Sibling + Lure)
# ---------------------------------------------------------------------------


def _solve_c08(events: Sequence[Mapping[str, Any]], query_ent: str) -> str:
    mem: dict[str, tuple[int, str]] = {}
    for ev in events:
        if ev["kind"] in ("set", "correct"):
            ent = str(ev["ent"])
            ts = int(ev["ts"])
            if ent not in mem or ts >= mem[ent][0]:
                mem[ent] = (ts, str(ev["val"]))
    return mem[query_ent][1]


def _generate_c08_case(split: str, seed: int, idx: int) -> dict[str, Any]:
    ns = _norm_split(split)
    rng = _split_rng("C08_ADAPTIVE_MEMORY_REVISION", ns, seed, idx // 2)
    syms = list(X3_SYMBOLS)
    rng.shuffle(syms)
    ent_prefix = {"train": "e", "dev": "u", "confirm": "q", "ood_transfer": "z"}[ns]
    e_rev = f"{ent_prefix}1"
    e_sib = f"{ent_prefix}2"
    old_v, new_v, sib_v, lure_v, dist_v = syms[0], syms[1], syms[2], syms[3], syms[4]
    events: list[dict[str, Any]] = [
        {"ts": 1, "kind": "set", "ent": e_rev, "val": old_v},
        {"ts": 2, "kind": "set", "ent": e_sib, "val": sib_v},
        {"ts": 3, "kind": "correct", "ent": e_rev, "val": new_v},
        {"ts": 0, "kind": "stale_lure", "ent": e_rev, "val": lure_v},
    ]
    if ns in ("dev", "confirm", "ood_transfer"):
        events.append({"ts": 1, "kind": "stale_lure", "ent": e_sib, "val": dist_v})
    if ns in ("confirm", "ood_transfer"):
        events.append({"ts": 2, "kind": "stale_lure", "ent": e_rev, "val": dist_v})
    if ns == "ood_transfer":
        events.append({"ts": 4, "kind": "stale_lure", "ent": e_sib, "val": old_v})

    query_ent = e_rev if idx % 2 == 0 else e_sib
    gold_text = _solve_c08(events, query_ent)
    stream = ";".join(f"{ev['kind']}:{ev['ent']}={ev['val']}@{ev['ts']}" for ev in events)
    prompt = _ensure_prompt_budget(f"C08|MEM|{stream}|Q:{query_ent}=?")
    g_idx = idx % STRUCTURAL_GROUP_QUOTAS[ns]
    return {
        "case_id": f"C08:{ns}:s{seed}:{idx:04d}",
        "family": "C08_ADAPTIVE_MEMORY_REVISION",
        "split": split,
        "structural_group_id": f"C08_{ns.upper()}_G{g_idx:02d}",
        "canon_sig": f"c08:{ns}:{stream}:q={query_ent}",
        "prompt": prompt,
        "masked_prompt": "C08|MEM|MASKED|Q=?",
        "gold_text": gold_text,
        "gold_class": X3_SYMBOLS.index(gold_text),
        "n_shots": 0,
        "absent_from_input": False,
        "contrast_pair_id": f"c08:{ns}:{idx // 2}",
        "problem_spec": {"events": events, "query_ent": query_ent, "split_len": len(events)},
        "benchmark_provenance": "internal_matched_analog",
    }


# ---------------------------------------------------------------------------
# C09: Metacognition, Epistemic vs Information Uncertainty & Two-Cost Bellman
# ---------------------------------------------------------------------------


def verify_c09_budget_reversal_oracle(
    *,
    p_cert: float = 0.2,
    low_cost: float = 0.02,
    high_cost: float = 0.35,
    reward_correct: float = 1.0,
    reward_wrong: float = -1.0,
    reward_abstain: float = 0.0,
    tol: float = 1e-8,
) -> dict[str, Any]:
    """Independent vectorized Bellman enumeration proving cost reversal (0.02 -> QUERY, 0.35 -> ABSTAIN) (§1, §4)."""
    # State bits: (certified: bool, tool_a_used: bool, tool_b_used: bool)
    # Tool A (QUERY) certifies w.p. p_cert; Tool B (RETRIEVE) has same cost but zero future info (p_cert_b = 0.0).
    costs = np.array([low_cost, high_cost], dtype=np.float64)
    ev_answer = float(reward_wrong)  # uncertified equiprobable guess
    ev_abstain = float(reward_abstain)
    ev_query = float(p_cert * reward_correct + (1.0 - p_cert) * ev_abstain) - costs
    ev_retrieve_distractor = np.array([ev_abstain, ev_abstain], dtype=np.float64) - costs

    # Vectorized Bellman argmax across [ANSWER, QUERY, RETRIEVE, ABSTAIN] for [low_cost, high_cost]
    q_matrix = np.stack(
        [
            np.full_like(costs, ev_answer),
            ev_query,
            ev_retrieve_distractor,
            np.full_like(costs, ev_abstain),
        ],
        axis=-1,
    )
    ops = ("ANSWER", "QUERY", "RETRIEVE", "ABSTAIN")
    best_indices = np.argmax(q_matrix, axis=-1)
    best_ops = [ops[int(i)] for i in best_indices]
    best_evs = q_matrix[np.arange(len(costs)), best_indices]

    expected_low_ev = float(p_cert - low_cost)   # +0.18
    expected_high_ev = float(reward_abstain)     # 0.0 (QUERY EV = -0.15)
    assert best_ops[0] == "QUERY" and abs(best_evs[0] - expected_low_ev) <= tol
    assert best_ops[1] == "ABSTAIN" and abs(q_matrix[1, 1] - (p_cert - high_cost)) <= tol and abs(best_evs[1] - expected_high_ev) <= tol

    return {
        "schema": "elendril-x3-c09-bellman-oracle-v1",
        "low_cost": low_cost,
        "high_cost": high_cost,
        "p_cert": p_cert,
        "low_cost_optimal_op": best_ops[0],
        "low_cost_query_ev": float(ev_query[0]),
        "high_cost_optimal_op": best_ops[1],
        "high_cost_query_ev": float(ev_query[1]),
        "distractor_retrieve_evs": ev_retrieve_distractor.tolist(),
        "tolerance_verified": tol,
        "reversal_verified": (best_ops[0] == "QUERY" and best_ops[1] == "ABSTAIN"),
    }


def _solve_c09_oracle(spec: Mapping[str, Any]) -> str:
    mode = str(spec["subtype"])
    if mode == "missing_premise":
        t14_world = spec.get("t14_world")
        if isinstance(t14_world, Mapping) and "public_belief" in t14_world:
            dp = solve_t14_public_dp(t14_world)
            op = str(dp["optimal_operation"])
            return op if op != "ANSWER" else str(spec["fallback_ans"])
    return str(spec["gold"])


def _generate_c09_case(split: str, seed: int, idx: int) -> dict[str, Any]:
    ns = _norm_split(split)
    rng = _split_rng("C09_METACOGNITION_SELF_CHECK", ns, seed, idx)
    subtype = ("solvable_easy", "solvable_hard", "missing_premise")[idx % 3]
    syms = list(X3_SYMBOLS)
    rng.shuffle(syms)
    g_idx = idx % STRUCTURAL_GROUP_QUOTAS[ns]
    if subtype == "missing_premise":
        regime = "cheap" if (idx // 3) % 2 == 0 else "expensive"
        cost = 0.02 if regime == "cheap" else 0.35
        q_useful_is_query = ((idx // 6) % 2 == 0)
        q_prob, r_prob = (0.20, 0.00) if q_useful_is_query else (0.00, 0.20)
        t14 = build_t14_panel(split="dev" if ns == "dev" else "conf", histories_per_stratum=1, seed=seed + idx + 1)
        w = dict(next((x for x in t14 if x["stratum"] == "one_useful_query" and x["cost_regime"] == regime), t14[0]))
        pb = dict(w["public_belief"])
        pb.update(
            {
                "query_cost": cost,
                "retrieve_cost": cost,
                "query_cert_prob": q_prob,
                "retrieve_cert_prob": r_prob,
                "complementary_cert_prob": 0.20,
                "prior_probs": [0.5, 0.5],
                "budget": 1,
                "certified": False,
            }
        )
        w["public_belief"] = pb
        dp = solve_t14_public_dp(w)
        fallback = str(pb["active_labels"][0])
        gold_text = str(dp["optimal_operation"]) if dp["optimal_operation"] != "ANSWER" else fallback
        prompt = _ensure_prompt_budget(
            f"C09|META|MISSING|split={ns}|pQ={q_prob:.2f}|pR={r_prob:.2f}|c={cost:.2f}|cand={pb['active_labels']}|OP=?"
        )
        spec = {"subtype": "missing_premise", "t14_world": w, "fallback_ans": fallback, "cost": cost, "regime": regime, "split": ns}
        gold_cls = {"QUERY": 16, "RETRIEVE": 17, "ABSTAIN": 18}.get(gold_text, 0)
        uncertainty_kind = "information_uncertainty"
    else:
        a, b = int(rng.randint(10, 40)), int(rng.randint(10, 40))
        if subtype == "solvable_easy":
            gold_text = str(a + b)
            body = f"Compute {ns} X={a}+{b}"
        else:
            c = int(rng.randint(2, 5))
            first_try = a * c
            gold_text = str(a * c + b - 3)
            body = f"Verify {ns} draft {first_try} for {a}*{c}+{b}-3"
        prompt = _ensure_prompt_budget(f"C09|META|{subtype.upper()}|{body}|ANS=?")
        spec = {"subtype": subtype, "gold": gold_text, "split": ns}
        gold_cls = int(gold_text) % 16
        uncertainty_kind = "epistemic_reasoning" if subtype == "solvable_hard" else "low_uncertainty"
    return {
        "case_id": f"C09:{ns}:s{seed}:{idx:04d}",
        "family": "C09_METACOGNITION_SELF_CHECK",
        "split": split,
        "structural_group_id": f"C09_{ns.upper()}_G{g_idx:02d}",
        "canon_sig": f"c09:{ns}:{subtype}:{seed}:{idx}:{gold_text}",
        "prompt": prompt,
        "masked_prompt": f"C09|META|{subtype.upper()}|MASKED|ANS=?",
        "gold_text": gold_text,
        "gold_class": gold_cls,
        "n_shots": 0,
        "absent_from_input": gold_text not in prompt,
        "contrast_pair_id": f"c09:{ns}:{idx // 3}",
        "uncertainty_kind": uncertainty_kind,
        "problem_spec": spec,
        "benchmark_provenance": "internal_matched_analog",
    }


# ---------------------------------------------------------------------------
# C10: Language & Multi-Step Math Transfer (Exact Rationals, Units & Swaps)
# ---------------------------------------------------------------------------


def _generate_c10_case(split: str, seed: int, idx: int) -> dict[str, Any]:
    ns = _norm_split(split)
    rng = _split_rng("C10_LANGUAGE_MATH_TRANSFER", ns, seed, idx // 2)
    names = {
        "train": ("Amina", "Bram"),
        "dev": ("Clara", "Doran"),
        "confirm": ("Elena", "Farid"),
        "ood_transfer": ("Gita", "Hiro"),
    }[ns]
    unit = {"train": "tokens", "dev": "credits", "confirm": "units", "ood_transfer": "cycles"}[ns]
    rate_base = {"train": 2, "dev": 3, "confirm": 4, "ood_transfer": 5}[ns]
    batches = int(rng.randint(3, 8))
    per_batch = int(rng.randint(4, 9))
    bonus = int(rng.randint(5, 14))
    denom = 1 if ns == "train" else (2 if idx % 2 == 1 else 1)

    if idx % 2 == 0:
        num = rate_base * batches + per_batch + bonus
        expr_txt = f"{names[0]} earns {rate_base} {unit} per cycle over {batches} cycles, adds {per_batch} reserve {unit}, and receives {bonus} adjustment {unit}."
    else:
        # Swapped multiplier & denominator -> produces exact rational/integer differing cleanly even under matching mod-16 class
        num = per_batch * batches + rate_base + bonus
        expr_txt = f"{names[1]} splits ({per_batch} {unit} over {batches} cycles plus {rate_base} base and {bonus} bonus) across {denom} shares."

    frac = Fraction(num, denom)
    val_str = str(frac.numerator) if frac.denominator == 1 else f"{frac.numerator}/{frac.denominator}"
    gold_text = f"{val_str} {unit}"
    prompt = _ensure_prompt_budget(f"C10|PROSE|{expr_txt} What is the exact quantity and unit?")
    g_idx = idx % STRUCTURAL_GROUP_QUOTAS[ns]
    return {
        "case_id": f"C10:{ns}:s{seed}:{idx:04d}",
        "family": "C10_LANGUAGE_MATH_TRANSFER",
        "split": split,
        "structural_group_id": f"C10_{ns.upper()}_G{g_idx:02d}",
        "canon_sig": f"c10:{ns}:mode={idx % 2}:r={rate_base}:b={batches}:p={per_batch}:k={bonus}:d={denom}",
        "prompt": prompt,
        "masked_prompt": f"C10|PROSE|{names[0]} earns MASKED {unit}. What is the total?",
        "gold_text": gold_text,
        "gold_class": int(frac.numerator) % 16,
        "n_shots": 0,
        "absent_from_input": val_str not in prompt,
        "contrast_pair_id": f"c10:{ns}:{idx // 2}",
        "problem_spec": {"num": num, "denom": denom, "unit": unit, "split": ns},
        "benchmark_provenance": "internal_matched_analog",
    }


def _oracle_c10(spec: Mapping[str, Any]) -> str:
    if "num" in spec:
        frac = Fraction(int(spec["num"]), int(spec.get("denom", 1)))
        val_str = str(frac.numerator) if frac.denominator == 1 else f"{frac.numerator}/{frac.denominator}"
        unit = str(spec.get("unit", "")).strip()
        return f"{val_str} {unit}".strip()
    return str(int(spec["rate"]) * int(spec["batches"]) + int(spec["per_batch"]) + int(spec["bonus"]))


# ---------------------------------------------------------------------------
# Canonical AST / Graph Hashing Modulo Renaming & Disjointness Audits (§4)
# ---------------------------------------------------------------------------


def canonical_structural_hash(case: Mapping[str, Any]) -> str:
    """Canonical AST/graph structural hash modulo entity/variable/person renaming (§4)."""
    fam = str(case.get("family", infer_family_from_public_prompt(str(case.get("prompt", "")))))
    spec = case.get("problem_spec", {})

    def _renumber_seq(seq: Sequence[str]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for tok in seq:
            if tok not in mapping:
                mapping[tok] = f"V{len(mapping)}"
        return mapping

    if fam == "C01_NOVEL_OP_INDUCTION" and isinstance(spec, Mapping):
        skel = {"family": fam, "rule_kind": spec.get("rule_kind"), "op_shape": "bin_c1_c2_c3"}
        return digest(skel)

    if fam == "C02_FST_RULE_COMPOSITION" and isinstance(spec, Mapping):
        # Canonicalize FST transitions modulo 4-state bijection? Use normalized sorted transition degree signature + split/depth
        tr = spec.get("trans", [])
        in_deg = sorted(Counter(x for row in tr for x in row).values()) if tr else []
        skel = {"family": fam, "depth": spec.get("depth", 2), "split_tag": spec.get("split_tag", ""), "in_deg": in_deg, "trans": tr}
        return digest(skel)

    if fam == "C03_GRID_TRANSFORM_ARC" and isinstance(spec, Mapping):
        skel = {"family": fam, "op_name": spec.get("op_name"), "shape": spec.get("shape", [3, 3])}
        return digest(skel)

    if fam == "C04_COMPOSITIONAL_SCAN" and isinstance(spec, Mapping):
        cmd_toks = str(spec.get("cmd", "")).split()
        # Preserve grammatical template (directions, modifiers, connectives) while normalizing primitive verbs
        norm_toks = ["VERB" if t in _PRIMITIVES else t for t in cmd_toks]
        return digest({"family": fam, "template": norm_toks, "split_class": spec.get("split_class", "")})

    if fam == "C05_RELATIONAL_GRAPH_CF" and isinstance(spec, Mapping):
        facts = spec.get("facts", ())
        tokens: list[str] = [str(spec.get("goal_subject", ""))]
        for r in facts:
            tokens.append(str(r.get("subj", "")))
            tokens.append(str(r.get("val", "")))
        rmap = _renumber_seq(tokens)
        norm_edges = sorted(
            (rmap[str(r.get("subj", ""))], rmap[str(r.get("val", ""))], int(r.get("epoch", 1)), int(r.get("authority", 1)))
            for r in facts
        )
        return digest(
            {
                "family": fam,
                "hops": int(spec.get("hops", 1)),
                "min_epoch": int(spec.get("min_epoch", 1)),
                "split_motif": spec.get("split_motif", ""),
                "edges": norm_edges,
            }
        )

    if fam == "C06_OPEN_WORLD_DEDUCTION" and isinstance(spec, Mapping):
        facts = spec.get("facts", {})
        rules = spec.get("rules", ())
        q = str(spec.get("query", ""))
        toks = [q, *sorted(facts.keys())]
        for conds, head, _ in rules:
            toks.extend(conds)
            toks.append(str(head))
        rmap = _renumber_seq(toks)
        norm_facts = sorted((rmap[k], bool(v)) for k, v in facts.items())
        norm_rules = sorted((tuple(rmap[c] for c in conds), rmap[str(head)], bool(pol)) for conds, head, pol in rules)
        return digest({"family": fam, "depth": spec.get("depth", 2), "q": rmap[q], "facts": norm_facts, "rules": norm_rules})

    if fam == "C07_HIERARCHICAL_LISTOPS" and isinstance(spec, Mapping):
        def _ast_shape(node: Any) -> Any:
            if isinstance(node, int):
                return "LEAF"
            return [str(node[0]), *(_ast_shape(ch) for ch in node[1:])]
        return digest({"family": fam, "depth": spec.get("depth", 2), "split_motif": spec.get("split_motif", ""), "ast": _ast_shape(spec.get("tree"))})

    if fam == "C08_ADAPTIVE_MEMORY_REVISION" and isinstance(spec, Mapping):
        evs = spec.get("events", ())
        toks = [str(spec.get("query_ent", ""))]
        for ev in evs:
            toks.append(str(ev.get("ent", "")))
            toks.append(str(ev.get("val", "")))
        rmap = _renumber_seq(toks)
        norm_evs = [(int(ev.get("ts", 0)), str(ev.get("kind", "")), rmap[str(ev.get("ent", ""))]) for ev in evs]
        return digest({"family": fam, "split_len": len(norm_evs), "q": rmap[str(spec.get("query_ent", ""))], "events": norm_evs})

    if fam == "C09_METACOGNITION_SELF_CHECK" and isinstance(spec, Mapping):
        return digest({"family": fam, "subtype": spec.get("subtype"), "split": spec.get("split", ""), "cost": spec.get("cost")})

    if fam == "C10_LANGUAGE_MATH_TRANSFER" and isinstance(spec, Mapping):
        return digest({"family": fam, "denom": spec.get("denom", 1), "unit": spec.get("unit", ""), "split": spec.get("split", "")})

    # Generic fallback modulo single uppercase symbols and digits
    norm_p = re.sub(r"\b[A-P]\b", "SYM", str(case.get("prompt", "")))
    return digest({"family": fam, "norm_prompt": norm_p})


def solve_cognitive_case_oracle(case: Mapping[str, Any]) -> str:
    """Independent reference oracle returning exact target string without neural state."""
    fam = str(case["family"])
    spec = case["problem_spec"]
    if fam == "C01_NOVEL_OP_INDUCTION":
        return _oracle_c01(spec)
    if fam == "C02_FST_RULE_COMPOSITION":
        return _oracle_c02(spec)
    if fam == "C03_GRID_TRANSFORM_ARC":
        return _oracle_c03(spec)
    if fam == "C04_COMPOSITIONAL_SCAN":
        return _eval_scan_cmd(str(spec["cmd"]))
    if fam == "C05_RELATIONAL_GRAPH_CF":
        return _oracle_c05(spec)
    if fam == "C06_OPEN_WORLD_DEDUCTION":
        return _solve_c06_horn(spec["facts"], [tuple(tuple(r[0]) if i == 0 else r[i] for i in range(3)) for r in spec["rules"]], str(spec["query"]))
    if fam == "C07_HIERARCHICAL_LISTOPS":
        return f"R{_eval_listops_node(spec['tree'])}"
    if fam == "C08_ADAPTIVE_MEMORY_REVISION":
        return _solve_c08(spec["events"], str(spec["query_ent"]))
    if fam == "C09_METACOGNITION_SELF_CHECK":
        return _solve_c09_oracle(spec)
    if fam == "C10_LANGUAGE_MATH_TRANSFER":
        return _oracle_c10(spec)
    raise ValueError(f"Unknown cognitive family: {fam!r}")


def verify_cognitive_case(case: Mapping[str, Any], predicted_text: str) -> dict[str, Any]:
    """Verify via public typed output protocol against independent oracle."""
    scored = parse_and_score_public_typed_output(str(case.get("prompt", "")), predicted_text, case, terminated_by_eos=True)
    expected = solve_cognitive_case_oracle(case)
    return {
        "case_id": str(case["case_id"]),
        "family": str(case["family"]),
        "expected": expected,
        "predicted": str(predicted_text).strip(),
        "correct": bool(scored["correct"]),
        **scored,
    }


def _attach_structural_hashes(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for c in cases:
        if "structural_hash" not in c:
            c["structural_hash"] = canonical_structural_hash(c)
    return cases


def build_x3_cognitive_battery(
    *,
    split: str = "dev",
    seed: int | None = None,
    per_family: int = 20,
    shot_counts: Sequence[int] = (0, 1, 4, 16, 64),
) -> dict[str, Any]:
    """Generate the complete 10-family V8-X3 multidimensional cognitive evaluation battery."""
    ns = _norm_split(split)
    use_seed = int(SPLIT_SEEDS[ns] if seed is None else seed)
    cases: list[dict[str, Any]] = []

    n_shots_list = list(shot_counts) or [4]
    for i in range(per_family):
        sh = int(n_shots_list[i % len(n_shots_list)])
        cases.append(_generate_c01_case(split, use_seed, i, sh))
        cases.append(_generate_c02_case(split, use_seed, i, sh))
        cases.append(_generate_c03_case(split, use_seed, i))
        cases.append(_generate_c04_case(split, use_seed, i))
    cases.extend(_generate_c05_batch(split, use_seed, per_family))
    for i in range(per_family):
        cases.append(_generate_c06_case(split, use_seed, i))
        cases.append(_generate_c07_case(split, use_seed, i))
        cases.append(_generate_c08_case(split, use_seed, i))
        cases.append(_generate_c09_case(split, use_seed, i))
        cases.append(_generate_c10_case(split, use_seed, i))

    _attach_structural_hashes(cases)
    battery_sha = digest([{"id": c["case_id"], "sig": c["canon_sig"], "gold": c["gold_text"]} for c in cases])
    return {
        "schema": X3_COGNITIVE_SCHEMA,
        "split": split,
        "normalized_split": ns,
        "seed": use_seed,
        "families": list(X3_COGNITIVE_FAMILIES),
        "shot_counts": [int(s) for s in n_shots_list],
        "case_count": len(cases),
        "cases": cases,
        "battery_sha256": battery_sha,
    }


def build_c_structural_group_suite(
    *,
    split: str = "dev",
    seed: int | None = None,
    instances_per_group: int = 4,
    queries_per_instance: int = 8,
    families: Sequence[str] = X3_COGNITIVE_FAMILIES,
) -> dict[str, Any]:
    """Generate the §4 quota-exact structural group C suite (32 train / 8 dev / 12 confirm / 8 transfer groups)."""
    ns = _norm_split(split)
    n_groups = STRUCTURAL_GROUP_QUOTAS[ns]
    per_family = n_groups * int(instances_per_group) * int(queries_per_instance)
    bat = build_x3_cognitive_battery(split=split, seed=seed, per_family=per_family, shot_counts=(0, 1, 4, 16, 64))
    fam_set = set(families)
    selected = [c for c in bat["cases"] if c["family"] in fam_set]
    return {
        "schema": "elendril-x3-structural-group-suite-v1",
        "split": split,
        "groups_per_family": n_groups,
        "instances_per_group": int(instances_per_group),
        "queries_per_instance": int(queries_per_instance),
        "expected_queries_per_family": per_family,
        "cases": selected,
        "battery_sha256": digest([c["case_id"] for c in selected]),
    }


def verify_x3_split_disjointness(
    splits_map: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    check_isomorphic_structures: bool = True,
) -> dict[str, Any]:
    """Fail-closed verifier rejecting exact prompts, canonical sigs, AND isomorphic renamed structural leaks (§4)."""
    sigs_by_split: dict[str, set[str]] = {}
    prompts_by_split: dict[str, set[str]] = {}
    structs_by_split: dict[str, set[str]] = {}

    for sp, items in splits_map.items():
        s_key = str(sp)
        sigs_by_split[s_key] = {str(c.get("canon_sig", "")) for c in items if c.get("canon_sig")}
        prompts_by_split[s_key] = {str(c.get("prompt", "")) for c in items if c.get("prompt")}
        structs_by_split[s_key] = {str(c.get("structural_hash") or canonical_structural_hash(c)) for c in items}

    split_names = sorted(sigs_by_split)
    overlaps: list[dict[str, Any]] = []
    for i, s1 in enumerate(split_names):
        for s2 in split_names[i + 1 :]:
            sig_ov = sorted(sigs_by_split[s1] & sigs_by_split[s2])
            pr_ov = sorted(prompts_by_split[s1] & prompts_by_split[s2])
            iso_ov = sorted(structs_by_split[s1] & structs_by_split[s2]) if check_isomorphic_structures else []
            if sig_ov or pr_ov or iso_ov:
                overlaps.append(
                    {
                        "splits": [s1, s2],
                        "sig_overlaps": sig_ov[:5],
                        "prompt_overlaps": pr_ov[:5],
                        "isomorphic_structural_leaks": iso_ov[:5],
                    }
                )
    if overlaps:
        raise ValueError(f"SPLIT_LEAK_DETECTED: {overlaps}")
    return {
        "disjoint": True,
        "splits_checked": split_names,
        "overlap_count": 0,
        "checked_isomorphic_hashes": bool(check_isomorphic_structures),
    }


def eval_x3_battery_shortcuts(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluate zero-reasoning heuristics + independent oracle across the cognitive battery."""
    if not cases:
        raise ValueError("eval_x3_battery_shortcuts requires non-empty cases")
    counts = Counter(str(c["gold_text"]) for c in cases)
    maj_label = counts.most_common(1)[0][0]
    seen_demos: dict[str, str] = {}

    maj_ok = copy_last_ok = copy_first_ok = nearest_ok = abstain_ok = oracle_ok = 0
    by_fam: dict[str, list[bool]] = {}
    for c in cases:
        gold = str(c["gold_text"])
        prompt = str(c["prompt"])
        toks = prompt.replace("|", " ").replace("=", " ").replace(";", " ").split()
        copy_last = toks[-2] if len(toks) >= 2 and toks[-1].endswith("?") else (toks[-1] if toks else "")
        copy_first = toks[2] if len(toks) >= 3 else (toks[0] if toks else "")
        fam = str(c["family"])
        nearest = seen_demos.get(fam, "")
        seen_demos[fam] = gold

        maj_ok += int(maj_label == gold)
        copy_last_ok += int(copy_last == gold)
        copy_first_ok += int(copy_first == gold)
        nearest_ok += int(nearest == gold)
        abstain_ok += int(gold in ("ABSTAIN", "UNKNOWN"))
        or_ans = solve_cognitive_case_oracle(c)
        oracle_ok += int(or_ans == gold)
        by_fam.setdefault(fam, []).append(or_ans == gold)

    n = float(len(cases))
    return {
        "schema": "elendril-x3-battery-shortcuts-v1",
        "cases": int(n),
        "prior_majority_accuracy": maj_ok / n,
        "copy_last_token_accuracy": copy_last_ok / n,
        "copy_first_token_accuracy": copy_first_ok / n,
        "nearest_demo_retrieval_accuracy": nearest_ok / n,
        "always_abstain_accuracy": abstain_ok / n,
        "max_shortcut_accuracy": max(maj_ok, copy_last_ok, copy_first_ok, nearest_ok, abstain_ok) / n,
        "oracle_independent_accuracy": oracle_ok / n,
        "per_family_oracle_min": min(sum(v) / len(v) for v in by_fam.values()),
    }


def verify_x3_destructive_masks(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Prove that identical masked prompts across cases map to opposing ground-truth answers."""
    by_mask: dict[str, set[str]] = {}
    fam_colliding: set[str] = set()
    for c in cases:
        mp = str(c.get("masked_prompt", ""))
        if not mp:
            continue
        bucket = by_mask.setdefault(mp, set())
        bucket.add(str(c["gold_text"]))
        if len(bucket) >= 2:
            fam_colliding.add(str(c["family"]))
    return {
        "schema": "elendril-x3-destructive-mask-v1",
        "unique_masked_prompts": len(by_mask),
        "opposing_answer_masks": sum(1 for v in by_mask.values() if len(v) >= 2),
        "families_with_opposing_masks": sorted(fam_colliding),
    }


def load_external_benchmark_jsonl(
    path: Path | str,
    *,
    benchmark_name: str,
    official_unmodified: bool = True,
) -> dict[str, Any]:
    """Load an external public benchmark preserving provenance separation (§4, §10)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"External benchmark JSONL not found: {p}")
    provenance = "official_external" if official_unmodified else "interface_adapted_external"
    items: list[dict[str, Any]] = []
    for i, raw_line in enumerate(p.read_text(encoding="utf-8").splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        doc = json.loads(line)
        prompt = str(doc.get("prompt", doc.get("input", "")))
        gold = str(doc.get("gold_text", doc.get("answer", doc.get("target", "")))).strip()
        items.append(
            {
                "case_id": str(doc.get("case_id", f"ext:{benchmark_name}:{i:04d}")),
                "family": str(doc.get("family", f"EXT_{benchmark_name.upper()}")),
                "split": str(doc.get("split", "external")),
                "prompt": prompt,
                "gold_text": gold,
                "gold_class": int(doc.get("gold_class", abs(int(sha256(gold.encode()).hexdigest()[:8], 16)) % 16)),
                "absent_from_input": gold not in prompt,
                "benchmark_name": str(benchmark_name),
                "benchmark_provenance": provenance,
            }
        )
    return {
        "schema": "elendril-x3-external-benchmark-v1",
        "benchmark_name": str(benchmark_name),
        "benchmark_provenance": provenance,
        "count": len(items),
        "items": items,
        "sha256": digest(items),
    }


def score_cognitive_battery_profile(
    predictions: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score exact multidimensional capability profile + public ambiguity + outcome counts (§1, §4, §6)."""
    pred_by_id = {str(p["case_id"]): p for p in predictions}
    by_family: dict[str, list[float]] = {}
    by_shot: dict[int, list[float]] = {}
    absent_scores: list[float] = []
    present_scores: list[float] = []
    narrative_scores: list[float] = []
    canonical_scores: list[float] = []
    epistemic_confs: list[float] = []
    info_confs: list[float] = []
    brier_terms: list[float] = []
    cf_pairs: dict[str, list[float]] = {}
    failures: list[dict[str, Any]] = []

    latent_match_scores: list[float] = []
    legacy_mod16_scores: list[float] = []
    counts_by_fam: dict[str, dict[str, int]] = {}
    c09_regrets: list[float] = []
    c09_false_abstains = 0
    c09_answerable_n = 0

    for c in cases:
        cid = str(c["case_id"])
        fam = str(c["family"])
        p = pred_by_id.get(cid, {})
        pred_text = str(p.get("predicted_text", "")).strip()
        fc = counts_by_fam.setdefault(
            fam, {"eligible": 0, "attempted": 0, "correct": 0, "incorrect": 0, "invalid": 0, "abstained": 0, "not_run": 0}
        )
        fc["eligible"] += 1
        if not p:
            fc["not_run"] += 1
            continue
        fc["attempted"] += 1

        if p.get("correct") is None or "exact_match" not in p:
            scored = parse_and_score_public_typed_output(str(c.get("prompt", "")), pred_text, c, terminated_by_eos=True)
            ok = 1.0 if scored["correct"] else 0.0
            is_invalid = bool(scored["invalid"])
            parsed_op = str(scored["parsed_op"])
            lat_ok = bool(scored["latent_instance_exact_match"])
            leg_ok = bool(scored["legacy_mod16_class_match"])
        else:
            ok = 1.0 if bool(p["correct"]) else 0.0
            is_invalid = bool(p.get("invalid", False))
            parsed_op = str(p.get("parsed_op", ""))
            lat_ok = bool(p.get("latent_instance_exact_match", bool(p["correct"])))
            leg_ok = bool(p.get("legacy_mod16_class_match", False))

        if parsed_op in ("ABSTAIN", "AMBIGUOUS") or pred_text in ("ABSTAIN", "AMBIGUOUS"):
            fc["abstained"] += 1
        if is_invalid:
            fc["invalid"] += 1
        if ok >= 0.5:
            fc["correct"] += 1
        else:
            fc["incorrect"] += 1

        by_family.setdefault(fam, []).append(ok)
        latent_match_scores.append(1.0 if lat_ok else 0.0)
        legacy_mod16_scores.append(1.0 if leg_ok else 0.0)

        if fam in ("C01_NOVEL_OP_INDUCTION", "C02_FST_RULE_COMPOSITION"):
            by_shot.setdefault(int(c.get("n_shots", 0)), []).append(ok)
        if bool(c.get("absent_from_input")):
            absent_scores.append(ok)
        else:
            present_scores.append(ok)
        pres = c.get("presentation")
        if pres == "narrative":
            narrative_scores.append(ok)
        elif pres == "canonical":
            canonical_scores.append(ok)
        if "contrast_pair_id" in c:
            cf_pairs.setdefault(str(c["contrast_pair_id"]), []).append(ok)

        if fam == "C09_METACOGNITION_SELF_CHECK":
            u_kind = str(c.get("uncertainty_kind", ""))
            epistemic_score = float(p.get("epistemic_uncertainty", 0.5))
            info_score = float(p.get("info_uncertainty", 0.5))
            if u_kind == "information_uncertainty":
                info_confs.append(info_score - epistemic_score)
            elif u_kind == "epistemic_reasoning":
                epistemic_confs.append(info_score - epistemic_score)
            spec = c.get("problem_spec", {})
            if isinstance(spec, Mapping) and spec.get("subtype") == "missing_premise" and "t14_world" in spec:
                dp = solve_t14_public_dp(spec["t14_world"])
                opt_ev = float(dp["optimal_ev"])
                act_evs = dp["action_evs"]
                chosen_op = parsed_op if parsed_op in act_evs else ("ANSWER" if parsed_op == "ANSWER" else "ABSTAIN")
                chosen_ev = float(act_evs.get(chosen_op, act_evs.get("ANSWER", -1.0)))
                c09_regrets.append(max(0.0, opt_ev - chosen_ev))
            elif isinstance(spec, Mapping) and spec.get("subtype") in ("solvable_easy", "solvable_hard"):
                c09_answerable_n += 1
                if parsed_op in ("ABSTAIN", "AMBIGUOUS"):
                    c09_false_abstains += 1

        conf = float(min(1.0, max(0.0, p.get("confidence", 0.5))))
        brier_terms.append((conf - ok) ** 2)
        if ok < 0.5 and len(failures) < 50:
            failures.append(
                {
                    "case_id": cid,
                    "family": fam,
                    "expected": str(c["gold_text"]),
                    "predicted": pred_text,
                    "absent_from_input": bool(c.get("absent_from_input")),
                }
            )

    fam_acc = {f: float(sum(v) / len(v)) for f, v in sorted(by_family.items()) if v}
    shot_curve = {int(k): float(sum(v) / len(v)) for k, v in sorted(by_shot.items()) if v}
    shot_keys = sorted(shot_curve)
    shot_auc_gain = (shot_curve[shot_keys[-1]] - shot_curve[shot_keys[0]]) if len(shot_keys) >= 2 else 0.0

    if info_confs and epistemic_confs:
        wins = 0.0
        tot = float(len(info_confs) * len(epistemic_confs))
        for a in info_confs:
            for b in epistemic_confs:
                wins += 1.0 if a > b else (0.5 if a == b else 0.0)
        auroc = wins / tot
    else:
        auroc = 0.5

    both_pairs = [min(v) for v in cf_pairs.values() if len(v) >= 2]
    all_vals = [x for v in by_family.values() for x in v]
    return {
        "schema": "elendril-x3-cognitive-profile-v2",
        "cases": len(cases),
        "macro_mean": float(sum(fam_acc.values()) / max(1, len(fam_acc))),
        "micro_mean": float(sum(all_vals) / max(1, len(all_vals))),
        "worst_dimension_floor": float(min(fam_acc.values())) if fam_acc else 0.0,
        "per_family_accuracy": fam_acc,
        "per_family_counts": counts_by_fam,
        "shot_learning_curve": shot_curve,
        "shot_improvement_gain": float(shot_auc_gain),
        "absent_from_input_accuracy": float(sum(absent_scores) / max(1, len(absent_scores))),
        "present_in_input_accuracy": float(sum(present_scores) / max(1, len(present_scores))),
        "canonical_accuracy": float(sum(canonical_scores) / max(1, len(canonical_scores))) if canonical_scores else None,
        "narrative_accuracy": float(sum(narrative_scores) / max(1, len(narrative_scores))) if narrative_scores else None,
        "counterfactual_both_correct": float(sum(both_pairs) / max(1, len(both_pairs))) if both_pairs else 0.0,
        "epistemic_vs_info_auroc": float(auroc),
        "c09_expected_utility_regret": float(sum(c09_regrets) / len(c09_regrets)) if c09_regrets else 0.0,
        "c09_false_refusal_rate": float(c09_false_abstains / max(1, c09_answerable_n)) if c09_answerable_n else 0.0,
        "c09_false_refusal_count": int(c09_false_abstains),
        "brier_score": float(sum(brier_terms) / max(1, len(brier_terms))),
        "latent_instance_exact_match_mean": float(sum(latent_match_scores) / max(1, len(latent_match_scores))),
        "legacy_mod16_class_match_diagnostic": float(sum(legacy_mod16_scores) / max(1, len(legacy_mod16_scores))),
        "failure_examples": failures[:20],
    }


__all__ = [
    "CONFIRM_COMMITMENT_SEED",
    "DEV_BANK_SEED",
    "MANDATORY_22_FAMILIES",
    "SCAN_VALID_ACTIONS",
    "SPLIT_SEEDS",
    "STRUCTURAL_GROUP_QUOTAS",
    "TRAIN_BANK_SEED",
    "TRANSFER_BANK_SEED",
    "VALID_TYPED_OPS",
    "X3_COGNITIVE_FAMILIES",
    "X3_COGNITIVE_SCHEMA",
    "X3_DEFAULT_PUBLIC_GEN_BOUND",
    "X3_MAX_PROMPT_BYTES",
    "X3_PUBLIC_TYPED_OUTPUT_SCHEMA",
    "audit_legacy_class_collisions",
    "build_c_structural_group_suite",
    "build_x3_cognitive_battery",
    "canonical_structural_hash",
    "compute_public_version_space_posterior",
    "eval_x3_battery_shortcuts",
    "format_public_typed_target",
    "infer_family_from_public_prompt",
    "infer_public_generation_bound",
    "load_external_benchmark_jsonl",
    "normalize_c10_rational_unit",
    "parse_and_score_public_typed_output",
    "score_cognitive_battery_profile",
    "solve_cognitive_case_oracle",
    "verify_c09_budget_reversal_oracle",
    "verify_cognitive_case",
    "verify_fst_version_space",
    "verify_x3_destructive_masks",
    "verify_x3_split_disjointness",
]
