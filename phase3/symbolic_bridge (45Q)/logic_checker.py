"""
logic_checker.py — Logic and Proof Engine for 45Q Symbolic Logic Bridge.

Provides:
  - Truth table generation and classification (≤ 12 variables)
  - DPLL SAT solving (> 12 variables or as verification pass)
  - Syllogism verification (Barbara, MP, MT, HS, DS)
  - Natural deduction proof checking
  - Counterexample generation for invalid arguments

All results go through the dual-pass self-check layer before return.
"""

from __future__ import annotations
import itertools
from dataclasses import dataclass, field
from typing import Any, Optional

from .cnf_converter import (
    Formula, Var, Not, And, Or, Implies, Iff,
    parse_formula, to_cnf, Literal, Clause,
)
from .dpll_solver import solve_cnf, verify_assignment, SATResult
from .natural_deduction import check_proof, ProofCheckResult
from .response import (
    Mode, Verdict, VerifiedResult, VerificationPass, error_result
)
from . import config


# ── Truth Table ────────────────────────────────────────────────────────────────

@dataclass
class TruthTableResult:
    """
    Result of truth table evaluation.

    Attributes
    ----------
    variables : list[str]
        Variable names in evaluation order.
    rows : list[tuple[dict[str, bool], bool]]
        Each row: (assignment_dict, formula_value).
    classification : str
        One of: TAUTOLOGY, CONTRADICTION, SATISFIABLE.
    counterexample : Optional[dict[str, bool]]
        First row where formula is False (None if tautology).
    satisfying_example : Optional[dict[str, bool]]
        First row where formula is True (None if contradiction).
    num_rows : int
        Total number of rows.
    table_str : str
        ASCII-formatted truth table.
    """
    variables: list[str]
    rows: list[tuple[dict[str, bool], bool]]
    classification: str
    counterexample: Optional[dict[str, bool]]
    satisfying_example: Optional[dict[str, bool]]
    num_rows: int
    table_str: str


def _collect_vars(formula: Formula) -> list[str]:
    """
    Collect all variable names from a formula in alphabetical order.

    Parameters
    ----------
    formula : Formula
        Any propositional formula.

    Returns
    -------
    list[str]
        Sorted list of unique variable names.
    """
    names: set[str] = set()

    def _walk(f: Formula) -> None:
        if isinstance(f, Var):
            names.add(f.name)
        elif isinstance(f, Not):
            _walk(f.sub)
        elif isinstance(f, (And, Or, Implies, Iff)):
            _walk(f.left)
            _walk(f.right)

    _walk(formula)
    return sorted(names)


def _eval_formula(formula: Formula, assignment: dict[str, bool]) -> bool:
    """
    Evaluate a formula under a given truth assignment.

    Parameters
    ----------
    formula : Formula
        The formula to evaluate.
    assignment : dict[str, bool]
        Maps variable names to truth values.

    Returns
    -------
    bool
        The truth value of the formula.

    Raises
    ------
    KeyError
        If a variable in the formula is not in the assignment.
    """
    if isinstance(formula, Var):
        return assignment[formula.name]
    elif isinstance(formula, Not):
        return not _eval_formula(formula.sub, assignment)
    elif isinstance(formula, And):
        return _eval_formula(formula.left, assignment) and _eval_formula(formula.right, assignment)
    elif isinstance(formula, Or):
        return _eval_formula(formula.left, assignment) or _eval_formula(formula.right, assignment)
    elif isinstance(formula, Implies):
        return (not _eval_formula(formula.left, assignment)) or _eval_formula(formula.right, assignment)
    elif isinstance(formula, Iff):
        return _eval_formula(formula.left, assignment) == _eval_formula(formula.right, assignment)
    raise TypeError(f"Unknown formula type: {type(formula)}")


def build_truth_table(formula: Formula) -> TruthTableResult:
    """
    Generate the complete truth table for a propositional formula.

    Only called when the number of variables ≤ TRUTH_TABLE_MAX_VARS.
    Enumerates all 2^n assignments and evaluates the formula for each.

    Parameters
    ----------
    formula : Formula
        The formula to evaluate.

    Returns
    -------
    TruthTableResult
        Complete truth table with classification.
    """
    variables = _collect_vars(formula)
    n = len(variables)
    rows: list[tuple[dict[str, bool], bool]] = []
    all_true = True
    all_false = True
    counterexample: Optional[dict[str, bool]] = None
    satisfying_example: Optional[dict[str, bool]] = None

    for values in itertools.product([False, True], repeat=n):
        assignment = dict(zip(variables, values))
        result = _eval_formula(formula, assignment)
        rows.append((assignment, result))
        if result:
            all_false = False
            if satisfying_example is None:
                satisfying_example = dict(assignment)
        else:
            all_true = False
            if counterexample is None:
                counterexample = dict(assignment)

    if all_true:
        classification = "TAUTOLOGY"
    elif all_false:
        classification = "CONTRADICTION"
    else:
        classification = "SATISFIABLE"

    table_str = _format_truth_table(variables, rows, formula)

    return TruthTableResult(
        variables=variables,
        rows=rows,
        classification=classification,
        counterexample=counterexample,
        satisfying_example=satisfying_example,
        num_rows=len(rows),
        table_str=table_str,
    )


def _format_truth_table(
    variables: list[str],
    rows: list[tuple[dict[str, bool], bool]],
    formula: Formula,
) -> str:
    """
    Format a truth table as an ASCII grid.

    Parameters
    ----------
    variables : list[str]
        Variable names (columns).
    rows : list[tuple[dict[str, bool], bool]]
        Truth table rows.
    formula : Formula
        The formula (used as the header of the result column).

    Returns
    -------
    str
        ASCII-formatted truth table.
    """
    formula_header = str(formula)
    # Trim header if too long
    if len(formula_header) > 40:
        formula_header = formula_header[:37] + "..."

    col_widths = [max(len(v), 1) for v in variables]
    result_width = max(len(formula_header), 6)

    # Header row
    header_cells = [v.center(w) for v, w in zip(variables, col_widths)]
    header_cells.append(formula_header.center(result_width))
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths + [result_width]) + "+"
    header = "| " + " | ".join(header_cells) + " |"

    lines = [sep, header, sep]

    max_display = min(len(rows), config.MAX_TRUTH_TABLE_DISPLAY_ROWS)
    for assignment, value in rows[:max_display]:
        cells = []
        for v, w in zip(variables, col_widths):
            cells.append(("T" if assignment[v] else "F").center(w))
        cells.append(("T" if value else "F").center(result_width))
        lines.append("| " + " | ".join(cells) + " |")

    lines.append(sep)
    if len(rows) > max_display:
        lines.append(f"  ... ({len(rows) - max_display} more rows truncated)")

    return "\n".join(lines)


# ── Logic checker main function ────────────────────────────────────────────────

def check_formula(formula_str: str) -> VerifiedResult:
    """
    Check whether a propositional formula is a tautology, contradiction,
    or satisfiable. Uses truth table for ≤ 12 variables, DPLL for larger.

    Both methods are applied and results cross-checked (dual-pass).

    Parameters
    ----------
    formula_str : str
        A propositional formula string (natural language or symbolic).

    Returns
    -------
    VerifiedResult
        Complete result with verdict, truth table, and DPLL confirmation.
    """
    try:
        formula = parse_formula(formula_str)
    except ValueError as e:
        return error_result(Mode.LOGIC, formula_str, f"Parse error: {e}")

    variables = _collect_vars(formula)
    n = len(variables)
    steps: list[str] = [
        f"Parsed formula: {formula}",
        f"Variables: {variables} (n={n})",
    ]

    # ── Pass A: Truth table (if feasible) ─────────────────────────────────────
    truth_table_result: Optional[TruthTableResult] = None
    pass_a: Optional[VerificationPass] = None

    if n <= config.TRUTH_TABLE_MAX_VARS:
        truth_table_result = build_truth_table(formula)
        pass_a = VerificationPass(
            method="Truth Table (2^n exhaustive)",
            result=truth_table_result.classification,
            success=True,
        )
        steps.append(f"Truth table: {truth_table_result.num_rows} rows → {truth_table_result.classification}")
        steps.append(truth_table_result.table_str)

    # ── Pass B: DPLL SAT solver ────────────────────────────────────────────────
    # For SATISFIABLE check: is the formula SAT?
    # For TAUTOLOGY check: is the negation UNSAT?
    cnf_result = to_cnf(formula)
    dpll_sat = solve_cnf(cnf_result.cnf, cnf_result.original_vars)

    # Check if negation is SAT (to determine tautology)
    neg_formula = Not(formula)
    neg_cnf = to_cnf(neg_formula)
    dpll_neg_sat = solve_cnf(neg_cnf.cnf, neg_cnf.original_vars)

    if not dpll_sat.satisfiable:
        dpll_class = "CONTRADICTION"
    elif not dpll_neg_sat.satisfiable:
        dpll_class = "TAUTOLOGY"
    else:
        dpll_class = "SATISFIABLE"

    pass_b = VerificationPass(
        method="DPLL SAT Solver",
        result=dpll_class,
        success=True,
    )
    steps.append(f"DPLL: formula SAT={dpll_sat.satisfiable}, negation SAT={dpll_neg_sat.satisfiable}")
    steps.append(f"DPLL classification: {dpll_class}")

    # ── Cross-check ────────────────────────────────────────────────────────────
    tt_class = truth_table_result.classification if truth_table_result else None

    if tt_class is not None:
        agree = (tt_class == dpll_class)
        confidence = 1.0 if agree else 0.0
        if not agree:
            return VerifiedResult(
                mode=Mode.LOGIC,
                verdict=Verdict.UNCERTAIN,
                confidence=0.0,
                answer_text=f"VERIFIER DISAGREEMENT: Truth table={tt_class}, DPLL={dpll_class}",
                raw_input=formula_str,
                parsed_repr=str(formula),
                steps=steps,
                passes=[pass_a, pass_b],
                warnings=["Both verifiers disagree — manual review required"],
            )
        classification = tt_class
    else:
        confidence = 0.95  # Only DPLL, no truth table cross-check
        classification = dpll_class

    # ── Determine verdict ──────────────────────────────────────────────────────
    verdict_map = {
        "TAUTOLOGY": Verdict.TAUTOLOGY,
        "CONTRADICTION": Verdict.CONTRADICTION,
        "SATISFIABLE": Verdict.SATISFIABLE,
    }
    verdict = verdict_map[classification]

    counterexample = None
    if truth_table_result and truth_table_result.counterexample:
        counterexample = truth_table_result.counterexample
    elif dpll_neg_sat.satisfiable and dpll_neg_sat.assignment:
        # Extract original vars from DPLL assignment
        counterexample = {
            k: v for k, v in dpll_neg_sat.assignment.items()
            if k in variables
        }

    result_text = classification
    if counterexample and classification != "TAUTOLOGY":
        result_text += f" (counterexample: {counterexample})"

    passes = [p for p in [pass_a, pass_b] if p is not None]

    return VerifiedResult(
        mode=Mode.LOGIC,
        verdict=verdict,
        confidence=confidence,
        answer=classification,
        answer_text=result_text,
        raw_input=formula_str,
        parsed_repr=str(formula),
        steps=steps,
        passes=passes,
        counterexample=counterexample,
    )


# ── Syllogism verifier ─────────────────────────────────────────────────────────

SYLLOGISM_FORMS = {
    "BARBARA": "All M are P. All S are M. ∴ All S are P.",
    "MODUS_PONENS": "P → Q. P. ∴ Q.",
    "MODUS_TOLLENS": "P → Q. ¬Q. ∴ ¬P.",
    "HYPOTHETICAL_SYLLOGISM": "P → Q. Q → R. ∴ P → R.",
    "DISJUNCTIVE_SYLLOGISM": "P ∨ Q. ¬P. ∴ Q.",
}


def verify_syllogism(premises: list[str], conclusion: str) -> VerifiedResult:
    """
    Verify whether a set of premises logically entails a conclusion.

    Constructs the formula (P1 ∧ P2 ∧ ... ∧ Pn → C) and checks if it
    is a tautology using both truth table and DPLL.

    Parameters
    ----------
    premises : list[str]
        Premise strings (propositional formulas).
    conclusion : str
        Conclusion string (propositional formula).

    Returns
    -------
    VerifiedResult
        Whether the argument is valid, with explanation.
    """
    try:
        parsed_premises = [parse_formula(p) for p in premises]
        parsed_conclusion = parse_formula(conclusion)
    except ValueError as e:
        raw = f"Premises: {premises}  Conclusion: {conclusion}"
        return error_result(Mode.LOGIC, raw, f"Parse error: {e}")

    # Build P1 ∧ P2 ∧ ... ∧ Pn → C
    if not parsed_premises:
        return error_result(Mode.LOGIC, "", "No premises provided")

    conjunction = parsed_premises[0]
    for p in parsed_premises[1:]:
        conjunction = And(conjunction, p)

    entailment_formula = Implies(conjunction, parsed_conclusion)
    entailment_str = str(entailment_formula)

    result = check_formula(entailment_str)
    result.raw_input = f"Premises: {premises} ∴ {conclusion}"
    result.steps.insert(0, f"Entailment formula: {entailment_str}")

    if result.verdict == Verdict.TAUTOLOGY:
        result.answer_text = "VALID argument: conclusion necessarily follows from premises"
    elif result.verdict == Verdict.SATISFIABLE:
        result.answer_text = (
            "INVALID argument: conclusion does NOT necessarily follow. "
            f"Counterexample: {result.counterexample}"
        )
    elif result.verdict == Verdict.CONTRADICTION:
        result.answer_text = "PREMISES are contradictory (argument is vacuously valid)"

    return result


def verify_proof(proof_text: str) -> VerifiedResult:
    """
    Check a natural deduction proof and return a VerifiedResult.

    Parameters
    ----------
    proof_text : str
        Multi-line proof with numbered steps and justifications.

    Returns
    -------
    VerifiedResult
        VALID_PROOF or INVALID_PROOF with error details.
    """
    check = check_proof(proof_text)
    steps = list(check.steps)
    steps.insert(0, "Natural Deduction Proof Check:")

    if check.valid:
        return VerifiedResult(
            mode=Mode.LOGIC,
            verdict=Verdict.VALID_PROOF,
            confidence=1.0,
            answer=True,
            answer_text=f"VALID PROOF ({check.steps_checked} steps verified)",
            raw_input=proof_text,
            steps=steps,
            passes=[VerificationPass(
                method="Natural Deduction Rule Checker",
                result="VALID",
                success=True,
            )],
        )
    else:
        return VerifiedResult(
            mode=Mode.LOGIC,
            verdict=Verdict.INVALID_PROOF,
            confidence=1.0,
            answer=False,
            answer_text=(
                f"INVALID PROOF: error at line {check.first_error_line}. "
                f"{check.error_message}"
            ),
            raw_input=proof_text,
            steps=steps,
            passes=[VerificationPass(
                method="Natural Deduction Rule Checker",
                result="INVALID",
                success=True,
            )],
            warnings=[check.error_message],
        )
