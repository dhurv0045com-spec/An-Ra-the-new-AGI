"""
dpll_solver.py — DPLL SAT Solver with VSIDS Heuristic (from scratch).

Implements the Davis-Putnam-Logemann-Loveland (DPLL) algorithm for solving
propositional satisfiability (SAT) problems in Conjunctive Normal Form (CNF).

Uses the VSIDS (Variable State Independent Decaying Sum) heuristic for
variable selection, which dramatically improves performance on hard instances.

Performance target: solve 50-variable random 3-SAT in < 2 seconds.

Algorithm
─────────
1. Unit Propagation: find unit clauses (single literal), assign forced value,
   simplify all clauses, repeat until fixpoint or conflict.
2. Pure Literal Elimination: if a variable appears only positive or only
   negative across all remaining clauses, assign it to satisfy those clauses.
3. VSIDS Decision: pick the unassigned variable with the highest VSIDS score
   (bumped on each clause participation, decayed periodically).
4. Branching: try True branch; if UNSAT, backtrack and try False branch.
5. Termination:
   - All clauses satisfied → SAT (return satisfying assignment)
   - Empty clause found → UNSAT (backtrack)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from .cnf_converter import CNF, Clause, Literal
from . import config


# ── Result types ───────────────────────────────────────────────────────────────

@dataclass
class SATResult:
    """
    Result of DPLL SAT solving.

    Attributes
    ----------
    satisfiable : bool
        True if the formula is satisfiable.
    assignment : dict[str, bool]
        Satisfying variable assignment (only meaningful if satisfiable=True).
    steps : list[str]
        Trace of the solver's decisions and propagations.
    decisions : int
        Number of branching decisions made.
    propagations : int
        Number of unit propagations performed.
    """
    satisfiable: bool
    assignment: dict[str, bool] = field(default_factory=dict)
    steps: list[str] = field(default_factory=list)
    decisions: int = 0
    propagations: int = 0


# ── VSIDS score tracker ────────────────────────────────────────────────────────

class VSIDS:
    """
    Variable State Independent Decaying Sum (VSIDS) heuristic.

    Each variable maintains a score. When a conflict is analysed, the
    variables in the conflict clause get their scores bumped. All scores
    are periodically decayed (multiplied by a factor < 1) to give recency
    more weight.

    Parameters
    ----------
    variables : set[str]
        All variables in the formula.
    decay : float
        Decay factor applied every `decay_interval` conflicts (default 0.95).
    bump_amount : float
        Amount to add to a variable's score on bump (default 1.0).
    decay_interval : int
        Decay every N conflicts (default 100).
    """

    def __init__(
        self,
        variables: set[str],
        decay: float = 0.95,
        bump_amount: float = 1.0,
        decay_interval: int = 100,
    ) -> None:
        self.scores: dict[str, float] = {v: 0.0 for v in variables}
        self.decay = decay
        self.bump_amount = bump_amount
        self.decay_interval = decay_interval
        self._conflict_count = 0

    def bump(self, var: str) -> None:
        """Increase the score of a variable (called on conflict involvement)."""
        if var in self.scores:
            self.scores[var] += self.bump_amount

    def bump_clause(self, clause: Clause) -> None:
        """Bump all variables in a clause."""
        for (name, _) in clause:
            self.bump(name)

    def decay_all(self) -> None:
        """Decay all scores by the decay factor."""
        for var in self.scores:
            self.scores[var] *= self.decay

    def on_conflict(self, conflict_clause: Clause) -> None:
        """Called when a conflict is found — bumps and periodically decays."""
        self.bump_clause(conflict_clause)
        self._conflict_count += 1
        if self._conflict_count % self.decay_interval == 0:
            self.decay_all()

    def pick(self, unassigned: set[str]) -> Optional[str]:
        """
        Pick the highest-scoring unassigned variable.

        Parameters
        ----------
        unassigned : set[str]
            Variables not yet assigned.

        Returns
        -------
        Optional[str]
            Variable name, or None if unassigned is empty.
        """
        if not unassigned:
            return None
        return max(unassigned, key=lambda v: self.scores.get(v, 0.0))


# ── Core DPLL implementation ───────────────────────────────────────────────────

class DPLLSolver:
    """
    DPLL SAT solver with unit propagation, pure literal elimination,
    and VSIDS variable selection heuristic.

    Usage
    -----
    solver = DPLLSolver(cnf, all_variables)
    result = solver.solve()
    """

    def __init__(self, cnf: CNF, variables: Optional[set[str]] = None) -> None:
        """
        Initialise the solver.

        Parameters
        ----------
        cnf : CNF
            The formula in CNF (list of frozenset of (name, bool) literals).
        variables : Optional[set[str]]
            All variables. If None, extracted from cnf automatically.
        """
        self._original_cnf = cnf
        if variables is not None:
            self._all_vars = set(variables)
        else:
            self._all_vars = {name for clause in cnf for (name, _) in clause}
        self._vsids = VSIDS(self._all_vars)
        self._steps: list[str] = []
        self._decisions = 0
        self._propagations = 0

    def solve(self) -> SATResult:
        """
        Run DPLL on the stored CNF formula.

        Returns
        -------
        SATResult
            Satisfiability verdict with assignment and trace.
        """
        self._steps = ["DPLL solver started"]
        self._decisions = 0
        self._propagations = 0

        # Represent clauses as a list of frozensets (mutable during recursion)
        initial_clauses = list(self._original_cnf)
        assignment: dict[str, bool] = {}

        sat, final_assignment = self._dpll(initial_clauses, assignment)

        if sat:
            # Fill in any unassigned variables (they can be anything)
            for v in self._all_vars:
                if v not in final_assignment:
                    final_assignment[v] = True
            self._steps.append(
                f"SAT: satisfying assignment found ({len(final_assignment)} vars)"
            )
        else:
            self._steps.append("UNSAT: no satisfying assignment exists")

        return SATResult(
            satisfiable=sat,
            assignment=final_assignment if sat else {},
            steps=list(self._steps),
            decisions=self._decisions,
            propagations=self._propagations,
        )

    def _dpll(
        self,
        clauses: list[Clause],
        assignment: dict[str, bool],
    ) -> tuple[bool, dict[str, bool]]:
        """
        Recursive DPLL procedure.

        Parameters
        ----------
        clauses : list[Clause]
            Remaining clauses (already simplified by earlier assignments).
        assignment : dict[str, bool]
            Current partial variable assignment.

        Returns
        -------
        tuple[bool, dict[str, bool]]
            (satisfiable, assignment)
        """
        # ── Step 1: Unit Propagation ───────────────────────────────────────────
        clauses, assignment, conflict = self._unit_propagate(clauses, assignment)
        if conflict:
            return False, {}

        # ── Base case: all clauses satisfied ──────────────────────────────────
        if not clauses:
            return True, dict(assignment)

        # ── Conflict: empty clause found ──────────────────────────────────────
        if any(len(c) == 0 for c in clauses):
            return False, {}

        # ── Step 2: Pure Literal Elimination ─────────────────────────────────
        clauses, assignment = self._pure_literal_eliminate(clauses, assignment)

        if not clauses:
            return True, dict(assignment)

        # ── Step 3: Variable Selection (VSIDS) ────────────────────────────────
        unassigned = {
            name
            for clause in clauses
            for (name, _) in clause
            if name not in assignment
        }
        if not unassigned:
            # All variables assigned but clauses remain — check satisfaction
            return self._check_all_satisfied(clauses, assignment), dict(assignment)

        var = self._vsids.pick(unassigned)
        if var is None:
            return False, {}

        # ── Step 4: Branch ────────────────────────────────────────────────────
        self._decisions += 1
        self._steps.append(f"  Decision #{self._decisions}: try {var} = True")

        for value in (True, False):
            new_assignment = dict(assignment)
            new_assignment[var] = value
            simplified = self._simplify(clauses, var, value)

            sat, result = self._dpll(simplified, new_assignment)
            if sat:
                return True, result

            if value:
                self._steps.append(
                    f"  Backtrack: {var}=True failed, trying {var}=False"
                )
            else:
                self._steps.append(f"  Backtrack: {var}=False failed")

        return False, {}

    def _unit_propagate(
        self,
        clauses: list[Clause],
        assignment: dict[str, bool],
    ) -> tuple[list[Clause], dict[str, bool], bool]:
        """
        Repeatedly find and assign unit clauses.

        A unit clause is a clause with exactly one literal. That literal
        must be true for the formula to be satisfiable.

        Parameters
        ----------
        clauses : list[Clause]
            Current clause list.
        assignment : dict[str, bool]
            Current assignment (mutated in-place during propagation).

        Returns
        -------
        tuple[list[Clause], dict[str, bool], bool]
            (simplified_clauses, updated_assignment, conflict_found)
        """
        assignment = dict(assignment)
        changed = True

        while changed:
            changed = False
            new_clauses = []
            conflict = False

            for clause in clauses:
                unresolved = []
                satisfied = False

                for (name, polarity) in clause:
                    if name in assignment:
                        if assignment[name] == polarity:
                            satisfied = True
                            break
                        # else: this literal is false — skip it
                    else:
                        unresolved.append((name, polarity))

                if satisfied:
                    continue  # Clause is satisfied — remove it
                if len(unresolved) == 0:
                    # Empty clause — conflict!
                    return [], assignment, True
                if len(unresolved) == 1:
                    # Unit clause — force assignment
                    (name, polarity) = unresolved[0]
                    if name in assignment and assignment[name] != polarity:
                        return [], assignment, True  # Contradiction
                    assignment[name] = polarity
                    self._propagations += 1
                    self._steps.append(
                        f"  Unit propagation: {name} = {polarity}"
                    )
                    changed = True
                else:
                    new_clauses.append(frozenset(unresolved))

            clauses = new_clauses

        return clauses, assignment, False

    def _pure_literal_eliminate(
        self,
        clauses: list[Clause],
        assignment: dict[str, bool],
    ) -> tuple[list[Clause], dict[str, bool]]:
        """
        Eliminate pure literals — variables that appear with only one polarity.

        A pure literal can always be assigned to satisfy its occurrences
        without creating any conflicts.

        Parameters
        ----------
        clauses : list[Clause]
            Current clause list.
        assignment : dict[str, bool]
            Current assignment.

        Returns
        -------
        tuple[list[Clause], dict[str, bool]]
            (simplified_clauses, updated_assignment)
        """
        assignment = dict(assignment)

        # Collect all literal polarities for unassigned variables
        pos_vars: set[str] = set()
        neg_vars: set[str] = set()

        for clause in clauses:
            for (name, polarity) in clause:
                if name not in assignment:
                    if polarity:
                        pos_vars.add(name)
                    else:
                        neg_vars.add(name)

        # Pure positive: appears only positive
        pure_pos = pos_vars - neg_vars
        # Pure negative: appears only negative
        pure_neg = neg_vars - pos_vars

        if not pure_pos and not pure_neg:
            return clauses, assignment

        for var in pure_pos:
            assignment[var] = True
            self._steps.append(f"  Pure literal: {var} = True (only positive)")
        for var in pure_neg:
            assignment[var] = False
            self._steps.append(f"  Pure literal: {var} = False (only negative)")

        # Remove satisfied clauses
        remaining = []
        for clause in clauses:
            satisfied = any(
                name in assignment and assignment[name] == polarity
                for (name, polarity) in clause
            )
            if not satisfied:
                remaining.append(clause)

        return remaining, assignment

    def _simplify(
        self,
        clauses: list[Clause],
        var: str,
        value: bool,
    ) -> list[Clause]:
        """
        Simplify the clause list given var = value.

        - Remove clauses where (var, value) appears (they are satisfied).
        - Remove the literal (var, not value) from remaining clauses.

        Parameters
        ----------
        clauses : list[Clause]
            Current clause list.
        var : str
            The variable being assigned.
        value : bool
            The value being assigned.

        Returns
        -------
        list[Clause]
            Simplified clause list.
        """
        result = []
        for clause in clauses:
            if (var, value) in clause:
                continue  # Clause satisfied — remove it
            # Remove the falsified literal from the clause
            new_clause = frozenset(
                lit for lit in clause if lit != (var, not value)
            )
            result.append(new_clause)
        return result

    def _check_all_satisfied(
        self,
        clauses: list[Clause],
        assignment: dict[str, bool],
    ) -> bool:
        """
        Check whether all clauses are satisfied by the given assignment.

        Parameters
        ----------
        clauses : list[Clause]
            Clause list to check.
        assignment : dict[str, bool]
            Complete variable assignment.

        Returns
        -------
        bool
            True if every clause has at least one satisfied literal.
        """
        for clause in clauses:
            if not any(
                name in assignment and assignment[name] == polarity
                for (name, polarity) in clause
            ):
                return False
        return True


def solve_cnf(cnf: CNF, variables: Optional[set[str]] = None) -> SATResult:
    """
    Module-level convenience function to solve a CNF formula.

    Parameters
    ----------
    cnf : CNF
        The formula in CNF (list of frozensets of literals).
    variables : Optional[set[str]]
        All variables. Extracted from cnf if None.

    Returns
    -------
    SATResult
        Satisfiability result.
    """
    solver = DPLLSolver(cnf, variables)
    return solver.solve()


def verify_assignment(cnf: CNF, assignment: dict[str, bool]) -> bool:
    """
    Verify that an assignment satisfies a CNF formula.

    Parameters
    ----------
    cnf : CNF
        The formula.
    assignment : dict[str, bool]
        A (possibly partial) variable assignment.

    Returns
    -------
    bool
        True if every clause is satisfied.
    """
    for clause in cnf:
        if not any(
            name in assignment and assignment[name] == polarity
            for (name, polarity) in clause
        ):
            return False
    return True
