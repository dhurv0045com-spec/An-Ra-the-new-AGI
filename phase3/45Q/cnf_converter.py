"""
cnf_converter.py — Tseitin CNF Transformation (from scratch).

Converts any propositional formula to Conjunctive Normal Form (CNF) using
the Tseitin transformation, which avoids the exponential blowup of naive
distribution-based conversion.

The Tseitin method
──────────────────
For each sub-formula φᵢ, introduce a fresh variable tᵢ and add clauses
encoding tᵢ ↔ φᵢ. The top-level formula is then just: (t_root).

This ensures the CNF is linearly larger than the original formula.
The resulting CNF is equisatisfiable (not equivalent) to the original.

Formula AST
───────────
We represent formulas as plain Python objects:

  Var(name)           — atomic variable
  Not(sub)            — ¬sub
  And(left, right)    — left ∧ right
  Or(left, right)     — left ∨ right
  Implies(left, right)— left → right  (equivalent to ¬left ∨ right)
  Iff(left, right)    — left ↔ right  (left→right ∧ right→left)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Union


# ── Formula AST nodes ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Var:
    """Propositional variable. name is its string identifier."""
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Not:
    """Negation of a formula."""
    sub: "Formula"

    def __str__(self) -> str:
        return f"¬{self.sub}"


@dataclass(frozen=True)
class And:
    """Conjunction of two formulas."""
    left: "Formula"
    right: "Formula"

    def __str__(self) -> str:
        return f"({self.left} ∧ {self.right})"


@dataclass(frozen=True)
class Or:
    """Disjunction of two formulas."""
    left: "Formula"
    right: "Formula"

    def __str__(self) -> str:
        return f"({self.left} ∨ {self.right})"


@dataclass(frozen=True)
class Implies:
    """Material implication: left → right."""
    left: "Formula"
    right: "Formula"

    def __str__(self) -> str:
        return f"({self.left} → {self.right})"


@dataclass(frozen=True)
class Iff:
    """Biconditional: left ↔ right."""
    left: "Formula"
    right: "Formula"

    def __str__(self) -> str:
        return f"({self.left} ↔ {self.right})"


Formula = Union[Var, Not, And, Or, Implies, Iff]

# A clause is a frozenset of literals. A literal is (variable_name, is_positive).
Literal = tuple[str, bool]       # (name, True) = positive, (name, False) = negated
Clause  = frozenset[Literal]
CNF     = list[Clause]


@dataclass
class TseitinResult:
    """
    Result of the Tseitin transformation.

    Attributes
    ----------
    cnf : CNF
        List of clauses (each clause is a frozenset of literals).
    tseitin_vars : set[str]
        Names of fresh variables introduced by the transformation.
    original_vars : set[str]
        Names of variables from the original formula.
    root_var : str
        The Tseitin variable representing the entire formula.
    num_clauses : int
        Total number of clauses in the CNF.
    steps : list[str]
        Description of each sub-formula and the clauses it generated.
    """
    cnf: CNF
    tseitin_vars: set[str]
    original_vars: set[str]
    root_var: str
    num_clauses: int
    steps: list[str]


class TseitinConverter:
    """
    Converts a propositional formula to CNF via the Tseitin transformation.

    Usage
    -----
    converter = TseitinConverter()
    result = converter.to_cnf(formula)
    """

    def __init__(self) -> None:
        self._counter: int = 0
        self._tseitin_vars: set[str] = set()
        self._original_vars: set[str] = set()
        self._clauses: list[Clause] = []
        self._steps: list[str] = []

    def _fresh(self) -> str:
        """
        Generate a fresh Tseitin variable name.

        Returns
        -------
        str
            A unique variable name of the form t_1, t_2, etc.
        """
        self._counter += 1
        name = f"t_{self._counter}"
        self._tseitin_vars.add(name)
        return name

    def _lit(self, name: str, positive: bool = True) -> Literal:
        """Create a literal tuple."""
        return (name, positive)

    def _add_clause(self, literals: list[Literal], reason: str) -> None:
        """
        Add a clause to the CNF and log the step.

        Parameters
        ----------
        literals : list[Literal]
            The literals forming this clause.
        reason : str
            Human-readable explanation of why this clause was added.
        """
        clause = frozenset(literals)
        self._clauses.append(clause)
        # Format clause for display
        parts = []
        for (name, pos) in sorted(literals):
            parts.append(name if pos else f"¬{name}")
        self._steps.append(f"  Clause ({' ∨ '.join(parts)})  [{reason}]")

    def _encode(self, formula: Formula) -> str:
        """
        Encode a sub-formula and return the Tseitin variable representing it.

        For each node type, adds the clauses encoding (t ↔ φ):
        - The clauses are split into implications t → φ and φ → t.

        Parameters
        ----------
        formula : Formula
            The sub-formula to encode.

        Returns
        -------
        str
            The name of the Tseitin variable t such that (t ↔ formula).
        """
        if isinstance(formula, Var):
            # Atoms: no fresh variable needed — atom is its own Tseitin var
            self._original_vars.add(formula.name)
            return formula.name

        t = self._fresh()
        self._steps.append(f"Encoding {t} ↔ {formula}")

        if isinstance(formula, Not):
            # t ↔ ¬s
            # Encode sub
            s = self._encode(formula.sub)
            # Clauses:
            #   t → ¬s   ≡   ¬t ∨ ¬s
            #   ¬s → t   ≡   s ∨ t
            self._add_clause([self._lit(t, False), self._lit(s, False)],
                             f"{t} → ¬{s}")
            self._add_clause([self._lit(s, True),  self._lit(t, True)],
                             f"¬{s} → {t}")

        elif isinstance(formula, And):
            # t ↔ (l ∧ r)
            l = self._encode(formula.left)
            r = self._encode(formula.right)
            # Clauses:
            #   t → l   ≡   ¬t ∨ l
            #   t → r   ≡   ¬t ∨ r
            #   (l ∧ r) → t   ≡   ¬l ∨ ¬r ∨ t
            self._add_clause([self._lit(t, False), self._lit(l, True)],
                             f"{t} → {l}")
            self._add_clause([self._lit(t, False), self._lit(r, True)],
                             f"{t} → {r}")
            self._add_clause([self._lit(l, False), self._lit(r, False), self._lit(t, True)],
                             f"{l} ∧ {r} → {t}")

        elif isinstance(formula, Or):
            # t ↔ (l ∨ r)
            l = self._encode(formula.left)
            r = self._encode(formula.right)
            # Clauses:
            #   t → (l ∨ r)   ≡   ¬t ∨ l ∨ r
            #   l → t         ≡   ¬l ∨ t
            #   r → t         ≡   ¬r ∨ t
            self._add_clause([self._lit(t, False), self._lit(l, True), self._lit(r, True)],
                             f"{t} → {l} ∨ {r}")
            self._add_clause([self._lit(l, False), self._lit(t, True)],
                             f"{l} → {t}")
            self._add_clause([self._lit(r, False), self._lit(t, True)],
                             f"{r} → {t}")

        elif isinstance(formula, Implies):
            # t ↔ (l → r)  which is t ↔ (¬l ∨ r)
            # Rewrite as Or(Not(formula.left), formula.right) and encode
            l = self._encode(formula.left)
            r = self._encode(formula.right)
            # Clauses for t ↔ (¬l ∨ r):
            #   t → ¬l ∨ r   ≡   ¬t ∨ ¬l ∨ r
            #   ¬l → t        ≡   l ∨ t
            #   r → t         ≡   ¬r ∨ t
            self._add_clause(
                [self._lit(t, False), self._lit(l, False), self._lit(r, True)],
                f"{t} → ¬{l} ∨ {r}")
            self._add_clause([self._lit(l, True),  self._lit(t, True)],
                             f"¬{l} → {t}")
            self._add_clause([self._lit(r, False), self._lit(t, True)],
                             f"{r} → {t}")

        elif isinstance(formula, Iff):
            # t ↔ (l ↔ r)
            # (l ↔ r)  ≡  (l → r) ∧ (r → l)  ≡  (¬l ∨ r) ∧ (¬r ∨ l)
            l = self._encode(formula.left)
            r = self._encode(formula.right)
            # Clauses for t ↔ ((¬l ∨ r) ∧ (¬r ∨ l)):
            #   t → ¬l ∨ r     ≡   ¬t ∨ ¬l ∨ r
            #   t → ¬r ∨ l     ≡   ¬t ∨ ¬r ∨ l
            #   (l = r) → t   split into:
            #     ¬l ∨ ¬r ∨ t   and   l ∨ r ∨ t
            self._add_clause(
                [self._lit(t, False), self._lit(l, False), self._lit(r, True)],
                f"{t} → (¬{l} ∨ {r})")
            self._add_clause(
                [self._lit(t, False), self._lit(r, False), self._lit(l, True)],
                f"{t} → (¬{r} ∨ {l})")
            self._add_clause(
                [self._lit(l, False), self._lit(r, False), self._lit(t, True)],
                f"¬{l}∧¬{r} → {t}")
            self._add_clause(
                [self._lit(l, True),  self._lit(r, True),  self._lit(t, True)],
                f"{l}∧{r} → {t}")

        else:
            raise TypeError(f"Unknown formula node type: {type(formula)}")

        return t

    def to_cnf(self, formula: Formula) -> TseitinResult:
        """
        Convert formula to CNF via the Tseitin transformation.

        Parameters
        ----------
        formula : Formula
            The propositional formula to convert.

        Returns
        -------
        TseitinResult
            The CNF along with variable sets and diagnostic information.

        Notes
        -----
        The result is equisatisfiable with the original formula:
          original is SAT  ↔  CNF is SAT
        The CNF is always linear in the size of the formula (O(|formula|)).
        """
        # Reset state for a fresh conversion
        self._counter = 0
        self._tseitin_vars = set()
        self._original_vars = set()
        self._clauses = []
        self._steps = ["Tseitin CNF Transformation:"]

        # Encode the formula; get the root Tseitin variable
        root = self._encode(formula)

        # Assert the root variable is true (the whole formula must hold)
        self._add_clause([self._lit(root, True)], "root assertion (formula is true)")

        return TseitinResult(
            cnf=list(self._clauses),
            tseitin_vars=set(self._tseitin_vars),
            original_vars=set(self._original_vars),
            root_var=root,
            num_clauses=len(self._clauses),
            steps=list(self._steps),
        )


def to_cnf(formula: Formula) -> TseitinResult:
    """
    Module-level convenience function for Tseitin conversion.

    Parameters
    ----------
    formula : Formula
        Any propositional formula.

    Returns
    -------
    TseitinResult
        The CNF result.
    """
    return TseitinConverter().to_cnf(formula)


# ── Formula parser ─────────────────────────────────────────────────────────────

def parse_formula(text: str) -> Formula:
    """
    Parse a propositional formula string into a Formula AST.

    Supported syntax:
      Variables  : single uppercase letters (A, B, C, ...) or names
      NOT        : ¬X, ~X, NOT X, not X
      AND        : A ∧ B, A AND B, A & B
      OR         : A ∨ B, A OR B, A | B
      IMPLIES    : A → B, A -> B, A IMPLIES B, IF A THEN B
      IFF        : A ↔ B, A <-> B, A IFF B
      Parentheses: (A ∧ B) ∨ C

    Precedence (tightest to loosest):
      NOT > AND > OR > IMPLIES > IFF

    Parameters
    ----------
    text : str
        The formula string.

    Returns
    -------
    Formula
        The parsed formula AST.

    Raises
    ------
    ValueError
        If the formula cannot be parsed.
    """
    tokens = _tokenise(text)
    parser = _Parser(tokens)
    formula = parser.parse_iff()
    if parser.pos < len(parser.tokens):
        remaining = parser.tokens[parser.pos:]
        raise ValueError(f"Unexpected tokens after formula: {remaining}")
    return formula


def _tokenise(text: str) -> list[str]:
    """
    Tokenise a formula string into a list of token strings.

    Parameters
    ----------
    text : str
        Raw formula text.

    Returns
    -------
    list[str]
        List of token strings (operators, parentheses, variable names).
    """
    import re
    # Normalise Unicode operators to ASCII equivalents
    text = (text
            .replace("¬", "~")
            .replace("∧", "&")
            .replace("∨", "|")
            .replace("→", "->")
            .replace("↔", "<->")
            .replace("⊕", "XOR"))

    token_pattern = re.compile(
        r'<->|->|<=>'
        r'|[()~&|]'
        r'|\b(?:NOT|AND|OR|IFF|XOR|IF|THEN|IMPLIES)\b'
        r'|[A-Za-z][A-Za-z0-9_]*',
        re.IGNORECASE,
    )
    return token_pattern.findall(text)


class _Parser:
    """
    Recursive descent parser for propositional formulas.
    Precedence: IFF < IMPLIES < OR < AND < NOT < atom.
    """

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, expected: str | None = None) -> str:
        token = self.tokens[self.pos]
        if expected and token.upper() != expected.upper():
            raise ValueError(f"Expected {expected!r}, got {token!r}")
        self.pos += 1
        return token

    def parse_iff(self) -> Formula:
        left = self.parse_implies()
        while self.peek() and self.peek().upper() in ("<->", "IFF", "<=>"):
            self.pos += 1
            right = self.parse_implies()
            left = Iff(left, right)
        return left

    def parse_implies(self) -> Formula:
        left = self.parse_or()
        while self.peek() and self.peek().upper() in ("->", "IMPLIES"):
            self.pos += 1
            right = self.parse_or()
            left = Implies(left, right)
        # Handle IF...THEN
        if self.peek() and self.peek().upper() == "THEN":
            self.pos += 1
            right = self.parse_or()
            left = Implies(left, right)
        return left

    def parse_or(self) -> Formula:
        left = self.parse_and()
        while self.peek() and self.peek().upper() in ("|", "OR"):
            self.pos += 1
            right = self.parse_and()
            left = Or(left, right)
        return left

    def parse_and(self) -> Formula:
        left = self.parse_not()
        while self.peek() and self.peek().upper() in ("&", "AND"):
            self.pos += 1
            right = self.parse_not()
            left = And(left, right)
        return left

    def parse_not(self) -> Formula:
        if self.peek() and self.peek().upper() in ("~", "NOT"):
            self.pos += 1
            sub = self.parse_not()  # Right-associative
            return Not(sub)
        return self.parse_atom()

    def parse_atom(self) -> Formula:
        tok = self.peek()
        if tok is None:
            raise ValueError("Unexpected end of formula")
        if tok == "(":
            self.pos += 1
            formula = self.parse_iff()
            if self.peek() != ")":
                raise ValueError("Missing closing parenthesis")
            self.pos += 1
            return formula
        if tok.upper() == "IF":
            self.pos += 1
            return self.parse_implies()
        # Variable
        if tok.replace("_", "").isalnum():
            self.pos += 1
            return Var(tok)
        raise ValueError(f"Unexpected token: {tok!r}")
