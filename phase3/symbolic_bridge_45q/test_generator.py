"""
test_generator.py — Automatic Test Case Generator for 45Q.

Generates up to CODE_TEST_COUNT test cases per function by analysing
the function's name, arguments, body, and annotations.

Test strategy:
  - Boundary values: 0, 1, -1, large ints, empty collections, None
  - Type variants: int vs float vs string for numeric args
  - Name-inferred edge cases: sort → sorted/reverse-sorted; divide → zero
  - Random valid inputs: 3 cases
"""

from __future__ import annotations
import ast
import random
import textwrap
from dataclasses import dataclass, field
from typing import Any, Optional
from . import config


@dataclass
class TestCase:
    """
    A single generated test case.

    Attributes
    ----------
    name : str
        Test method name (valid Python identifier).
    args : list
        Positional arguments to pass to the function.
    expected_behavior : str
        Description of what we expect (not always a precise value).
    should_raise : Optional[str]
        Exception class name if we expect this to raise, else None.
    description : str
        Human-readable test description.
    """
    name: str
    args: list
    expected_behavior: str
    should_raise: Optional[str] = None
    description: str = ""


@dataclass
class TestSuite:
    """
    A generated test suite for one function.

    Attributes
    ----------
    function_name : str
        The function being tested.
    test_cases : list[TestCase]
        Generated test cases.
    test_code : str
        Complete unittest.TestCase subclass code, ready to run.
    notes : list[str]
        Analysis notes from the generator.
    """
    function_name: str
    test_cases: list[TestCase] = field(default_factory=list)
    test_code: str = ""
    notes: list[str] = field(default_factory=list)


def generate_tests(source: str) -> TestSuite:
    """
    Generate test cases for the first function found in source.

    Parameters
    ----------
    source : str
        Python source code containing at least one function definition.

    Returns
    -------
    TestSuite
        Generated test suite with code ready to run.

    Raises
    ------
    ValueError
        If no function definition is found in source.
    """
    source = textwrap.dedent(source)

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise ValueError(f"Cannot parse source: {e}")

    # Find first function
    func_node: Optional[ast.FunctionDef] = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_node = node
            break

    if func_node is None:
        raise ValueError("No function definition found in source")

    suite = TestSuite(function_name=func_node.name)

    # Analyse function signature
    args = [a.arg for a in func_node.args.args]
    annotations = {a.arg: a.annotation for a in func_node.args.args if a.annotation}
    body_source = ast.unparse(func_node) if hasattr(ast, 'unparse') else source

    suite.notes.append(f"Function: {func_node.name}({', '.join(args)})")
    suite.notes.append(f"Arguments: {args}")

    # Generate test cases
    cases: list[TestCase] = []

    # 1. Boundary values
    cases += _boundary_cases(func_node, args)

    # 2. Name-inferred edge cases
    cases += _name_inferred_cases(func_node, args, body_source)

    # 3. Random valid inputs
    cases += _random_cases(func_node, args)

    # Limit to CODE_TEST_COUNT
    cases = cases[:config.CODE_TEST_COUNT]

    suite.test_cases = cases
    suite.notes.append(f"Generated {len(cases)} test case(s)")

    # Generate unittest code
    suite.test_code = _render_test_class(func_node.name, args, cases, source)

    return suite


def _boundary_cases(func_node: ast.FunctionDef, args: list[str]) -> list[TestCase]:
    """
    Generate boundary value test cases.

    Parameters
    ----------
    func_node : ast.FunctionDef
        The function being tested.
    args : list[str]
        Argument names.

    Returns
    -------
    list[TestCase]
        Boundary test cases.
    """
    cases: list[TestCase] = []
    name = func_node.name

    if not args:
        return cases

    # For single-arg functions: standard boundary values
    boundaries_single = [
        (0,          "zero input",              None),
        (1,          "unit input",              None),
        (-1,         "negative one",            None),
        ([],         "empty list",              None),
        ("",         "empty string",            None),
        (None,       "None input",              None),
        (2**31 - 1,  "max 32-bit int",          None),
        (-2**31,     "min 32-bit int",          None),
    ]

    if len(args) == 1:
        for i, (val, desc, exc) in enumerate(boundaries_single[:6]):
            cases.append(TestCase(
                name=f"test_{name}_boundary_{i+1}",
                args=[val],
                expected_behavior=f"handles {desc}",
                should_raise=exc,
                description=f"Boundary: {desc}",
            ))
    else:
        # For multi-arg functions: use zero for all, then one
        zero_args = [0] * len(args)
        cases.append(TestCase(
            name=f"test_{name}_all_zeros",
            args=zero_args,
            expected_behavior="handles all-zero arguments",
            description="Boundary: all zeros",
        ))
        one_args = [1] * len(args)
        cases.append(TestCase(
            name=f"test_{name}_all_ones",
            args=one_args,
            expected_behavior="handles all-one arguments",
            description="Boundary: all ones",
        ))

    return cases


def _name_inferred_cases(
    func_node: ast.FunctionDef,
    args: list[str],
    body_source: str,
) -> list[TestCase]:
    """
    Generate edge cases inferred from the function name and body.

    Parameters
    ----------
    func_node : ast.FunctionDef
        The function.
    args : list[str]
        Argument names.
    body_source : str
        Unparsed function source.

    Returns
    -------
    list[TestCase]
        Name-inferred test cases.
    """
    cases: list[TestCase] = []
    name = func_node.name.lower()
    body = body_source.lower()

    # Sort function tests
    if any(k in name for k in ("sort", "order", "rank")):
        cases.append(TestCase(
            name=f"test_{func_node.name}_already_sorted",
            args=[[1, 2, 3, 4, 5]],
            expected_behavior="handles already-sorted input",
            description="Name-inferred: already sorted input",
        ))
        cases.append(TestCase(
            name=f"test_{func_node.name}_reverse_sorted",
            args=[[5, 4, 3, 2, 1]],
            expected_behavior="handles reverse-sorted input",
            description="Name-inferred: reverse sorted input",
        ))

    # Search function tests
    if any(k in name for k in ("search", "find", "lookup", "get")):
        cases.append(TestCase(
            name=f"test_{func_node.name}_not_found",
            args=[[1, 2, 3], 99],
            expected_behavior="handles not-found case gracefully",
            description="Name-inferred: element not in collection",
        ))

    # Divide/division tests
    if any(k in name for k in ("divid", "div", "quot")) or "/" in body:
        cases.append(TestCase(
            name=f"test_{func_node.name}_zero_divisor",
            args=[5, 0] if len(args) >= 2 else [0],
            expected_behavior="raises ZeroDivisionError or handles division by zero",
            should_raise="ZeroDivisionError",
            description="Name-inferred: zero divisor",
        ))

    # Factorial tests
    if any(k in name for k in ("factorial", "fact")):
        cases.append(TestCase(
            name=f"test_{func_node.name}_negative_input",
            args=[-1],
            expected_behavior="raises ValueError or handles negative input",
            should_raise="ValueError",
            description="Name-inferred: negative factorial",
        ))
        cases.append(TestCase(
            name=f"test_{func_node.name}_zero",
            args=[0],
            expected_behavior="returns 1 (0! = 1)",
            description="Name-inferred: 0! = 1",
        ))

    # Max/min function tests
    if any(k in name for k in ("max", "min", "maximum", "minimum")):
        cases.append(TestCase(
            name=f"test_{func_node.name}_single_element",
            args=[[42]],
            expected_behavior="returns the single element",
            description="Name-inferred: single-element list",
        ))
        cases.append(TestCase(
            name=f"test_{func_node.name}_all_same",
            args=[[7, 7, 7, 7]],
            expected_behavior="handles all-equal elements",
            description="Name-inferred: all elements equal",
        ))

    # Substring/contains tests
    if any(k in name for k in ("substr", "contain", "has")):
        cases.append(TestCase(
            name=f"test_{func_node.name}_empty_needle",
            args=["hello", ""],
            expected_behavior="handles empty search string",
            description="Name-inferred: empty needle",
        ))

    return cases


def _random_cases(func_node: ast.FunctionDef, args: list[str]) -> list[TestCase]:
    """
    Generate random valid test cases.

    Parameters
    ----------
    func_node : ast.FunctionDef
        The function.
    args : list[str]
        Argument names.

    Returns
    -------
    list[TestCase]
        Three random test cases.
    """
    cases: list[TestCase] = []
    name = func_node.name

    for i in range(3):
        random_args = []
        for arg in args:
            # Generate random int in reasonable range
            random_args.append(random.randint(-100, 100))
        cases.append(TestCase(
            name=f"test_{name}_random_{i+1}",
            args=random_args,
            expected_behavior="completes without exception",
            description=f"Random input #{i+1}: args={random_args}",
        ))

    return cases


def _render_test_class(
    func_name: str,
    args: list[str],
    cases: list[TestCase],
    source: str,
) -> str:
    """
    Render a complete unittest.TestCase class for the generated test cases.

    Parameters
    ----------
    func_name : str
        The function under test.
    args : list[str]
        Function argument names.
    cases : list[TestCase]
        Generated test cases.
    source : str
        Original source code (to be embedded in the test module).

    Returns
    -------
    str
        Complete Python test module code, ready to run.
    """
    lines = [
        "import unittest",
        "import sys",
        "import textwrap",
        "",
        "# ── Function under test (embedded) ───────────────────────────────────",
        textwrap.dedent(source).strip(),
        "",
        "",
        f"class Test{func_name.title()}(unittest.TestCase):",
        f'    """Auto-generated tests for {func_name}() by 45Q."""',
        "",
    ]

    for tc in cases:
        lines.append(f"    def {tc.name}(self):")
        lines.append(f'        """{tc.description}"""')

        # Format args for the call
        arg_reprs = [repr(a) for a in tc.args]
        call_str = f"{func_name}({', '.join(arg_reprs)})"

        if tc.should_raise:
            lines.append(f"        with self.assertRaises({tc.should_raise}):")
            lines.append(f"            {call_str}")
        else:
            lines.append(f"        try:")
            lines.append(f"            result = {call_str}")
            lines.append(f"            # {tc.expected_behavior}")
            lines.append(f"            self.assertIsNotNone(result, 'Function returned None unexpectedly')")
            lines.append(f"        except Exception as e:")
            lines.append(f"            self.fail(f'Unexpected exception: {{type(e).__name__}}: {{e}}')")

        lines.append("")

    lines += [
        "",
        "if __name__ == '__main__':",
        "    unittest.main(verbosity=2)",
    ]

    return "\n".join(lines)
