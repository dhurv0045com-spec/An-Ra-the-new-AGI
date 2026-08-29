"""Static import-boundary enforcement for independent research/control planes."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


FORBIDDEN = {
    "v5_contracts": {"e0_cognition", "e1_tokenizer"},
    "e0_cognition": {"e1_tokenizer"},
    "e1_tokenizer": {"e0_cognition"},
    "v5_model": {"v5_training", "v5_evaluation", "v5_promotion", "connector"},
    "v5_training": {"e0_cognition", "v5_evaluation", "v5_promotion", "connector"},
    "v5_evaluation": {"v5_training", "v5_data", "v5_promotion", "connector"},
    "v5_promotion": {"v5_training", "v5_data", "connector"},
}


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.add(node.module.split(".", 1)[0])
    return imports


def scan_repository(repository: Path) -> list[str]:
    violations: list[str] = []
    for package, forbidden in FORBIDDEN.items():
        package_root = repository / package
        if not package_root.exists():
            continue
        for path in sorted(package_root.rglob("*.py")):
            collisions = _top_level_imports(path) & forbidden
            for imported in sorted(collisions):
                violations.append(f"{path.relative_to(repository)} imports forbidden plane {imported}")
    return violations


def main() -> int:
    repository = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    violations = scan_repository(repository)
    if violations:
        print("\n".join(violations), file=sys.stderr)
        return 1
    print("import boundaries: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
