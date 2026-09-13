"""Validate this chief design packet, not learner behavior. No model imports."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[3]
PACKET = ROOT / "engineering/master_program_20260913"
manifest = json.loads((PACKET / "program.json").read_text(encoding="utf-8"))
errors = []
packages = manifest["packages"]
ids = [package["id"] for package in packages]
expected = [f"M{index:02d}" for index in range(19)]
if ids != expected:
    errors.append("Package IDs are not exactly M00 through M18 in order")
graph = {package["id"]: package["depends_on"] for package in packages}
visiting, visited = set(), set()


def visit(node):
    if node in visiting:
        errors.append(f"Dependency cycle at {node}")
        return
    if node in visited:
        return
    if node not in graph:
        errors.append(f"Unknown dependency {node}")
        return
    visiting.add(node)
    for prerequisite in graph[node]:
        visit(prerequisite)
    visiting.remove(node)
    visited.add(node)


visit("M18")
if visited != set(ids):
    errors.append("M18 does not depend transitively on every package")
execution = (PACKET / "EXECUTION.md").read_text(encoding="utf-8")
headings = re.findall(r"^## (M\d\d) —", execution, re.MULTILINE)
if headings != expected:
    errors.append("Execution headings differ from manifest packages")
for package in packages:
    if package["id"] == "M00":
        continue
    section = execution.split(f"## {package['id']} —", 1)[1].split("\n## ", 1)[0]
    dependency_line = re.search(r"\*\*Depends on:\*\* ([^\n]+)", section)
    actual = re.findall(r"M\d\d", dependency_line.group(1)) if dependency_line else []
    if actual != package["depends_on"]:
        errors.append(f"Dependency text differs for {package['id']}")

documents = [PACKET / name for name in manifest["primary_design_files"]]
entry_names = ["AGENTS.md", "README.md", "BRAMASTRA_PAPER.md", "engineering/README.md",
               "engineering/STATUS.md", "engineering/SYSTEM_ARCHITECTURE.md",
               "engineering/LEARNING_ALGORITHMS.md", "engineering/DATA_CONTRACTS.md",
               "engineering/EXECUTION_PLAN.md", "engineering/DECISIONS.md",
               "engineering/phase_b22_20260913/README.md",
               "engineering/phase_b22_20260913/AGENT_PROMPT.md"]
links_checked = 0
for path in documents + [ROOT / name for name in entry_names]:
    content = path.read_text(encoding="utf-8")
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", content):
        if "://" in target or target.startswith("#"):
            continue
        relative = target.split("#", 1)[0]
        links_checked += 1
        if not (path.parent / relative).exists():
            errors.append(f"Missing local link: {path.relative_to(ROOT)} -> {target}")

files = {}
for path in documents + [PACKET / "program.json"]:
    content = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    files[str(path.relative_to(ROOT)).replace("\\", "/")] = {
        "canonical_utf8_lf_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "whitespace_delimited_words": len(content.split()) if path.suffix == ".md" else None,
    }
result = {"check_kind": "design_structure_only", "packages": len(packages),
          "local_links_checked": links_checked, "errors": errors,
          "optimizer_updates": 0, "model_instantiations": 0,
          "files": files,
          "design_markdown_words": sum(item["whitespace_delimited_words"] or 0 for item in files.values())}
print(json.dumps(result, indent=2))
raise SystemExit(bool(errors))
