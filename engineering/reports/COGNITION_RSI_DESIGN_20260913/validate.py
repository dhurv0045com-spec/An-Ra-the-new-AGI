"""Structural checks for the combined M00-M24 design. No learner execution."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
EXT = ROOT / "engineering/cognition_rsi_20260913"
extension = json.loads((EXT / "program.json").read_text(encoding="utf-8"))
base_path = (EXT / extension["extends"]).resolve()
base = json.loads(base_path.read_text(encoding="utf-8"))
packages = base["packages"] + extension["packages"]
ids = [package["id"] for package in packages]
errors = []
if ids != [f"M{index:02d}" for index in range(25)]:
    errors.append("Combined package inventory differs from M00-M24")
graph = {package["id"]: package["depends_on"] for package in packages}
active, visited = set(), set()


def visit(node):
    if node in active:
        errors.append(f"Cycle: {node}")
        return
    if node in visited:
        return
    if node not in graph:
        errors.append(f"Unknown dependency: {node}")
        return
    active.add(node)
    for dependency in graph[node]:
        visit(dependency)
    active.remove(node)
    visited.add(node)


visit("M24")
if visited != set(ids):
    errors.append("M24 does not reach all combined prerequisites")
files = []
for directory, packet in [(base_path.parent, base), (EXT, extension)]:
    execution = (directory / "EXECUTION.md").read_text(encoding="utf-8")
    headings = re.findall(r"^## (M\d\d) —", execution, re.MULTILINE)
    if headings != [item["id"] for item in packet["packages"]]:
        errors.append(f"Package headings mismatch in {directory.name}")
    for package in packet["packages"]:
        if package["id"] == "M00":
            continue
        section = execution.split(f"## {package['id']} —", 1)[1].split("\n## ", 1)[0]
        match = re.search(r"\*\*Depends on:\*\* ([^\n]+)", section)
        dependencies = re.findall(r"M\d\d", match.group(1)) if match else []
        if dependencies != package["depends_on"]:
            errors.append(f"Text/manifest dependencies differ: {package['id']}")
    files.extend(directory / name for name in packet["primary_design_files"])

entry = [ROOT / name for name in ["AGENTS.md", "README.md", "BRAMASTRA_PAPER.md",
         "engineering/README.md", "engineering/STATUS.md",
         "engineering/reports/B2_2_CHIEF_20260913/REVIEW.md"]]
checked_links = 0
for path in files + entry:
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
        if "://" in target or target.startswith("#"):
            continue
        checked_links += 1
        if not (path.parent / target.split("#", 1)[0]).exists():
            errors.append(f"Broken link: {path.relative_to(ROOT)} -> {target}")

hashes = {}
word_counts = {"base": 0, "cognition_rsi_extension": 0}
for path in files + [base_path, EXT / "program.json"]:
    content = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if path.suffix == ".md":
        word_counts["cognition_rsi_extension" if path.parent == EXT else "base"] += len(content.split())
ledger = json.loads((ROOT / base["resource_ledger"]).read_text(encoding="utf-8"))
if extension["optimizer_updates_authorized"] != 0 or extension["accelerator_experiments_authorized"]:
    errors.append("Current zero-update/no-accelerator override is inconsistent")
if ledger["cpu_optimizer_updates"] != extension["resource_snapshot"]["cpu_optimizer_updates"]:
    errors.append("Live resource ledger differs from extension snapshot; review before dispatch")
result = {"kind": "combined_design_structure_only", "source_review": "acfa249",
          "combined_packages": len(packages), "local_links_checked": checked_links,
          "errors": errors, "word_counts": word_counts,
          "canonical_utf8_lf_sha256": hashes,
          "validation_optimizer_updates": 0, "validation_model_instantiations": 0,
          "live_cpu_optimizer_updates": ledger["cpu_optimizer_updates"],
          "live_cpu_learned_smoke_seconds": ledger["cpu_learned_smoke_seconds"]}
print(json.dumps(result, indent=2))
raise SystemExit(bool(errors))
