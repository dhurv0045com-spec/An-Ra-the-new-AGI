"""Validate K8 design arithmetic/links. Does not launch experiments."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
campaign = json.loads((HERE / "campaign.json").read_text(encoding="utf-8"))
errors = []
phases = campaign["phases"]
if [phase["id"] for phase in phases] != [f"E{i}" for i in range(7)]:
    errors.append("Phase inventory must be E0-E6")
previous = 0
for phase in phases:
    if phase["start_minute"] != previous:
        errors.append(f"Schedule discontinuity: {phase['id']}")
    if phase["end_minute"] - phase["start_minute"] != phase["max_wall_minutes"]:
        errors.append(f"Duration mismatch: {phase['id']}")
    if phase["max_provisioned_gpu_minutes"] != 2 * phase["max_wall_minutes"]:
        errors.append(f"Two-GPU accounting mismatch: {phase['id']}")
    previous = phase["end_minute"]
wall = sum(phase["max_wall_minutes"] for phase in phases)
gpu = sum(phase["max_provisioned_gpu_minutes"] for phase in phases)
if wall != 480 or gpu != 960 or phases[-1]["start_minute"] != 450:
    errors.append("Global training/export/time budget mismatch")
for block in campaign["e5"]["blocks"]:
    trials = block["tasks_per_worker"] * block["methods_per_task"] * block["trial_seconds"] / 60
    total = trials + block["proposer_training_minutes"] + block["overhead_minutes"]
    if total != block["wall_minutes"]:
        errors.append(f"E5 block arithmetic mismatch: {block['id']}")
if sum(block["wall_minutes"] for block in campaign["e5"]["blocks"]) != 135:
    errors.append("E5 total differs from phase allocation")
pairs = [(slot[gpu_name]["arm"], slot[gpu_name]["seed"])
         for slot in campaign["e1_slots"] for gpu_name in ("gpu0", "gpu1")]
if sorted(pairs) != sorted((arm, seed) for arm in ("A", "B") for seed in campaign["seeds"]):
    errors.append("E1 does not cover both paired arms/seeds exactly once")
if campaign["owner_authorization"]["local_optimizer_updates_authorized"] != 0:
    errors.append("Unexpected local learning allocation")

files = [ROOT / name for name in campaign["required_design_files"]]
files += [ROOT / name for name in ("AGENTS.md", "README.md", "engineering/README.md",
                                 "engineering/STATUS.md")]
links_checked = 0
for path in files:
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
        if "://" in target or target.startswith("#"):
            continue
        links_checked += 1
        if not (path.parent / target.split("#", 1)[0]).exists():
            errors.append(f"Missing local link: {path.relative_to(ROOT)} -> {target}")
hashes = {}
words = 0
for path in [ROOT / name for name in campaign["required_design_files"]] + [HERE / "campaign.json"]:
    content = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if path.suffix == ".md":
        words += len(content.split())
result = {"kind": "design_validation_not_experiment_result", "errors": errors,
          "wall_minutes": wall, "provisioned_gpu_minutes": gpu,
          "experiment_phases": len(phases), "local_links_checked": links_checked,
          "design_markdown_words": words, "canonical_utf8_lf_sha256": hashes,
          "model_instantiations": 0, "optimizer_updates": 0}
print(json.dumps(result, indent=2))
raise SystemExit(bool(errors))
