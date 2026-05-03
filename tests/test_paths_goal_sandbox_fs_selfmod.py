from __future__ import annotations

from pathlib import Path

from anra_paths import FAISS_INDEX_LOCAL, GHOST_DB_LOCAL, REQUIRED_DIRS, get_dataset_file
from execution.fs_agent import FSAgent
from execution.sandbox import CodeSandbox
from goals.goal_queue import GoalQueue
from self_modification.type_a import ToolLibraryMutation
from self_modification.type_b import AgentCodeMutation


def test_paths_include_canonical_dataset_and_local_memory_dirs() -> None:
    assert get_dataset_file().name in {"anra_training.txt", "anra_dataset_v6_1.txt"}
    assert GHOST_DB_LOCAL.parent in REQUIRED_DIRS
    assert FAISS_INDEX_LOCAL.parent in REQUIRED_DIRS


def test_goal_queue_fail_successor_and_report(tmp_path: Path) -> None:
    queue = GoalQueue(tmp_path / "goals.json")
    item = queue.push("g1", "build", priority=1)
    assert item.metadata == {}
    assert queue.pop().goal_id == "g1"
    assert queue.fail("g1", "retry")
    successor = queue.generate_successor("g1", "build next")
    report = queue.status_report()
    assert successor.parent_id == "g1"
    assert report["total"] == 2


def test_sandbox_strips_secret_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    result = CodeSandbox(tmp_path / "sandbox").execute("import os\nprint(os.environ.get('OPENAI_API_KEY', ''))")
    assert result.success
    assert "secret" not in result.stdout


def test_fs_agent_atomic_write_and_git_commit(tmp_path: Path) -> None:
    agent = FSAgent(tmp_path, tmp_path / "audit.log")
    agent.write("a.txt", "hello")
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "hello"
    code, out = agent.git_commit("initial")
    assert code in {0, 1}
    assert "fatal: not a git repository" not in out.lower()


def test_self_modification_type_a_and_type_b_rollback(tmp_path: Path) -> None:
    mut = ToolLibraryMutation(tmp_path / "registry.json", tmp_path / "tools")
    tool = mut.add_tool("adder", "def run(x=1):\n    return x + 1\n")
    assert Path(tool["path"]).exists()

    target = tmp_path / "agent.py"
    target.write_text("old", encoding="utf-8")
    result = AgentCodeMutation(tmp_path / "backups").mutate(target, "new", benchmark_cmd=["python3", "-c", "raise SystemExit(1)"])
    assert not result["accepted"]
    assert target.read_text(encoding="utf-8") == "old"
