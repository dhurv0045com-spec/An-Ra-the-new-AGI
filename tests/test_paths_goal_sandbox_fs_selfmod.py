from __future__ import annotations

from pathlib import Path

from anra_paths import FAISS_INDEX_LOCAL, GHOST_DB_LOCAL, REQUIRED_DIRS, get_dataset_file
from execution.fs_agent import FSAgent
from execution.sandbox import CodeSandbox
from goals.goal_queue import GoalQueue
from self_modification.type_a import ToolLibraryMutation
from self_modification.type_b import AgentCodeMutation


def test_paths_include_canonical_dataset_and_local_memory_dirs() -> None:
    assert get_dataset_file().name == "anra_training.txt"
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


def test_type_b_propose_method_exists():
    assert hasattr(AgentCodeMutation, "propose")


def test_type_b_propose_accepts_no_improvement(tmp_path: Path):
    target = tmp_path / "agent_logic.py"
    target.write_text("def old(): pass\n", encoding="utf-8")
    mut = AgentCodeMutation(backup_dir=tmp_path)
    result = mut.propose(
        file_path=target,
        new_content="def new(): return 42\n",
        reason="improved function",
    )
    assert result["accepted"] is True
    assert "score_before" in result
    assert "score_after" in result
    assert "diff" in result


def test_type_b_propose_rollback_on_worse(tmp_path: Path):
    class _FakeVerifier:
        call_count = 0

        def score(self, *a, **kw):
            self.call_count += 1

            class _R:
                score = 0.9 if _FakeVerifier.call_count == 1 else 0.5

            return _R()

    class _FakeSandbox:
        def execute(self, code):
            class _R:
                success = True
                return_code = 0
                stdout = "ok"
                stderr = ""

            return _R()

    target = tmp_path / "agent.py"
    original = "def good(): return 'quality'\n"
    target.write_text(original, encoding="utf-8")
    mut = AgentCodeMutation(backup_dir=tmp_path)
    result = mut.propose(
        file_path=target,
        new_content="def bad(): pass\n",
        sandbox=_FakeSandbox(),
        verifier=_FakeVerifier(),
        benchmark_tasks=[{"code": "print(1)", "type": "code"}],
        drop_threshold=0.02,
    )
    assert result["accepted"] is False
    assert target.read_text(encoding="utf-8") == original


def test_type_b_mutate_still_works(tmp_path: Path):
    target = tmp_path / "f.py"
    target.write_text("x = 1\n", encoding="utf-8")
    mut = AgentCodeMutation(backup_dir=tmp_path)
    result = mut.mutate(target, "x = 2\n", reason="update")
    assert result["accepted"] is True
